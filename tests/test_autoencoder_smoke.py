import torch
import torch.nn.functional as F

from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, collate_to_pyg_batch
from dataset.grammar_actions import simplified_ast_to_graph, graph_to_actions, ActionKind
from models.autoencoder import ProgramAutoencoder


def _codes():
    return [
        """
def program(n):
    return n + 1
""",
        """
def program(a, b):
    x = a
    y = b
    return x
""",
    ]


def _actions_to_ids(ae: ProgramAutoencoder, actions):
    return [ae.action_to_id[a.kind] for a in actions]


def test_autoencoder_smoke_forward_backward():
    # Build graphs
    enc = GraphEncoder()
    graphs = [enc.encode(ASTSimplifier.ast_to_graph(code)) for code in _codes()]
    batch = collate_to_pyg_batch(graphs)

    # Build action ids per sample using simplified graph
    action_id_seqs = []
    for code in _codes():
        simp = simplified_ast_to_graph(code)
        actions = graph_to_actions(simp)
        ids = _actions_to_ids(ProgramAutoencoder(), actions)  # temp instance for mapping
        action_id_seqs.append(ids)

    # Model
    ae = ProgramAutoencoder(d_model=64, num_gnn_layers=2, decoder_layers=1, num_heads=4)

    # Prepare teacher tokens (BOS at t=0)
    action_ids = ae.prepare_teacher_tokens(action_id_seqs)

    # Forward
    out = ae.forward(batch, action_ids)
    logits = out["logits"]

    # Shapes
    B, T = action_ids.shape
    assert logits.shape[:2] == (B, T)
    assert logits.shape[-1] == ae.num_actions

    # Labels and loss: ignore BOS positions, and any rows where mask is all invalid
    labels = action_ids.clone()
    labels[labels == ae.num_actions] = -100

    valid_mask = ae.build_valid_action_masks(action_ids)
    rows_valid = valid_mask.any(-1) & (labels != -100)
    # If no valid rows (unlikely), skip loss
    if rows_valid.any():
        logits_sel = logits[rows_valid]
        labels_sel = labels[rows_valid]
        loss = F.cross_entropy(logits_sel, labels_sel.long(), ignore_index=-100)
        loss.backward()


