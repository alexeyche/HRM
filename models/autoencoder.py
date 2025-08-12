from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


class ProgramAutoencoder(nn.Module):
    """Grammar-masked autoencoder:

    - Graph encoder: NodeFeatureBuilder + lightweight GNN + mean readout -> [B, d_model]
    - Action decoder: Transformer decoder over action tokens with cross-attn to graph context
    - Grammar masking: masks invalid actions per time-step using ParserState from dataset.grammar_actions

    Forward signature expects PyG Batch and teacher-forced action ids.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_gnn_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        node_feature_mode: str = "sum",
    ) -> None:
        super().__init__()
        try:
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv, global_mean_pool
        except Exception as e:  # pragma: no cover
            raise ImportError("PyTorch and PyG are required for ProgramAutoencoder") from e

        from models.features import NodeFeatureBuilder
        from dataset.grammar_actions import ActionKind

        self.torch = torch
        self.nn = nn
        self.F = F
        self.global_mean_pool = global_mean_pool
        self.ActionKind = ActionKind

        # Action vocab
        self.action_list: List[ActionKind] = list(ActionKind)
        self.action_to_id: Dict[ActionKind, int] = {k: i for i, k in enumerate(self.action_list)}
        self.num_actions: int = len(self.action_list)

        # Node features and GNN
        self.feature_builder = NodeFeatureBuilder(d_model=d_model, mode=node_feature_mode)
        gnn_layers: List[nn.Module] = []
        in_dim = d_model
        for _ in range(num_gnn_layers):
            gnn_layers.append(GCNConv(in_dim, d_model))
            in_dim = d_model
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.gnn_norm = nn.LayerNorm(d_model)

        # Encoder projection for graph context
        self.graph_proj = nn.Linear(d_model, d_model)

        # Decoder token embedding and positional embedding
        self.token_emb = nn.Embedding(self.num_actions + 1, d_model)  # +1 for BOS id at index num_actions
        self.pos_emb = nn.Embedding(1024, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Output head
        self.out_proj = nn.Linear(d_model, self.num_actions)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    # note: NodeFeatureBuilder is not an nn.Module; its parameters (embeddings/MLPs) will not be registered
    # for optimization. For a quick smoke autoencoder this is acceptable; swap to a Module variant later if needed.

    # Helper: build masks from teacher tokens using ParserState
    def build_valid_action_masks(self, batch_action_ids: Any) -> Any:
        """Return mask tensor [B, T, num_actions] where True = valid.

        Expects batch_action_ids of shape [B, T] with values in [0, num_actions] where value == num_actions denotes BOS.
        The mask at timestep t corresponds to valid next actions after applying prefix up to t-1 (ignores BOS tokens in state).
        """
        import torch
        from dataset.grammar_actions import Action, ParserState, create_action_mask

        B, T = batch_action_ids.shape
        masks = torch.zeros((B, T, self.num_actions), dtype=torch.bool, device=batch_action_ids.device)

        for b in range(B):
            # Start with fresh parser
            state = ParserState()
            for t in range(T):
                # Compute valid set BEFORE consuming token t
                valid = create_action_mask(state)
                # Map to vector
                for i, kind in enumerate(self.action_list):
                    masks[b, t, i] = bool(valid.get(kind, False))

                # Advance state with current teacher token if it's not BOS padding
                tok = int(batch_action_ids[b, t].item())
                if tok == self.num_actions:  # BOS id
                    continue
                # Apply action; attribute setters leave stack unchanged inside ParserState
                kind = self.action_list[tok]
                try:
                    state.apply_action(Action(kind))
                except Exception:
                    # If teacher token is malformed relative to our simplified parser, keep state unchanged
                    pass

        return masks

    def _encode_graph(self, batch: Any) -> Tuple[Any, Any]:
        """Encode graph nodes -> [N, d], then mean-pool per graph to [B, d]."""
        x = self.feature_builder.forward(batch)
        for conv in self.gnn_layers:
            x = self.F.relu(conv(x, batch.edge_index))
        x = self.gnn_norm(x)
        # batch.batch exists on PyG Batch; if missing, treat as single graph of all nodes
        if hasattr(batch, "batch") and batch.batch is not None:
            graph_repr = self.global_mean_pool(x, batch.batch)
        else:
            graph_repr = x.mean(dim=0, keepdim=True)
        graph_repr = self.graph_proj(graph_repr)
        return x, graph_repr

    def forward(
        self,
        batch: Any,
        action_ids: Any,
        valid_action_mask: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Teacher-forced forward pass.

        - batch: PyG Batch from encoding.collate_to_pyg_batch
        - action_ids: LongTensor [B, T] with tokens in [0..num_actions] and BOS as num_actions
        - valid_action_mask: Optional BoolTensor [B, T, num_actions]; if None, computed on-the-fly
        Returns dict with 'logits' [B, T, num_actions]
        """
        import torch

        # Encode graph
        _, graph_ctx = self._encode_graph(batch)  # [B, d]

        B, T = action_ids.shape
        device = graph_ctx.device

        # Build input embeddings (shifted right by 1 inside caller; here we embed given ids as-is)
        positions = self.torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_emb(action_ids)
        pos_emb = self.pos_emb(positions)
        dec_inp = self.dropout(tok_emb + pos_emb)  # [B, T, d]

        # Memory from graph context: use 1 token per graph
        memory = graph_ctx.unsqueeze(1)  # [B, 1, d]

        # Causal mask for decoder (allow attending to previous positions only)
        causal_mask = self.torch.triu(self.torch.ones(T, T, device=device, dtype=self.torch.bool), diagonal=1)

        dec_out = self.decoder(
            tgt=dec_inp,
            memory=memory,
            tgt_mask=causal_mask,
        )  # [B, T, d]

        logits = self.out_proj(dec_out)  # [B, T, num_actions]

        # Apply grammar mask if provided/compute
        if valid_action_mask is None:
            mask = self.build_valid_action_masks(action_ids)
        else:
            # Trust caller; ensure boolean dtype
            mask = valid_action_mask

        # Mask invalid actions by setting logits to large negative
        invalid = self.torch.logical_not(mask)
        logits = logits.masked_fill(invalid, float("-inf"))

        return {"logits": logits}

    # Utility to prepare teacher tokens (add BOS id at position 0 and shift)
    def prepare_teacher_tokens(self, actions_per_sample: List[List[int]], max_len: Optional[int] = None) -> Any:
        """Convert ragged lists of action ids to a padded tensor with BOS at t=0.

        Returns LongTensor [B, T] where BOS id is self.num_actions.
        """
        import torch

        if max_len is None:
            max_len = max(len(seq) for seq in actions_per_sample) + 1  # +1 for BOS

        B = len(actions_per_sample)
        bos_id = self.num_actions
        out = torch.full((B, max_len), fill_value=bos_id, dtype=torch.long)

        for b, seq in enumerate(actions_per_sample):
            # Place BOS then sequence (possibly truncated/padded)
            length = min(len(seq), max_len - 1)
            out[b, 0] = bos_id
            if length > 0:
                out[b, 1 : 1 + length] = torch.tensor(seq[:length], dtype=torch.long)

        return out


__all__ = ["ProgramAutoencoder"]


