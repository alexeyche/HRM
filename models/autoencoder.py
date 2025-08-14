from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from models.features import NodeFeatureBuilder
from dataset.ast import ASTNodeType, EdgeType
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn import GCNConv

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

        # Action vocab
        self.ast_list: List[ASTNodeType] = list(ASTNodeType)
        self.ast_to_id: Dict[ASTNodeType, int] = {k: i for i, k in enumerate(self.ast_list)}
        self.num_ast: int = len(self.ast_list)

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
        self.token_emb = nn.Embedding(self.num_ast + 1, d_model)  # +1 for BOS id at index num_ast
        self.pos_emb = nn.Embedding(1024, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Output head
        self.out_proj = nn.Linear(d_model, self.num_ast)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    # note: NodeFeatureBuilder is not an nn.Module; its parameters (embeddings/MLPs) will not be registered
    # for optimization. For a quick smoke autoencoder this is acceptable; swap to a Module variant later if needed.

    # Helper: build masks from teacher tokens using ASTNodeType
    def build_valid_ast_masks(self, batch_ast_ids: Any) -> Any:
        """Return mask tensor [B, T, num_ast] where True = valid.

        Expects batch_ast_ids of shape [B, T] with values in [0, num_ast] where value == num_ast denotes BOS.
        The mask at timestep t corresponds to valid next actions after applying prefix up to t-1 (ignores BOS tokens in state).
        """

        B, T = batch_ast_ids.shape
        masks = torch.zeros((B, T, self.num_ast), dtype=torch.bool, device=batch_ast_ids.device)


        return masks

    def _encode_graph(self, batch: Any) -> Tuple[Any, Any]:
        """Encode graph nodes -> [N, d], then mean-pool per graph to [B, d]."""
        x = self.feature_builder.forward(batch)
        for conv in self.gnn_layers:
            x = F.relu(conv(x, batch.edge_index))
        x = self.gnn_norm(x)
        # batch.batch exists on PyG Batch; if missing, treat as single graph of all nodes
        if hasattr(batch, "batch") and batch.batch is not None:
            graph_repr = global_mean_pool(x, batch.batch)
        else:
            graph_repr = x.mean(dim=0, keepdim=True)
        graph_repr = self.graph_proj(graph_repr)
        return x, graph_repr

    def forward(
        self,
        batch: Any,
        ast_ids: Any,
        valid_ast_mask: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Teacher-forced forward pass.

        - batch: PyG Batch from encoding.collate_to_pyg_batch
        - ast_ids: LongTensor [B, T] with tokens in [0..num_ast] and BOS as num_ast
        - valid_ast_mask: Optional BoolTensor [B, T, num_ast]; if None, computed on-the-fly
        Returns dict with 'logits' [B, T, num_ast]
        """

        # Encode graph
        _, graph_ctx = self._encode_graph(batch)  # [B, d]

        B, T = ast_ids.shape
        device = graph_ctx.device

        # Build input embeddings (shifted right by 1 inside caller; here we embed given ids as-is)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_emb(ast_ids)
        pos_emb = self.pos_emb(positions)
        dec_inp = self.dropout(tok_emb + pos_emb)  # [B, T, d]

        # Memory from graph context: use 1 token per graph
        memory = graph_ctx.unsqueeze(1)  # [B, 1, d]

        # Causal mask for decoder (allow attending to previous positions only)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        dec_out = self.decoder(
            tgt=dec_inp,
            memory=memory,
            tgt_mask=causal_mask,
        )  # [B, T, d]

        logits = self.out_proj(dec_out)  # [B, T, num_actions]

        # Apply grammar mask if provided/compute
        if valid_ast_mask is None:
            mask = self.build_valid_ast_masks(ast_ids)
        else:
            # Trust caller; ensure boolean dtype
            mask = valid_ast_mask

        # Mask invalid actions by setting logits to large negative
        invalid = torch.logical_not(mask)
        logits = logits.masked_fill(invalid, float("-inf"))

        return {"logits": logits}



__all__ = ["ProgramAutoencoder"]


