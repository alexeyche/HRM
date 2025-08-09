from __future__ import annotations

from typing import Any, Dict, Optional, List

from dataset.ast import ASTNodeType
from dataset.encoding import GraphEncoder


class NodeFeatureBuilder:
    """Composable node feature builder with three modes:
    - sum: sum of per-field embeddings/heads (fast, default)
    - concat: concatenate all parts then MLP to d_model
    - attn: attention over parts treated as tokens (1-2 layers), then pool
    """

    def __init__(self, d_model: int = 128, mode: str = "sum", num_attn_heads: int = 4, attn_layers: int = 1):
        try:
            import torch
            import torch.nn as nn
        except Exception as e:  # pragma: no cover
            raise ImportError("PyTorch is required for NodeFeatureBuilder") from e

        self.torch = torch
        self.nn = nn
        self.d_model = d_model
        self.mode = mode

        # Categorical embedding tables (sizes are passed at build time)
        # We lazily build them on first call with provided vocab sizes
        self._built = False
        self.num_attn_heads = num_attn_heads
        self.attn_layers = attn_layers

    def _build(self, vocab_sizes: Dict[str, int]) -> None:
        torch, nn = self.torch, self.nn
        d = self.d_model

        # Embeddings
        self.type_emb = nn.Embedding(vocab_sizes["node_type"], d)
        self.op_emb = nn.Embedding(vocab_sizes["op"], d)
        self.ctx_emb = nn.Embedding(vocab_sizes["ctx"], d)
        self.dt_emb = nn.Embedding(vocab_sizes["dtype"], d)
        self.fn_emb = nn.Embedding(vocab_sizes["fn"], d)
        self.attr_emb = nn.Embedding(vocab_sizes["attr"], d)
        self.var_emb = nn.Embedding(vocab_sizes["var"], d)
        self.const_exact_emb = nn.Embedding(vocab_sizes["const_exact"], d)

        # Numeric heads
        self.const_head = nn.Sequential(nn.Linear(5, d), nn.ReLU(), nn.Linear(d, d))
        self.str_head = nn.Sequential(nn.Linear(1, d), nn.ReLU(), nn.Linear(d, d))
        self.list_head = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d))
        self.pos_head = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d))

        if self.mode == "attn":
            # Attention encoder over parts-as-tokens, then pool
            encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=self.num_attn_heads, dim_feedforward=4 * d, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.attn_layers)
            self.attn_cls = nn.Parameter(torch.zeros(1, 1, d))

        self._built = True

    @staticmethod
    def default_vocab_sizes() -> Dict[str, int]:
        # Use GraphEncoder's vocab to stay in sync by default
        ge = GraphEncoder()
        return ge.vocab_sizes()

    def forward(self, data: Any, vocab_sizes: Optional[Dict[str, int]] = None) -> Any:
        torch, nn = self.torch, self.nn
        if not self._built:
            self._build(vocab_sizes or self.default_vocab_sizes())

        # Clamp negative ids to 0 (PAD/UNK) to avoid embedding errors
        def safe(id_tensor):
            return torch.clamp(id_tensor, min=0)

        parts = []  # type: ignore[var-annotated]

        # Categorical embeddings
        parts.append(self.type_emb(safe(data.node_type)))
        parts.append(self.op_emb(safe(data.op_id)))
        parts.append(self.ctx_emb(safe(data.ctx_id)))
        parts.append(self.dt_emb(safe(data.dtype_id)))
        parts.append(self.fn_emb(safe(data.function_name_id)))
        parts.append(self.attr_emb(safe(data.attribute_name_id)))
        parts.append(self.var_emb(safe(data.var_id)))

        # Optional exact-const id
        parts.append(self.const_exact_emb(safe(data.const_exact_int_id)))

        # Numeric heads
        parts.append(self.const_head(data.const_numeric.float()))
        parts.append(self.str_head(data.str_numeric.float()))
        parts.append(self.list_head(data.list_summary.float()))
        parts.append(self.pos_head(data.position.float()))

        if self.mode == "sum":
            h = torch.stack(parts, dim=0).sum(dim=0)
            return h

        if self.mode == "concat":
            h = torch.cat(parts, dim=-1)
            d = self.d_model
            in_features = h.shape[-1]
            # lazily create projection matching current number of parts
            if not hasattr(self, "concat_proj") or getattr(self, "_concat_in", None) != in_features:
                self.concat_proj = self.nn.Sequential(self.nn.Linear(in_features, d), self.nn.ReLU(), self.nn.Linear(d, d))
                self._concat_in = in_features
            h = self.concat_proj(h)
            return h

        if self.mode == "attn":
            # treat each part as token; shape [B, T, d]
            tokens = torch.stack(parts, dim=1)
            cls = self.attn_cls.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            enc = self.encoder(tokens)
            h = enc[:, 0]  # CLS pooled
            return h

        raise ValueError(f"Unknown feature builder mode: {self.mode}")


__all__ = ["NodeFeatureBuilder"]


