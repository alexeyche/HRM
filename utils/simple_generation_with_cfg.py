# gpt_decoder_with_cfg.py
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Tiny LL(1) arithmetic grammar
# -----------------------------
# Nonterminals: E, E', T, T', F, NUM
# Terminals: '0'..'9', '+', '*', '(', ')', '<eos>'
# We build a predictive parsing table and a simulator that decides which terminals are valid next.

EPS = "ε"
END = "$"

TERMINALS = [str(d) for d in range(10)] + ['+', '*', '(', ')', '<eos>']
NONTERMINALS = ["E", "E'", "T", "T'", "F", "NUM"]


# Productions (for reference / FIRST/FOLLOW construction if you want to extend)
PRODUCTIONS = {
    "E":  [["T", "E'"]],
    "E'": [['+', "T", "E'"], [EPS]],
    "T":  [["F", "T'"]],
    "T'": [['*', "F", "T'"], [EPS]],
    "F":  [['(', "E", ')'], ["NUM"]],
    "NUM":[["DIGIT", "NUM"], ["DIGIT"]],
    # DIGIT is a pseudo-nonterminal that expands to any digit terminal; handled specially in the parser
}

# FIRST sets (pre-baked for simplicity)
FIRST = {
    "E":  set(['(', 'DIGIT']),
    "E'": set(['+']) | set([EPS]),
    "T":  set(['(', 'DIGIT']),
    "T'": set(['*']) | set([EPS]),
    "F":  set(['(', 'DIGIT']),
    "NUM":set(['DIGIT']),
    "DIGIT": set([str(d) for d in range(10)]),
}

# FOLLOW sets (only those we need)
FOLLOW = {
    "E":  set([')', '<eos>']),
    "E'": set([')', '<eos>']),
    "T":  set(['+', ')', '<eos>']),
    "T'": set(['+', ')', '<eos>']),
    "F":  set(['*', '+', ')', '<eos>']),
    "NUM":set(['*', '+', ')', '<eos>']),
}

# Predictive parsing table (LL(1)).
# Keys: (nonterminal, lookahead_terminal) -> RHS (list of symbols; [] means ε)
PARSING_TABLE: Dict[Tuple[str, str], List[str]] = {}

def _add_table(nt: str, looks: Set[str], rhs: List[str]):
    for a in looks:
        PARSING_TABLE[(nt, a)] = rhs

# E → T E'
_add_table("E", FIRST["T"], ["T", "E'"])  # FIRST(T) == {'(', 'DIGIT'}

# E' → + T E' | ε
_add_table("E'", {'+'}, ['+', 'T', "E'"])
# For ε, any a in FOLLOW(E') is valid
_add_table("E'", FOLLOW["E'"], [])

# T → F T'
_add_table("T", FIRST["F"], ["F", "T'"])

# T' → * F T' | ε
_add_table("T'", {'*'}, ['*', 'F', "T'"])
_add_table("T'", FOLLOW["T'"], [])

# F → ( E ) | NUM
_add_table("F", {'('}, ['(', 'E', ')'])
_add_table("F", FIRST["NUM"], ['NUM'])  # lookahead 'DIGIT'

# NUM → DIGIT NUM | DIGIT
# We handle DIGIT specially: on lookahead digit, choose appropriate production
_add_table("NUM", set([str(d) for d in range(10)]), ['DIGIT', 'NUM'])  # left-recursive expansion via right recursion is fine

# allow NUM → ε when the lookahead is in FOLLOW(NUM)
_add_table("NUM", FOLLOW["NUM"], [])

class ParserState:
    """LL(1) predictive parser state for the arithmetic grammar above."""
    def __init__(self):
        self.stack: List[str] = [END, "E"]  # Start symbol E

    def clone(self) -> "ParserState":
        c = ParserState()
        c.stack = self.stack.copy()
        return c

    def _top(self) -> str:
        return self.stack[-1] if self.stack else END

    def _pop(self):
        if self.stack:
            self.stack.pop()

    def _push(self, symbols: List[str]):
        for s in reversed(symbols):
            if s == EPS or s == []:
                continue
            self.stack.append(s)

    @staticmethod
    def _is_terminal(sym: str) -> bool:
        return sym in TERMINALS or sym in ['+', '*', '(', ')'] or sym == '<eos>' or sym.isdigit()

    def _expand_nonterminal(self, nt: str, lookahead: str) -> bool:
        # Handle DIGIT FIRST logic: map lookahead digit to 'DIGIT' class
        lh = lookahead
        if lookahead in [str(d) for d in range(10)]:
            lh = lookahead  # exact digit; table already added for NUM with digits
        key = (nt, 'DIGIT') if (nt, 'DIGIT') in PARSING_TABLE and lookahead.isdigit() else (nt, lh)
        # If not found, try ε via FOLLOW set (already added in table)
        rhs = PARSING_TABLE.get(key, None)
        if rhs is None:
            return False
        # Special-case: for NUM on digit, we might choose ['DIGIT'] as an alternative
        if nt == "NUM" and lookahead.isdigit():
            # Allow either ['DIGIT','NUM'] or ['DIGIT']; both are valid;
            # prefer ['DIGIT','NUM'] to allow multi-digit numbers during generation.
            rhs = ['DIGIT', 'NUM']
        self._pop()
        self._push(rhs)
        return True

    def would_accept(self, candidate: str) -> bool:
        """Return True if seeing 'candidate' next is grammar-valid from current stack."""
        st = self.clone()

        # Try to reduce/expand until a terminal is on top or END reached
        while True:
            top = st._top()
            # Accepting EOS?
            if top == END:
                # Only <eos> is allowed when stack is at END
                return candidate == '<eos>'
            # If top is nonterminal, expand using table and candidate lookahead
            if top in NONTERMINALS:
                ok = st._expand_nonterminal(top, candidate)
                if not ok:
                    return False
                continue
            # Pseudo-nonterminal DIGIT
            if top == "DIGIT":
                if candidate.isdigit():
                    return True
                else:
                    return False
            # Real terminal
            if self._is_terminal(top):
                # If the next required terminal equals candidate, it's valid to consume it
                return top == candidate

    def allowed_next_terminals(self) -> Set[str]:
        """Brute-force: test each terminal and keep the ones that would be accepted."""
        allowed = set()
        for t in TERMINALS:
            if self.would_accept(t):
                allowed.add(t)
        return allowed

    def consume(self, tok: str):
        """Advance the parser by actually consuming 'tok' (assumes tok is valid)."""
        while True:
            top = self._top()
            if top == END:
                # Only <eos> should arrive here; nothing to do
                return
            if top in NONTERMINALS:
                self._expand_nonterminal(top, tok)
                continue
            if top == "DIGIT":
                assert tok.isdigit(), f"Expected digit, got {tok}"
                self._pop()
                return
            # terminal
            if top == tok:
                self._pop()
                return
            else:
                # If mismatch happens, try ε-expansions already baked into table usage;
                # would_accept should prevent us from getting here in practice.
                raise ValueError(f"Parser mismatch: top={top}, tok={tok}")


# -----------------------------
# GPT-like decoder head
# -----------------------------
@dataclass
class GPTConfig:
    vocab: List[str]
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 4 * 384
    max_seq_len: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer("mask", None, persistent=False)

    def _causal_mask(self, T: int, device):
        # [1, 1, T, T]
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,h,T,d)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,h,T,T)
        mask = self._causal_mask(T, x.device)
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B,h,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.out(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTDecoderWithCFG(nn.Module):
    """
    A decoder-only Transformer head that:
      * conditions on an external embedding x (shape [B, d_cond])
      * generates tokens from a tiny vocab (digits + + * ( ) + <bos>/<eos>/<pad>)
      * uses an LL(1) parser to mask logits so outputs always satisfy the grammar
    """
    def __init__(self, cfg: GPTConfig, d_cond: int):
        super().__init__()
        self.cfg = cfg

        # Vocabulary & token indices
        self.itos = cfg.vocab
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

        self.tok_emb = nn.Embedding(len(cfg.vocab), cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, len(cfg.vocab), bias=False)

        # Conditioning: project external embedding to a "prefix token" in the same space
        self.cond_proj = nn.Linear(d_cond, cfg.d_model)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, target_ids: torch.Tensor, cond_x: torch.Tensor):
        """
        target_ids: (B, T) target tokens (does NOT include <bos>)
                    During training, the model predicts the next token at each step.
        cond_x: (B, d_cond) conditioning vectors

        Returns: logits (B, T+1, V)
                The +1 comes from prepending the cond_x embedding as token 0.
        """
        B, T = target_ids.shape
        if T + 1 > self.cfg.max_seq_len:
            raise ValueError("Sequence too long for configured max_seq_len")

        # Project cond_x to embedding space and treat it as first token
        cond_tok = self.cond_proj(cond_x).unsqueeze(1)  # (B, 1, C)

        # Embed target tokens for positions 1..T
        tok_emb = self.tok_emb(target_ids)              # (B, T, C)

        # Concatenate [cond_token, target_tokens]
        x = torch.cat([cond_tok, tok_emb], dim=1)       # (B, T+1, C)

        # Add positional embeddings
        pos_ids = torch.arange(T + 1, device=target_ids.device).unsqueeze(0)  # (1, T+1)
        pos_emb = self.pos_emb(pos_ids)
        x = x + pos_emb

        # Pass through transformer blocks
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T+1, V)
        return logits

    # ---------- grammar mask ----------
    def _allowed_token_ids(self, parser: ParserState) -> List[int]:
        allowed_terms = parser.allowed_next_terminals()  # set of terminals including '<eos>'
        # Map to vocab ids, ignoring any terminals not in vocab (shouldn't happen)
        ids = []
        for t in allowed_terms:
            if t in self.stoi:
                ids.append(self.stoi[t])
            elif t in [str(d) for d in range(10)]:
                ids.append(self.stoi[t])  # digits are in vocab
        return sorted(ids)

    def _mask_with_grammar(self, logits: torch.Tensor, allowed_ids: List[int]):
        """Set logits for disallowed tokens to -inf, keep allowed as-is."""
        mask = torch.full_like(logits, float('-inf'))
        mask[..., allowed_ids] = logits[..., allowed_ids]
        return mask

    # ---------- generation (inference) ----------
    @torch.no_grad()
    def generate(
        self,
        cond_x: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
        device: Optional[torch.device] = None,
    ) -> List[str]:
        """
        Generates grammar-valid sequences conditioned only on cond_x.
        """
        device = device or cond_x.device
        B = cond_x.size(0)

        # No token IDs yet — only conditioning embedding at position 0
        seq = torch.zeros((B, 0), dtype=torch.long, device=device)

        # One parser per batch element
        parsers = [ParserState() for _ in range(B)]

        # Prepare first hidden state from cond_x
        cond_tok = self.cond_proj(cond_x).unsqueeze(1)  # (B,1,C)
        pos_emb = self.pos_emb(torch.zeros(1, dtype=torch.long, device=device)).unsqueeze(0)  # (1,1,C)
        x = cond_tok + pos_emb

        # Autoregressive loop
        for step in range(max_new_tokens):
            # Run transformer starting from cond_token + token embeddings
            if seq.size(1) > 0:
                tok_emb = self.tok_emb(seq)  # (B,T,C)
                pos_ids = torch.arange(1, seq.size(1) + 1, device=device).unsqueeze(0)  # (1,T)
                pos_emb_rest = self.pos_emb(pos_ids)
                x_full = torch.cat([x, tok_emb + pos_emb_rest], dim=1)
            else:
                x_full = x  # Only cond_token so far

            h = self.drop(x_full)
            for blk in self.blocks:
                h = blk(h)
            h = self.ln_f(h)

            logits = self.lm_head(h)[:, -1, :]  # (B, V) last position only

            # Grammar-constrained token selection for each batch element
            new_token_ids = []
            for b in range(B):
                allowed_ids = self._allowed_token_ids(parsers[b])
                masked = self._mask_with_grammar(logits[b], allowed_ids)
                if temperature != 1.0:
                    masked = masked / max(1e-8, temperature)
                probs = F.softmax(masked, dim=-1)
                if top_k is not None:
                    topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.numel()))
                    filtered = torch.full_like(probs, 0.0)
                    filtered[topk_idx] = topk_vals
                    probs = filtered / (filtered.sum() + 1e-9)
                if greedy:
                    next_id = torch.argmax(probs).item()
                else:
                    next_id = torch.multinomial(probs, num_samples=1).item()
                new_token_ids.append(next_id)

            next_ids = torch.tensor(new_token_ids, device=device).unsqueeze(1)  # (B,1)

            # Update parser states
            for b in range(B):
                tok_str = self.itos[next_ids[b, 0].item()]
                if tok_str == "<eos>":
                    pass
                else:
                    parsers[b].consume(tok_str)

            seq = torch.cat([seq, next_ids], dim=1)

            # Early stop if all hit <eos>
            if torch.all(next_ids.squeeze(1) == self.eos_id):
                break

        # Convert token ids to strings
        outputs: List[str] = []
        for b in range(B):
            toks = [self.itos[i] for i in seq[b].tolist()]
            out = []
            for t in toks:
                if t == "<eos>":
                    break
                out.append(t)
            outputs.append("".join(out))
        return outputs


# -----------------------------
# Convenience: default vocab + demo
# -----------------------------
def default_vocab() -> List[str]:
    specials = ["<pad>", "<bos>", "<eos>"]
    digits = [str(d) for d in range(10)]
    ops = ['+', '*', '(', ')']
    return specials + digits + ops


if __name__ == "__main__":
    # Minimal smoke test (cpu)
    vocab = default_vocab()
    cfg = GPTConfig(vocab=vocab, d_model=256, n_heads=4, n_layers=4, max_seq_len=64, dropout=0.1)
    model = GPTDecoderWithCFG(cfg, d_cond=32)

    B = 2
    cond_x = torch.randn(B, 32)

    # Fake training batch
    # Example target strings that satisfy the grammar: "2+3*(4+5)"
    targets = ["2+3*(4+5)", "(1+2)*3"]

    batch = torch.zeros(B, 0, dtype=torch.long, device=cond_x.device)

    logits = model(batch, cond_x)  # (B,T,V)

    # Grammar-constrained generation
    outs = model.generate(cond_x, max_new_tokens=32, temperature=0.8, top_k=8, greedy=False)
    for i, s in enumerate(outs):
        print(f"[{i}] {s}")
