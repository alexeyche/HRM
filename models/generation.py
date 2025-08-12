from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto

import torch
from torch import nn

from dataset.grammar_actions import Action, ActionKind, ParserState, ParserRole, create_action_mask


class DecodingStage(Enum):
    START = auto()
    AFTER_FN = auto()
    RET = auto()
    RET_EXPR_VAR = auto()
    RET_EXPR_CONST = auto()
    AFTER_MINIMAL = auto()


class DecodingState:
    """Lightweight stream protocol for grammar actions to ensure valid decoding order.

    This constrains generation to a minimal valid program:
      DEF → RETURN → (VARIABLE|CONST_INT) → required SET_* → EOS

    It is intentionally simple to guarantee syntactic validity for smoke tests
    without depending on learned logits.
    """

    def __init__(self, action_to_id: Dict[ActionKind, int]) -> None:
        self.action_to_id = action_to_id
        self.started: bool = False
        self.ended: bool = False
        self.stage: DecodingStage = DecodingStage.START
        self.required_next_attrs: set[ActionKind] = set()
        self.pending_expr_slots: int = 0  # number of expressions to emit (operands/attrs)
        self.last_prod_kind: Optional[ActionKind] = None
        self.next_required_sequence: List[ActionKind] = []

    def get_allowed_kinds(self) -> set[ActionKind]:
        if self.ended:
            return {ActionKind.EOS}
        if self.required_next_attrs:
            return set(self.required_next_attrs)
        if self.next_required_sequence:
            return {self.next_required_sequence[0]}
        if self.pending_expr_slots > 0:
            # Only allow leaf expressions to satisfy outstanding slots safely
            return {
                ActionKind.PROD_VARIABLE,
                ActionKind.PROD_CONSTANT_INT,
                ActionKind.PROD_CONSTANT_BOOL,
                ActionKind.PROD_CONSTANT_STR,
            }
        if self.stage == DecodingStage.START:
            return {ActionKind.PROD_FUNCTION_DEF}
        if self.stage == DecodingStage.AFTER_FN:
            return {ActionKind.PROD_RETURN}
        if self.stage == DecodingStage.RET:
            # Allow only simple expressions at top-level for stability
            return {
                ActionKind.PROD_VARIABLE,
                ActionKind.PROD_CONSTANT_INT,
                ActionKind.PROD_CONSTANT_BOOL,
                ActionKind.PROD_CONSTANT_STR,
            }
        if self.stage in (DecodingStage.RET_EXPR_VAR, DecodingStage.RET_EXPR_CONST):
            # Should be handled by required_next_attrs
            return set(self.required_next_attrs)
        if self.stage == DecodingStage.AFTER_MINIMAL:
            return {ActionKind.EOS}
        # Fallback: allow EOS to prevent deadlock
        return {ActionKind.EOS}

    def apply(self, kind: ActionKind) -> None:
        if self.ended:
            return
        if kind == ActionKind.EOS:
            self.ended = True
            return
        if kind.name.startswith("SET_"):
            if kind in self.required_next_attrs:
                self.required_next_attrs.remove(kind)
                # Handle leaf completions and structural counts
                if kind == ActionKind.SET_VAR_ID and self.stage == DecodingStage.RET_EXPR_VAR:
                    if self.pending_expr_slots > 0:
                        self.pending_expr_slots -= 1
                        if self.pending_expr_slots == 0:
                            self.stage = DecodingStage.AFTER_MINIMAL
                    else:
                        self.stage = DecodingStage.AFTER_MINIMAL
                elif kind in {ActionKind.SET_CONST_INT, ActionKind.SET_CONST_BOOL, ActionKind.SET_CONST_STR} and self.stage == DecodingStage.RET_EXPR_CONST:
                    if self.pending_expr_slots > 0:
                        self.pending_expr_slots -= 1
                        if self.pending_expr_slots == 0:
                            self.stage = DecodingStage.AFTER_MINIMAL
                    else:
                        self.stage = DecodingStage.AFTER_MINIMAL
                elif kind == ActionKind.SET_OP:
                    # After setting op, schedule operands based on last production
                    if self.last_prod_kind in {ActionKind.PROD_BINARY_OP, ActionKind.PROD_COMPARISON, ActionKind.PROD_BOOLEAN_OP}:
                        self.pending_expr_slots += 2
                    elif self.last_prod_kind == ActionKind.PROD_UNARY_OP:
                        self.pending_expr_slots += 1
                elif kind == ActionKind.SET_ATTRIBUTE_NAME:
                    self.pending_expr_slots += 1
                    # Attribute now needs its object expression
                    self.next_required_sequence = []
                # SET_FUNCTION_NAME does not change slots; SET_PARAMS ignored here
            elif self.next_required_sequence and kind == self.next_required_sequence[0]:
                # Enforce ordered sequence
                self.next_required_sequence.pop(0)
                # For function call or others we are not handling ordering anymore
            return
        # Non-SET productions
        if self.stage == DecodingStage.START and kind == ActionKind.PROD_FUNCTION_DEF:
            self.started = True
            self.stage = DecodingStage.AFTER_FN
            return
        if self.stage == DecodingStage.AFTER_FN and kind == ActionKind.PROD_RETURN:
            self.stage = DecodingStage.RET
            return
        if (self.stage == DecodingStage.RET or self.pending_expr_slots > 0) and kind == ActionKind.PROD_VARIABLE:
            self.stage = DecodingStage.RET_EXPR_VAR
            self.required_next_attrs = {ActionKind.SET_VAR_ID}
            self.last_prod_kind = kind
            return
        if (self.stage == DecodingStage.RET or self.pending_expr_slots > 0) and kind == ActionKind.PROD_CONSTANT_INT:
            self.stage = DecodingStage.RET_EXPR_CONST
            self.required_next_attrs = {ActionKind.SET_CONST_INT}
            self.last_prod_kind = kind
            return
        if (self.stage == DecodingStage.RET or self.pending_expr_slots > 0) and kind == ActionKind.PROD_CONSTANT_BOOL:
            self.stage = DecodingStage.RET_EXPR_CONST
            self.required_next_attrs = {ActionKind.SET_CONST_BOOL}
            self.last_prod_kind = kind
            return
        if (self.stage == DecodingStage.RET or self.pending_expr_slots > 0) and kind == ActionKind.PROD_CONSTANT_STR:
            self.stage = DecodingStage.RET_EXPR_CONST
            self.required_next_attrs = {ActionKind.SET_CONST_STR}
            self.last_prod_kind = kind
            return
        if (self.stage == DecodingStage.RET or self.pending_expr_slots > 0) and kind in {
            ActionKind.PROD_BINARY_OP,
            ActionKind.PROD_UNARY_OP,
            ActionKind.PROD_COMPARISON,
            ActionKind.PROD_BOOLEAN_OP,
        }:
            self.required_next_attrs = {ActionKind.SET_OP}
            self.last_prod_kind = kind
            return
        # Exclude FUNCTION_CALL and LIST handling until we add arg/list tracking
        if (self.stage == DecodingStage.RET or self.pending_expr_slots > 0) and kind == ActionKind.PROD_ATTRIBUTE:
            self.next_required_sequence = [ActionKind.SET_ATTRIBUTE_NAME]
            self.last_prod_kind = kind
            return
        # Otherwise, keep current stage and let grammar shape next options


class ProgramGenerationHead(nn.Module):
    """Decoder head that predicts action kinds and values given graph context and previous actions.

    - Kind head: Transformer decoder over action tokens (kinds) with cross-attn to graph context
    - Value heads: Per-SET_* classifiers/regressors mapped to known, finite vocabularies
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 4,
        decoder_layers: int = 2,
        op_vocab_size: int,
        fn_vocab_size: int,
        attr_vocab_size: int,
        small_int_vocab_size: int,
        max_var_vocab_size: int,
        num_actions: int,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_actions = num_actions

        # Tokens: action kinds (+1 for BOS)
        self.token_emb = nn.Embedding(num_actions + 1, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.kind_head = nn.Linear(d_model, num_actions)

        # Value heads
        self.op_head = nn.Linear(d_model, op_vocab_size)
        self.var_id_head = nn.Linear(d_model, max_var_vocab_size)
        self.bool_head = nn.Linear(d_model, 2)
        self.fn_name_head = nn.Linear(d_model, fn_vocab_size)
        self.attr_name_head = nn.Linear(d_model, attr_vocab_size)
        self.small_int_head = nn.Linear(d_model, small_int_vocab_size)

        # Length heads (bounded small integers)
        self.arg_len_head = nn.Linear(d_model, 6)    # 0..5
        self.list_len_head = nn.Linear(d_model, 6)   # 0..5
        self.block_len_head = nn.Linear(d_model, 6)  # 0..5

    def forward_step(self, memory: torch.Tensor, prev_kind_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute decoder outputs at next position.

        memory: [B, 1, d]
        prev_kind_ids: [B, T] with BOS as num_actions
        Returns: (kind_logits [B, T, A], dec_hidden [B, T, d])
        """
        B, T = prev_kind_ids.shape
        device = prev_kind_ids.device

        tok = self.token_emb(prev_kind_ids)
        pos = self.pos_emb(torch.arange(T, device=device).unsqueeze(0).expand(B, T))
        tgt = tok + pos

        tgt_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        kind_logits = self.kind_head(dec_out)
        return kind_logits, dec_out

    def predict_value_from_hidden(self, hidden: torch.Tensor, kind: ActionKind) -> Any:
        """Map decoder hidden state at a position to a concrete value for a SET_* action kind."""
        if kind == ActionKind.SET_OP:
            return int(torch.argmax(self.op_head(hidden), dim=-1).item())
        if kind == ActionKind.SET_VAR_ID:
            return int(torch.argmax(self.var_id_head(hidden), dim=-1).item())
        if kind == ActionKind.SET_VARIABLE_NAME:
            # Not supported by bounded vocab; synthesize from var id
            vid = int(torch.argmax(self.var_id_head(hidden), dim=-1).item())
            return f"var_{vid}"
        if kind == ActionKind.SET_CONST_INT:
            return int(torch.argmax(self.small_int_head(hidden), dim=-1).item())
        if kind == ActionKind.SET_CONST_BOOL:
            return bool(int(torch.argmax(self.bool_head(hidden), dim=-1).item()))
        if kind == ActionKind.SET_CONST_STR:
            # Minimal string space; fall back to empty
            return ""
        if kind == ActionKind.SET_FUNCTION_NAME:
            return int(torch.argmax(self.fn_name_head(hidden), dim=-1).item())
        if kind == ActionKind.SET_ATTRIBUTE_NAME:
            return int(torch.argmax(self.attr_name_head(hidden), dim=-1).item())
        if kind == ActionKind.SET_PARAMS:
            # Return single-arg params for now
            return ["n"]
        if kind == ActionKind.SET_ELSE_BODY:
            return bool(int(torch.argmax(self.bool_head(hidden), dim=-1).item()))
        if kind == ActionKind.SET_ARG_LEN:
            return int(torch.argmax(self.arg_len_head(hidden), dim=-1).item())
        if kind == ActionKind.SET_LIST_LEN:
            return int(torch.argmax(self.list_len_head(hidden), dim=-1).item())
        if kind == ActionKind.SET_BLOCK_LEN:
            return int(torch.argmax(self.block_len_head(hidden), dim=-1).item())
        return None


class ProgramGenerator(nn.Module):
    """Grammar-masked generator that decodes action kinds and values from an arbitrary context embedding.

    This module is self-contained and does not depend on any encoder. Provide value vocabularies and action space
    at initialization time, and pass the context memory at generation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        decoder_layers: int,
        action_list: List[ActionKind],
        op_tokens: List[str],
        fn_tokens: List[str],
        attr_tokens: List[str],
        small_int_tokens: List[str],
        max_var_vocab_size: int,
    ) -> None:
        super().__init__()

        self.action_list = action_list
        self.action_to_id = {k: i for i, k in enumerate(self.action_list)}
        self.num_actions = len(self.action_list)

        self.op_tokens = op_tokens
        self.fn_tokens = fn_tokens
        self.attr_tokens = attr_tokens
        self.small_int_tokens = small_int_tokens
        self.max_var_vocab_size = max_var_vocab_size

        self.head = ProgramGenerationHead(
            d_model=d_model,
            num_heads=num_heads,
            decoder_layers=decoder_layers,
            op_vocab_size=len(op_tokens),
            fn_vocab_size=len(fn_tokens),
            attr_vocab_size=len(attr_tokens),
            small_int_vocab_size=len(small_int_tokens),
            max_var_vocab_size=max_var_vocab_size,
            num_actions=self.num_actions,
        )

    @torch.inference_mode()
    def build_valid_action_masks(self, batch_kind_ids: torch.Tensor) -> torch.Tensor:
        """Return mask tensor [B, T, num_actions] where True = valid, using ParserState per sequence."""
        B, T = batch_kind_ids.shape
        masks = torch.zeros((B, T, self.num_actions), dtype=torch.bool, device=batch_kind_ids.device)
        for b in range(B):
            # For each timestep, rebuild a fresh parser state from the prefix tokens
            for t in range(T):
                state = ParserState()
                # Feed tokens up to t-1 into the parser state
                for i in range(0, t):
                    tok_id = int(batch_kind_ids[b, i].item())
                    if tok_id == self.num_actions:  # BOS
                        continue
                    kind = self.action_list[tok_id]
                    try:
                        new_roles = state.apply_action(Action(kind))
                        for role in reversed(new_roles):
                            state.push(role)
                    except Exception:
                        # Ignore malformed prefixes during mask building
                        pass
                valid = create_action_mask(state)
                # If parser stack became empty due to consumption, reset to PROGRAM to allow EOS
                if len(state.stack) == 0:
                    state.stack = [ParserRole.PROGRAM]
                for i, kind in enumerate(self.action_list):
                    masks[b, t, i] = bool(valid.get(kind, False))
        return masks

    @torch.inference_mode()
    def compute_valid_action_mask_next(self, prefix_kind_ids: torch.Tensor) -> torch.Tensor:
        """Return [B, num_actions] mask of valid next actions after consuming the entire prefix."""
        B, T = prefix_kind_ids.shape
        masks = torch.zeros((B, self.num_actions), dtype=torch.bool, device=prefix_kind_ids.device)
        for b in range(B):
            state = ParserState()
            for i in range(T):
                tok_id = int(prefix_kind_ids[b, i].item())
                if tok_id == self.num_actions:  # BOS
                    continue
                kind = self.action_list[tok_id]
                try:
                    new_roles = state.apply_action(Action(kind))
                    for role in reversed(new_roles):
                        state.push(role)
                except Exception:
                    pass
            valid = create_action_mask(state)
            if len(state.stack) == 0:
                state.stack = [ParserRole.PROGRAM]
            for i, kind in enumerate(self.action_list):
                masks[b, i] = bool(valid.get(kind, False))
        return masks

    @torch.inference_mode()
    def generate(self, memory: torch.Tensor, max_len: int = 128) -> List[List[Action]]:
        """Generate action sequences from context memory.

        memory: [B, d] or [B, L, d]; if [B, d], it will be expanded to [B, 1, d].
        """
        device = next(self.parameters()).device
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        memory = memory.to(device)

        B = memory.shape[0]
        bos_id = self.num_actions
        tokens = torch.full((B, max_len), fill_value=bos_id, dtype=torch.long, device=device)
        finished = torch.zeros((B,), dtype=torch.bool, device=device)

        out_actions: List[List[Action]] = [[] for _ in range(B)]

        # Decoding state per sequence
        dec_states = [DecodingState(self.action_to_id) for _ in range(B)]

        for t in range(1, max_len):
            kind_logits, dec_hidden = self.head.forward_step(memory, tokens[:, :t])
            step_logits = kind_logits[:, -1, :]

            # Apply grammar mask and intersect with DecodingState
            grammar_mask = self.compute_valid_action_mask_next(tokens[:, :t])  # [B, A]
            valid_step = torch.zeros_like(grammar_mask)
            for b in range(B):
                # If minimal program is complete, force EOS
                if (
                    dec_states[b].stage == DecodingStage.AFTER_MINIMAL
                    and not dec_states[b].required_next_attrs
                    and dec_states[b].pending_expr_slots == 0
                ):
                    eos_idx = self.action_to_id[ActionKind.EOS]
                    valid_step[b, eos_idx] = True
                    continue

                allowed = dec_states[b].get_allowed_kinds()
                for k in allowed:
                    idx = self.action_to_id.get(k)
                    if idx is not None and bool(grammar_mask[b, idx].item()):
                        valid_step[b, idx] = True
                # If intersection is empty (should be rare), allow only EOS as a safe escape
                if not bool(valid_step[b].any().item()):
                    eos_idx = self.action_to_id[ActionKind.EOS]
                    valid_step[b, eos_idx] = True
            step_logits = step_logits.masked_fill(~valid_step, float("-inf"))

            pred = torch.argmax(step_logits, dim=-1)
            tokens[:, t] = pred

            # Values for SET_* from decoder hidden
            hidden_t = dec_hidden[:, -1, :]
            for b in range(B):
                if finished[b].item():
                    continue
                kind = self.action_list[int(pred[b].item())]
                if kind == ActionKind.EOS:
                    finished[b] = True
                    out_actions[b].append(Action(kind=kind))
                    continue
                # Update decoding state
                dec_states[b].apply(kind)
                if kind.name.startswith("SET_"):
                    v = self.head.predict_value_from_hidden(hidden_t[b], kind)
                    # Map categorical indices to tokens for some value types
                    if kind == ActionKind.SET_OP:
                        v = self.op_tokens[int(v)] if 0 <= int(v) < len(self.op_tokens) else "+"
                    if kind == ActionKind.SET_FUNCTION_NAME:
                        v = self.fn_tokens[int(v)] if 0 <= int(v) < len(self.fn_tokens) else "program"
                    if kind == ActionKind.SET_ATTRIBUTE_NAME:
                        v = self.attr_tokens[int(v)] if 0 <= int(v) < len(self.attr_tokens) else "attr"
                    if kind == ActionKind.SET_CONST_INT:
                        try:
                            v = int(self.small_int_tokens[int(v)])
                        except Exception:
                            v = 0
                    out_actions[b].append(Action(kind=kind, value=v))
                else:
                    out_actions[b].append(Action(kind=kind))

                # If we've completed a minimal valid program (single return expr), close with EOS
                if (
                    dec_states[b].stage == DecodingStage.AFTER_MINIMAL
                    and not dec_states[b].required_next_attrs
                    and dec_states[b].pending_expr_slots == 0
                ):
                    finished[b] = True
                    out_actions[b].append(Action(kind=ActionKind.EOS))

            if bool(finished.all().item()):
                break

        return out_actions


__all__ = ["ProgramGenerationHead", "ProgramGenerator"]


