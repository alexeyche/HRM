from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from dataset.grammar_actions import Action, ActionKind, ParserState, ParserRole, create_action_mask, actions_to_graph


# Removed the ad-hoc DecodingState; decoding will be governed by ParserState(strict_decoding=True)


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
                state = ParserState(strict_decoding=True)
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
            state = ParserState(strict_decoding=True)
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

        # Use ParserState(strict) per sequence to control next actions
        parser_states = [ParserState(strict_decoding=True) for _ in range(B)]

        for t in range(1, max_len):
            kind_logits, dec_hidden = self.head.forward_step(memory, tokens[:, :t])
            step_logits = kind_logits[:, -1, :]

            # Build strict grammar mask directly from ParserState
            valid_step = self.compute_valid_action_mask_next(tokens[:, :t])  # [B, A]
            # Fallback: if a row has no valid action, prefer starting/ending tokens
            for b in range(B):
                if not bool(valid_step[b].any().item()):
                    # At first step, force function def; otherwise allow EOS
                    if t == 1:
                        def_idx = self.action_to_id.get(ActionKind.PROD_FUNCTION_DEF)
                        if def_idx is not None:
                            valid_step[b, def_idx] = True
                    else:
                        eos_idx = self.action_to_id.get(ActionKind.EOS)
                        if eos_idx is not None:
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
                # Update parser state
                try:
                    new_roles = parser_states[b].apply_action(Action(kind))
                    for role in reversed(new_roles):
                        parser_states[b].push(role)
                except Exception:
                    # If invalid, force EOS next
                    finished[b] = True
                    out_actions[b].append(Action(kind=ActionKind.EOS))
                    continue
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

                # If parse completed (empty) or at PROGRAM and nothing expected, close with EOS immediately
                if (len(parser_states[b].stack) == 0 or (len(parser_states[b].stack) == 1 and parser_states[b].stack[0] == ParserRole.PROGRAM)) and not parser_states[b].expecting:
                    if len(parser_states[b].stack) == 0:
                        parser_states[b].stack = [ParserRole.PROGRAM]
                    finished[b] = True
                    out_actions[b].append(Action(kind=ActionKind.EOS))

            if bool(finished.all().item()):
                break

        # Best-effort completion for any unfinished sequences to guarantee parseability
        for b in range(B):
            if finished[b].item():
                continue
            # Complete up to a small cap to avoid loops
            steps_left = 64
            while steps_left > 0:
                steps_left -= 1
                # If expecting attribute(s), emit the first with a default value
                if parser_states[b].expecting:
                    need_kind = parser_states[b].expecting[0]
                    # Choose default values
                    default_val: Any = None
                    if need_kind == ActionKind.SET_VAR_ID:
                        default_val = 0
                    elif need_kind == ActionKind.SET_VARIABLE_NAME:
                        default_val = "n"
                    elif need_kind == ActionKind.SET_CONST_INT:
                        default_val = 0
                    elif need_kind == ActionKind.SET_CONST_BOOL:
                        default_val = False
                    elif need_kind == ActionKind.SET_CONST_STR:
                        default_val = ""
                    elif need_kind == ActionKind.SET_OP:
                        default_val = "+"
                    elif need_kind == ActionKind.SET_FUNCTION_NAME:
                        default_val = "program"
                    elif need_kind == ActionKind.SET_ATTRIBUTE_NAME:
                        default_val = "attr"
                    elif need_kind in (ActionKind.SET_ARG_LEN, ActionKind.SET_LIST_LEN, ActionKind.SET_BLOCK_LEN):
                        default_val = 0
                    else:
                        default_val = None
                    out_actions[b].append(Action(kind=need_kind, value=default_val))
                    try:
                        parser_states[b].apply_action(Action(need_kind, default_val))
                    except Exception:
                        pass
                    continue
                # No attribute expected: choose a minimal production to reduce stack
                if len(parser_states[b].stack) == 0 or (len(parser_states[b].stack) == 1 and parser_states[b].stack[0] == ParserRole.PROGRAM):
                    break
                top = parser_states[b].peek()
                if top in (ParserRole.STMT, ParserRole.IF_BODY, ParserRole.FOR_BODY, ParserRole.WHILE_BODY):
                    k = ActionKind.PROD_RETURN
                elif top in (ParserRole.EXPR, ParserRole.CONST, ParserRole.IF_COND, ParserRole.FOR_ITER, ParserRole.WHILE_COND, ParserRole.ASSIGN_VALUE):
                    k = ActionKind.PROD_CONSTANT_INT
                elif top in (ParserRole.ASSIGN_TARGET, ParserRole.NAME):
                    k = ActionKind.PROD_VARIABLE
                elif top in (ParserRole.EXPR_LIST, ParserRole.ARG_LIST):
                    # Close lists/args by emitting zero-length via prior expectation
                    # If we got here without expectation, just place a constant
                    k = ActionKind.PROD_CONSTANT_INT
                else:
                    # Default fallback: wrap an expression
                    k = ActionKind.PROD_EXPRESSION
                # Guard: for FOR_ITER fallback, prefer a simple iterable like range(0)
                if top == ParserRole.FOR_ITER:
                    k = ActionKind.PROD_FUNCTION_CALL
                    out_actions[b].append(Action(kind=k))
                    try:
                        new_roles = parser_states[b].apply_action(Action(k))
                        for role in reversed(new_roles):
                            parser_states[b].push(role)
                        # Expect function name next
                        out_actions[b].append(Action(kind=ActionKind.SET_FUNCTION_NAME, value="range"))
                        parser_states[b].apply_action(Action(ActionKind.SET_FUNCTION_NAME, "range"))
                        out_actions[b].append(Action(kind=ActionKind.SET_ARG_LEN, value=1))
                        parser_states[b].apply_action(Action(ActionKind.SET_ARG_LEN, 1))
                        # Emit constant 0 as argument
                        out_actions[b].append(Action(kind=ActionKind.PROD_CONSTANT_INT))
                        new_roles = parser_states[b].apply_action(Action(ActionKind.PROD_CONSTANT_INT))
                        for role in reversed(new_roles):
                            parser_states[b].push(role)
                        out_actions[b].append(Action(kind=ActionKind.SET_CONST_INT, value=0))
                        parser_states[b].apply_action(Action(ActionKind.SET_CONST_INT, 0))
                        continue
                    except Exception:
                        # Fall back to constant
                        k = ActionKind.PROD_CONSTANT_INT
                out_actions[b].append(Action(kind=k))
                try:
                    new_roles = parser_states[b].apply_action(Action(k))
                    for role in reversed(new_roles):
                        parser_states[b].push(role)
                except Exception:
                    break
            # Append EOS when structurally complete
            out_actions[b].append(Action(kind=ActionKind.EOS))

        # Post-process: ensure each sequence starts with PROD_FUNCTION_DEF and ends with a single EOS
        for b in range(B):
            seq = out_actions[b]
            if not seq or seq[0].kind != ActionKind.PROD_FUNCTION_DEF:
                seq = [Action(kind=ActionKind.PROD_FUNCTION_DEF)] + seq
            # Truncate after first EOS if any
            eos_pos = None
            for i, a in enumerate(seq):
                if a.kind == ActionKind.EOS:
                    eos_pos = i
                    break
            if eos_pos is not None:
                seq = seq[:eos_pos + 1]
            else:
                seq.append(Action(kind=ActionKind.EOS))
            out_actions[b] = seq

        # Validate parsability; if invalid, fall back to a minimal well-formed program
        for b in range(B):
            try:
                _ = actions_to_graph(out_actions[b])
            except Exception:
                out_actions[b] = [
                    Action(kind=ActionKind.PROD_FUNCTION_DEF),
                    Action(kind=ActionKind.PROD_RETURN),
                    Action(kind=ActionKind.PROD_CONSTANT_INT),
                    Action(kind=ActionKind.SET_CONST_INT, value=0),
                    Action(kind=ActionKind.EOS),
                ]

        return out_actions


__all__ = ["ProgramGenerationHead", "ProgramGenerator"]


