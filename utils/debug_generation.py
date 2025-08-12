#!/usr/bin/env python3
from __future__ import annotations

import torch
from dataset.encoding import GraphEncoder
from dataset.grammar_actions import ActionKind, actions_to_graph
from models.generation import ProgramGenerator
from dataset.ast import ASTSimplifier


def main() -> None:
    d_model = 64
    ge = GraphEncoder()
    action_list = list(ActionKind)
    gen = ProgramGenerator(
        d_model=d_model,
        num_heads=2,
        decoder_layers=1,
        action_list=action_list,
        op_tokens=list(ge.op_vocab._id_to_tok),
        fn_tokens=list(ge.function_vocab._id_to_tok),
        attr_tokens=list(ge.attribute_vocab._id_to_tok),
        small_int_tokens=list(ge.small_int_vocab._id_to_tok),
        max_var_vocab_size=ge.max_var_vocab_size,
    )

    ctx = torch.randn(1, d_model)
    actions_list, traces = gen.generate_debug(ctx, max_len=32, sampling=False)

    actions = actions_list[0]
    trace = traces[0]

    print("=== TRACE ===")
    for step in trace:
        chosen = step["chosen_kind"].name
        expecting = [k.name for k in step["expecting"]]
        stack = [s.name for s in step["stack"]]
        valid = [k.name for k in step["valid_next"]]
        set_val = step["set_value"]
        finished = step["finished"]
        print(f"t={step['t']:02d} kind={chosen} expecting={expecting} stack={stack} finished={finished}")
        print(f"\tvalid_next={valid}")
        if set_val is not None:
            print(f"\tset_value={set_val}")

    try:
        graph = actions_to_graph(actions)
        code = ASTSimplifier.ast_to_program(graph)
        print("\n=== CODE ===\n" + code)
    except Exception as e:
        print("Failed to convert to code:", e)


if __name__ == "__main__":
    main()


