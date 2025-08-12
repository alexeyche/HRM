from __future__ import annotations

import torch

from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, collate_to_pyg_batch
from dataset.grammar_actions import ActionKind
from models.autoencoder import ProgramAutoencoder
from models.generation import ProgramGenerator
from dataset.grammar_actions import actions_to_graph
import ast
import logging

log = logging.getLogger(__name__)

def test_program_generator_smoke():
    d_model = 64
    for _ in range(25):


        # Instantiate self-contained ProgramGenerator
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
        with torch.inference_mode():
            actions_list = gen.generate(
                ctx,
                max_len=32,
                use_heuristics=True,
                sampling=False,
                temperature=0.5,
                top_k=None
            )

        graph = actions_to_graph(actions_list[0])

        # Convert to code
        final_code = ASTSimplifier.ast_to_program(graph)

        # Debug: log.info the final code
        log.info(f"Final code:\n{final_code}")
        # Validate that the final code is valid Python
        try:
            ast.parse(final_code)
        except Exception as e:
            log.error(f"Invalid Python code generated: {e}")
            assert False, f"Invalid Python code generated: {e}"

        assert isinstance(actions_list, list) and len(actions_list) == 1
        actions = actions_list[0]
        assert len(actions) > 0  # generated some actions

