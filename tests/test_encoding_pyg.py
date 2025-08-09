import pytest

from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, to_pyg_data, collate_to_pyg_batch

import torch
import torch_geometric

@pytest.mark.skipif("torch_geometric" not in globals(), reason="PyG not installed in CI env")
def test_to_pyg_and_batch():

    code1 = """
def program(n):
    return n + 1
"""
    code2 = """
def program(a, b):
    x = a
    y = b
    return x
"""
    enc = GraphEncoder()
    g1 = enc.encode(ASTSimplifier.ast_to_graph(code1))
    g2 = enc.encode(ASTSimplifier.ast_to_graph(code2))

    d1 = to_pyg_data(g1)
    d2 = to_pyg_data(g2)
    assert hasattr(d1, "edge_index") and hasattr(d1, "node_type")
    batch = collate_to_pyg_batch([g1, g2])
    assert hasattr(batch, "batch")
    assert batch.num_graphs == 2


