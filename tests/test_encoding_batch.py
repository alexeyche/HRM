import numpy as np

from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, collate_encoded_graphs


def _make_graphs():
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
    return g1, g2


def test_collate_shapes_and_reindex():
    g1, g2 = _make_graphs()
    batched = collate_encoded_graphs([g1, g2])

    # Node counts add up
    assert batched.node_type.shape[0] == g1.node_type.shape[0] + g2.node_type.shape[0]
    # Edge counts add up
    assert batched.edge_index.shape[1] == g1.edge_index.shape[1] + g2.edge_index.shape[1]
    # Offsets applied: all edges in first graph point to < g1_nodes
    assert np.all(batched.edge_index[:, : g1.edge_index.shape[1]] < g1.node_type.shape[0])
    # And edges in second part are offset by g1 size
    if g2.edge_index.shape[1] > 0:
        assert np.all(
            batched.edge_index[:, g1.edge_index.shape[1] :] >= g1.node_type.shape[0]
        )
    # graph_id marks membership
    assert batched.graph_id.shape[0] == batched.node_type.shape[0]
    assert batched.num_graphs == 2
    # node_ptr monotonic
    assert np.all(batched.node_ptr[1:] >= batched.node_ptr[:-1])


