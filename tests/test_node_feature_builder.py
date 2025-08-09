import pytest

from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, to_pyg_data
from models.features import NodeFeatureBuilder


def _make_torch_data(code: str):
    enc = GraphEncoder()
    g = enc.encode(ASTSimplifier.ast_to_graph(code))
    d = to_pyg_data(g)
    return d


@pytest.mark.parametrize("mode", ["sum", "concat", "attn"])
def test_node_feature_builder_modes(mode):
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("PyTorch not available")

    code = """
def program(n):
    return n + 1
"""
    data = _make_torch_data(code)
    builder = NodeFeatureBuilder(d_model=64, mode=mode)
    x = builder.forward(data)
    assert x.shape[0] == data.num_nodes
    assert x.shape[1] == 64
    assert (x == x).all()  # no NaNs


