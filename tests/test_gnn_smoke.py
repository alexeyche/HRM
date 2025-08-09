import pytest

from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, collate_to_pyg_batch
from models.features import NodeFeatureBuilder
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def _make_batch():
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
    batch = collate_to_pyg_batch([g1, g2])
    return batch


@pytest.mark.parametrize("mode", ["sum", "concat", "attn"])
def test_gnn_smoke_forward_backward(mode):

    batch = _make_batch()
    builder = NodeFeatureBuilder(d_model=64, mode=mode)

    class Model(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.conv1 = GCNConv(64, 64)
            self.conv2 = GCNConv(64, 64)
            self.out = nn.Linear(64, num_classes)

        def forward(self, data):
            x = builder.forward(data)
            x = F.relu(self.conv1(x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            logits = self.out(x)
            return logits

    num_classes = int(batch.node_type.max().item()) + 1
    model = Model(num_classes=num_classes)
    logits = model(batch)
    assert logits.shape[0] == batch.num_nodes
    assert logits.shape[1] == num_classes

    # simple supervised loss on node_type
    loss = F.cross_entropy(logits, batch.node_type)
    loss.backward()


