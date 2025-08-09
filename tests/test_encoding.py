from dataset.ast import ASTSimplifier, ASTNodeType, EdgeType
from dataset.encoding import GraphEncoder


def test_encode_simple_binop():
    code = """
def program(n):
    return n + 1
"""
    graph = ASTSimplifier.ast_to_graph(code)
    enc = GraphEncoder()
    eg = enc.encode(graph)

    assert len(eg.node_type) == len(graph["nodes"])  # N
    assert len(eg.edge_index[0]) == len(graph["edges"])  # E
    # '+' operator present
    binop_idx = next(
        i for i, n in enumerate(graph["nodes"]) if n["type"] == ASTNodeType.BINARY_OPERATION
    )
    assert int(eg.op_id[binop_idx]) > 1  # in vocab (not PAD/UNK)
    # constant 1 encoded
    const_idx = next(
        i
        for i, n in enumerate(graph["nodes"])
        if n["type"] == ASTNodeType.CONSTANT and n.get("dtype") == "int" and n.get("value") == 1
    )
    # either exact id or numeric features non-zero
    cond = (int(eg.const_exact_int_id[const_idx]) >= 0) or any(v != 0 for v in eg.const_numeric[const_idx])
    assert cond


def test_variable_symbol_edges_align_var_ids():
    code = """
def program(a, b):
    x = a
    y = b
    return x
"""
    graph = ASTSimplifier.ast_to_graph(code)
    enc = GraphEncoder()
    eg = enc.encode(graph)

    nodes = graph["nodes"]
    # Find a variable occurrence and its symbol node for 'a'
    var_occ_idx = next(i for i, n in enumerate(nodes) if n["type"] == ASTNodeType.VARIABLE and n.get("name") == "a")
    sym_idx = next(i for i, n in enumerate(nodes) if n["type"] == ASTNodeType.VARIABLE_SYMBOL and n.get("name") == "a")

    # Check there is a SYMBOL edge
    has_symbol = any(
        src == var_occ_idx and dst == sym_idx and et == EdgeType.SYMBOL
        for src, dst, et in graph["edges"]
    )
    assert has_symbol
    # And var_id feature matches between occurrence and symbol node (symbol node var_id isn't encoded specially but should be equal in raw attrs)
    assert nodes[var_occ_idx]["var_id"] == nodes[sym_idx]["var_id"]
    # Encoded var_id for occurrence should be that var_id (non-zero)
    assert int(eg.var_id[var_occ_idx]) == nodes[var_occ_idx]["var_id"]


def test_position_scalars_are_normalized():
    code = """
def program(n):
    x = n
    y = x
    return y
"""
    graph = ASTSimplifier.ast_to_graph(code)
    enc = GraphEncoder()
    eg = enc.encode(graph)
    # position in [0,1]
    assert all(all(0.0 <= v <= 1.0 for v in row) for row in eg.position)


def test_string_and_attribute_vocab():
    code = """
def program(s):
    return s.upper()
"""
    graph = ASTSimplifier.ast_to_graph(code)
    enc = GraphEncoder()
    eg = enc.encode(graph)
    # Attribute node should have attribute id set
    attr_idx = next(i for i, n in enumerate(graph["nodes"]) if n["type"] == ASTNodeType.ATTRIBUTE)
    assert int(eg.attribute_name_id[attr_idx]) > 1


