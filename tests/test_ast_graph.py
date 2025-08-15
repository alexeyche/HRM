from dataset.ast import ASTSimplifier, ASTNodeType, EdgeType


def node_indices_by_type(nodes, node_type):
    return [i for i, n in enumerate(nodes) if n.type == node_type]


def test_simple_function_with_binop_and_constant():
    code = """
def program(n):
    return n + 1
"""
    graph = ASTSimplifier.ast_to_graph(code)
    nodes = graph["nodes"]
    edges = graph["edges"]

    # One symbol node for 'n'
    sym_idxs = [i for i, n in enumerate(nodes) if n.type == ASTNodeType.VARIABLE_SYMBOL and n.name == "n"]
    assert len(sym_idxs) == 1
    symbol_idx = sym_idxs[0]

    # One variable occurrence for 'n'
    var_occ_idxs = [i for i, n in enumerate(nodes) if n.type == ASTNodeType.VARIABLE and n.name == "n"]
    assert len(var_occ_idxs) >= 1

    # Ensure SYMBOL edge from occurrence to symbol node
    has_symbol_edge = any(src in var_occ_idxs and dst == symbol_idx and et == EdgeType.SYMBOL for src, dst, et in edges)
    assert has_symbol_edge

    # Function def exists with correct name and params
    fn_idxs = [i for i, n in enumerate(nodes) if n.type == ASTNodeType.FUNCTION_DEF and n.name == "program"]
    assert len(fn_idxs) == 1
    assert nodes[fn_idxs[0]].params == ["n"]

    # Return node exists
    assert any(n.type == ASTNodeType.RETURN for n in nodes)

    # Binary operation '+' exists
    binop_idxs = [i for i, n in enumerate(nodes) if n.type == ASTNodeType.BINARY_OPERATION and n.op == "+"]
    assert len(binop_idxs) == 1

    # Constant 1 exists and is typed as int
    const_idxs = [i for i, n in enumerate(nodes) if n.type == ASTNodeType.CONSTANT and n.dtype == "int" and n.value == 1]
    assert len(const_idxs) == 1

    # There should be at least one AST edge out of function def
    has_ast_child = any(src == fn_idxs[0] and et == EdgeType.AST for src, dst, et in edges)
    assert has_ast_child


def test_next_sibling_edges_in_function_body():
    code = """
def program(a, b):
    x = a
    y = b
    return x
"""
    graph = ASTSimplifier.ast_to_graph(code)
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Locate the function definition node
    fn_idx = next(i for i, n in enumerate(nodes) if n.type == ASTNodeType.FUNCTION_DEF and n.name == "program")

    # Get direct AST-children of the function (order-preserving)
    fn_children = [dst for src, dst, et in edges if src == fn_idx and et == EdgeType.AST]

    # Expect at least three statements: assignment, assignment, return
    assert len(fn_children) >= 3

    # Identify assignment children
    assign_children = [idx for idx in fn_children if nodes[idx].type == ASTNodeType.ASSIGNMENT]
    assert len(assign_children) >= 2

    # There should be a NEXT_SIBLING edge from the first assignment to the second
    first_assign, second_assign = assign_children[0], assign_children[1]
    has_next_sibling = any(src == first_assign and dst == second_assign and et == EdgeType.NEXT_SIBLING for src, dst, et in edges)
    assert has_next_sibling


