import pytest
from dataset.ast import ASTSimplifier, ASTNodeType, EdgeType
from dataset.grammar_actions import (
    graph_to_actions, actions_to_graph, ActionKind, Action,
    ParserState, ParserRole, create_action_mask, get_parser_state_for_actions
)


def test_roundtrip_simple_return_constant():
    code = """
def program(n):
    return 1
"""
    g = ASTSimplifier.ast_to_graph(code)
    actions = graph_to_actions(g)
    assert actions[-1].kind == ActionKind.EOS
    g2 = actions_to_graph(actions)

    # Root/module types
    assert g2["nodes"][g2["root"]]["type"] == ASTNodeType.MODULE
    fn = next(i for i, n in enumerate(g2["nodes"]) if n["type"] == ASTNodeType.FUNCTION_DEF)
    ret = next(i for i, n in enumerate(g2["nodes"]) if n["type"] == ASTNodeType.RETURN)
    const = next(i for i, n in enumerate(g2["nodes"]) if n["type"] == ASTNodeType.CONSTANT)
    assert g2["nodes"][const]["value"] == 1

    # Check edges
    edges = g2["edges"]
    assert any(e[0] == g2["root"] and e[1] == fn and e[2] == EdgeType.AST for e in edges)
    assert any(e[0] == fn and e[1] == ret and e[2] == EdgeType.AST for e in edges)
    assert any(e[0] == ret and e[1] == const and e[2] == EdgeType.AST for e in edges)


def test_roundtrip_binary_op():
    code = """
def program(n):
    return n + 1
"""
    g = ASTSimplifier.ast_to_graph(code)
    actions = graph_to_actions(g)
    g2 = actions_to_graph(actions)

    # Find binary op
    bin_op = next(i for i, n in enumerate(g2["nodes"]) if n["type"] == ASTNodeType.BINARY_OPERATION)
    assert g2["nodes"][bin_op]["op"] == "+"

    # Check children
    children = [e[1] for e in g2["edges"] if e[0] == bin_op and e[2] == EdgeType.AST]
    assert len(children) == 2
    var_node = next(i for i in children if g2["nodes"][i]["type"] == ASTNodeType.VARIABLE)
    const_node = next(i for i in children if g2["nodes"][i]["type"] == ASTNodeType.CONSTANT)
    assert g2["nodes"][var_node]["var_id"] == 1  # n (first variable gets var_id=1)
    assert g2["nodes"][const_node]["value"] == 1


def test_parser_state_initial():
    """Test initial parser state"""
    state = ParserState()
    assert state.peek() == ParserRole.PROGRAM
    assert len(state.stack) == 1

    valid_actions = state.get_valid_actions()
    assert ActionKind.PROD_FUNCTION_DEF in valid_actions
    assert ActionKind.EOS in valid_actions


def test_parser_state_function_def():
    """Test parser state after function definition"""
    state = ParserState()

    # Apply function def action
    new_roles = state.apply_action(Action(ActionKind.PROD_FUNCTION_DEF))
    assert new_roles == [ParserRole.STMT]

    # Push the new role
    for role in new_roles:
        state.push(role)

    assert state.peek() == ParserRole.STMT
    valid_actions = state.get_valid_actions()
    assert ActionKind.PROD_RETURN in valid_actions


def test_parser_state_return():
    """Test parser state after return statement"""
    state = ParserState()

    # Setup: PROGRAM -> STMT
    state.apply_action(Action(ActionKind.PROD_FUNCTION_DEF))
    state.push(ParserRole.STMT)

    # Apply return action
    new_roles = state.apply_action(Action(ActionKind.PROD_RETURN))
    assert new_roles == [ParserRole.EXPR]

    # Push the new role
    for role in new_roles:
        state.push(role)

    assert state.peek() == ParserRole.EXPR
    valid_actions = state.get_valid_actions()
    assert ActionKind.PROD_VARIABLE in valid_actions
    assert ActionKind.PROD_CONSTANT_INT in valid_actions
    assert ActionKind.PROD_BINARY_OP in valid_actions


def test_parser_state_binary_op():
    """Test parser state after binary operation"""
    state = ParserState()

    # Setup: PROGRAM -> STMT -> EXPR
    state.apply_action(Action(ActionKind.PROD_FUNCTION_DEF))
    state.push(ParserRole.STMT)
    state.apply_action(Action(ActionKind.PROD_RETURN))
    state.push(ParserRole.EXPR)

    # Apply binary op action
    new_roles = state.apply_action(Action(ActionKind.PROD_BINARY_OP))
    assert new_roles == [ParserRole.EXPR, ParserRole.EXPR]

    # Push the new roles (right to left for stack)
    for role in reversed(new_roles):
        state.push(role)

    assert state.peek() == ParserRole.EXPR
    valid_actions = state.get_valid_actions()
    assert ActionKind.PROD_VARIABLE in valid_actions
    assert ActionKind.PROD_CONSTANT_INT in valid_actions


def test_action_mask():
    """Test action masking functionality"""
    state = ParserState()
    mask = create_action_mask(state)

    # Initial state: only function def and EOS should be valid
    assert mask[ActionKind.PROD_FUNCTION_DEF] == True
    assert mask[ActionKind.EOS] == True
    assert mask[ActionKind.PROD_RETURN] == False
    assert mask[ActionKind.PROD_VARIABLE] == False


def test_get_parser_state_for_actions():
    """Test reconstructing parser state from actions"""
    actions = [
        Action(ActionKind.BOS),
        Action(ActionKind.PROD_FUNCTION_DEF),
        Action(ActionKind.PROD_RETURN),
        Action(ActionKind.PROD_VARIABLE),
        Action(ActionKind.SET_VAR_ID, 0),
        Action(ActionKind.EOS)
    ]

    state = get_parser_state_for_actions(actions)
    assert state.peek() == ParserRole.PROGRAM
    assert len(state.stack) == 1


def test_invalid_action_raises_error():
    """Test that invalid actions raise errors"""
    state = ParserState()

    # Try to apply return without being in STMT role
    with pytest.raises(ValueError, match="Invalid action"):
        state.apply_action(Action(ActionKind.PROD_RETURN))


def test_eos_with_non_empty_stack_raises_error():
    """Test that EOS with non-empty stack raises error"""
    state = ParserState()

    # Setup: PROGRAM -> STMT
    state.apply_action(Action(ActionKind.PROD_FUNCTION_DEF))
    # Don't push additional STMT - PROD_FUNCTION_DEF already replaced PROGRAM with STMT

    # Try to end program with non-empty stack
    with pytest.raises(ValueError, match="Cannot end program"):
        state.apply_action(Action(ActionKind.EOS))



