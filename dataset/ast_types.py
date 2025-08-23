"""
AST Node Types for Program Synthesis

This module defines AST node types based on the grammar in grammar.py
and standard Python AST nodes. Used for graph-based program representation.
"""

from enum import Enum, auto
from typing import Dict, Set


class ASTNodeType(Enum):
    """
    Enumeration of AST node types for program synthesis.
    
    Based on grammar.py non-terminals and Python AST node types.
    Each type corresponds to either a grammar production or a semantic construct.
    """
    
    # === Grammar-based Node Types ===
    # Function structure
    FUNC_DEF = auto()           # Function definition
    PARAMS = auto()             # Parameter list  
    BODY = auto()               # Function body
    
    # Statements
    STMT_LIST = auto()          # List of statements
    STMT_OR_BLOCK = auto()      # Statement or block
    ASSIGNMENT = auto()         # Assignment statement
    SIMPLE_ASSIGN = auto()      # Simple assignment (a = b)
    AUGMENTED_ASSIGN = auto()   # Augmented assignment (a += b)
    RETURN_STMT = auto()        # Return statement
    
    # Control flow
    IF_BLOCK = auto()           # If statement
    ELIF_BLOCK = auto()         # Elif statement
    ELIF_CHAIN = auto()         # Chain of elif statements
    ELSE_BLOCK = auto()         # Else statement
    WHILE_LOOP = auto()         # While loop
    FOR_LOOP = auto()           # For loop
    BREAK_STMT = auto()         # Break statement
    CONTINUE_STMT = auto()      # Continue statement
    
    # Expressions
    EXPR = auto()               # General expression
    OR_EXPR = auto()            # Boolean OR expression
    AND_EXPR = auto()           # Boolean AND expression
    NOT_EXPR = auto()           # Boolean NOT expression
    COMPARISON = auto()         # Comparison expression
    COMP_CHAIN = auto()         # Comparison chain (a < b < c)
    
    # Arithmetic
    ARITH_EXPR = auto()         # Arithmetic expression
    TERM = auto()               # Multiplication/division term
    POWER_EXPR = auto()         # Power expression (a**b)
    FACTOR = auto()             # Basic factor
    
    # Literals and identifiers
    VARIABLE = auto()           # Variable identifier
    DIGIT = auto()              # Numeric literal
    STRING = auto()             # String literal
    TRUE = auto()               # Boolean True
    FALSE = auto()              # Boolean False
    
    # Collections
    LIST_LITERAL = auto()       # List literal [1, 2, 3]
    LIST_CONTENTS = auto()      # Contents of list
    EXPR_LIST = auto()          # Expression list
    LIST_INDEX = auto()         # List indexing a[i]
    
    # Functions and methods
    FUNCTION_CALL = auto()      # Function call
    BUILTIN_FUNC = auto()       # Built-in function call
    RANGE_CALL = auto()         # Range function call
    METHOD_CALL = auto()        # Method call obj.method()
    ARG_LIST = auto()           # Argument list
    RANGE_ARGS = auto()         # Range arguments
    SIMPLE_EXPR = auto()        # Simple expression for ranges
    
    # Operators (semantic)
    BINARY_OP = auto()          # Binary operation (+, -, *, etc.)
    UNARY_OP = auto()           # Unary operation (-, not)
    COMPARE = auto()            # Comparison operation
    BOOL_OP = auto()            # Boolean operation (and, or)
    
    # === Python AST-based Node Types ===
    # Core constructs
    MODULE = auto()             # Module (top level)
    FUNCTIONDEF = auto()        # Function definition (AST)
    ARGUMENTS = auto()          # Function arguments
    ARG = auto()                # Single argument
    
    # Statements (AST)
    ASSIGN = auto()             # Assignment (AST)
    AUGASSIGN = auto()          # Augmented assignment (AST)
    RETURN = auto()             # Return (AST)
    IF = auto()                 # If statement (AST)
    WHILE = auto()              # While loop (AST)
    FOR = auto()                # For loop (AST)
    BREAK = auto()              # Break (AST)
    CONTINUE = auto()           # Continue (AST)
    
    # Expressions (AST)
    BINOP = auto()              # Binary operation (AST)
    UNARYOP = auto()            # Unary operation (AST)
    COMPARE_AST = auto()        # Compare (AST)
    BOOLOP = auto()             # Boolean operation (AST)
    CALL = auto()               # Function call (AST)
    SUBSCRIPT = auto()          # Subscript (AST)
    ATTRIBUTE = auto()          # Attribute access (AST)
    
    # Literals (AST)
    CONSTANT = auto()           # Constant value (AST)
    NAME = auto()               # Name/identifier (AST)
    LIST = auto()               # List (AST)
    
    # Special
    LOAD = auto()               # Load context
    STORE = auto()              # Store context
    DEL = auto()                # Delete context


class OperatorType(Enum):
    """Types of operators for semantic encoding."""
    
    # Arithmetic operators
    ADD = auto()                # +
    SUB = auto()                # -
    MULT = auto()               # *
    DIV = auto()                # /
    FLOORDIV = auto()           # //
    MOD = auto()                # %
    POW = auto()                # **
    
    # Unary operators
    UADD = auto()               # +a
    USUB = auto()               # -a
    NOT = auto()                # not a
    
    # Comparison operators
    EQ = auto()                 # ==
    NOTEQ = auto()              # !=
    LT = auto()                 # <
    LTE = auto()                # <=
    GT = auto()                 # >
    GTE = auto()                # >=
    IS = auto()                 # is
    ISNOT = auto()              # is not
    IN = auto()                 # in
    NOTIN = auto()              # not in
    
    # Boolean operators
    AND = auto()                # and
    OR = auto()                 # or


class EdgeType(Enum):
    """Types of edges in the AST graph."""
    
    # Structural edges (parent-child relationships)
    CHILD = auto()              # Parent -> Child
    NEXT_SIBLING = auto()       # Sibling -> Next sibling
    
    # Semantic edges
    CONDITION = auto()          # If/while -> condition
    BODY = auto()               # If/while/for -> body
    ORELSE = auto()             # If -> else body
    TARGET = auto()             # Assignment -> target
    VALUE = auto()              # Assignment -> value
    ARGS = auto()               # Call -> arguments
    FUNC = auto()               # Call -> function
    
    # Data flow edges
    DEF_USE = auto()            # Variable definition -> use
    USE_DEF = auto()            # Variable use -> definition
    
    # Control flow edges
    CONTROL_FLOW = auto()       # Control flow between statements


# Node type groupings for easier processing
STATEMENT_NODES: Set[ASTNodeType] = {
    ASTNodeType.ASSIGNMENT, ASTNodeType.SIMPLE_ASSIGN, ASTNodeType.AUGMENTED_ASSIGN,
    ASTNodeType.RETURN_STMT, ASTNodeType.IF_BLOCK, ASTNodeType.WHILE_LOOP,
    ASTNodeType.FOR_LOOP, ASTNodeType.BREAK_STMT, ASTNodeType.CONTINUE_STMT,
    ASTNodeType.ASSIGN, ASTNodeType.AUGASSIGN, ASTNodeType.RETURN,
    ASTNodeType.IF, ASTNodeType.WHILE, ASTNodeType.FOR, ASTNodeType.BREAK, ASTNodeType.CONTINUE
}

EXPRESSION_NODES: Set[ASTNodeType] = {
    ASTNodeType.EXPR, ASTNodeType.OR_EXPR, ASTNodeType.AND_EXPR, ASTNodeType.NOT_EXPR,
    ASTNodeType.COMPARISON, ASTNodeType.ARITH_EXPR, ASTNodeType.TERM, ASTNodeType.POWER_EXPR,
    ASTNodeType.FACTOR, ASTNodeType.BINARY_OP, ASTNodeType.UNARY_OP, ASTNodeType.COMPARE,
    ASTNodeType.BOOL_OP, ASTNodeType.BINOP, ASTNodeType.UNARYOP, ASTNodeType.COMPARE_AST,
    ASTNodeType.BOOLOP, ASTNodeType.CALL, ASTNodeType.SUBSCRIPT, ASTNodeType.ATTRIBUTE
}

LITERAL_NODES: Set[ASTNodeType] = {
    ASTNodeType.DIGIT, ASTNodeType.STRING, ASTNodeType.TRUE, ASTNodeType.FALSE,
    ASTNodeType.CONSTANT, ASTNodeType.VARIABLE, ASTNodeType.NAME
}

CONTAINER_NODES: Set[ASTNodeType] = {
    ASTNodeType.LIST_LITERAL, ASTNodeType.LIST, ASTNodeType.LIST_CONTENTS,
    ASTNodeType.EXPR_LIST, ASTNodeType.ARG_LIST
}


def get_node_category(node_type: ASTNodeType) -> str:
    """Get the category of a node type for grouping."""
    if node_type in STATEMENT_NODES:
        return "statement"
    elif node_type in EXPRESSION_NODES:
        return "expression"
    elif node_type in LITERAL_NODES:
        return "literal"
    elif node_type in CONTAINER_NODES:
        return "container"
    else:
        return "structural"


def is_semantic_node(node_type: ASTNodeType) -> bool:
    """Check if a node type represents semantic content vs structural."""
    structural_nodes = {
        ASTNodeType.BODY, ASTNodeType.STMT_LIST, ASTNodeType.EXPR_LIST,
        ASTNodeType.LIST_CONTENTS, ASTNodeType.ARG_LIST, ASTNodeType.PARAMS
    }
    return node_type not in structural_nodes


# Mapping from Python AST node names to our enum
PYTHON_AST_TO_ENUM: Dict[str, ASTNodeType] = {
    'Module': ASTNodeType.MODULE,
    'FunctionDef': ASTNodeType.FUNCTIONDEF,
    'arguments': ASTNodeType.ARGUMENTS,
    'arg': ASTNodeType.ARG,
    'Assign': ASTNodeType.ASSIGN,
    'AugAssign': ASTNodeType.AUGASSIGN,
    'Return': ASTNodeType.RETURN,
    'If': ASTNodeType.IF,
    'While': ASTNodeType.WHILE,
    'For': ASTNodeType.FOR,
    'Break': ASTNodeType.BREAK,
    'Continue': ASTNodeType.CONTINUE,
    'BinOp': ASTNodeType.BINOP,
    'UnaryOp': ASTNodeType.UNARYOP,
    'Compare': ASTNodeType.COMPARE_AST,
    'BoolOp': ASTNodeType.BOOLOP,
    'Call': ASTNodeType.CALL,
    'Subscript': ASTNodeType.SUBSCRIPT,
    'Attribute': ASTNodeType.ATTRIBUTE,
    'Constant': ASTNodeType.CONSTANT,
    'Name': ASTNodeType.NAME,
    'List': ASTNodeType.LIST,
    'Load': ASTNodeType.LOAD,
    'Store': ASTNodeType.STORE,
    'Del': ASTNodeType.DEL,
}


# Mapping from Python AST operator names to our enum
PYTHON_OPERATOR_TO_ENUM: Dict[str, OperatorType] = {
    'Add': OperatorType.ADD,
    'Sub': OperatorType.SUB,
    'Mult': OperatorType.MULT,
    'Div': OperatorType.DIV,
    'FloorDiv': OperatorType.FLOORDIV,
    'Mod': OperatorType.MOD,
    'Pow': OperatorType.POW,
    'UAdd': OperatorType.UADD,
    'USub': OperatorType.USUB,
    'Not': OperatorType.NOT,
    'Eq': OperatorType.EQ,
    'NotEq': OperatorType.NOTEQ,
    'Lt': OperatorType.LT,
    'LtE': OperatorType.LTE,
    'Gt': OperatorType.GT,
    'GtE': OperatorType.GTE,
    'Is': OperatorType.IS,
    'IsNot': OperatorType.ISNOT,
    'In': OperatorType.IN,
    'NotIn': OperatorType.NOTIN,
    'And': OperatorType.AND,
    'Or': OperatorType.OR,
}