"""
Context-Free Grammar (CFG) for generating Python code from AST graphs.

This module provides a production-based CFG system that can traverse AST graphs
and generate code deterministically using greedy selection instead of random choice.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
from enum import Enum
import numpy as np
from .ast import *


class CFGNonTerminal(Enum):
    """Non-terminal symbols in the CFG"""

    # Program structure
    PROGRAM = "PROGRAM"
    FUNCTION = "FUNCTION"
    FUNCTION_HEADER = "FUNCTION_HEADER"
    FUNCTION_BODY = "FUNCTION_BODY"
    STATEMENT = "STATEMENT"
    STATEMENT_LIST = "STATEMENT_LIST"

    # Control flow
    IF_STMT = "IF_STMT"
    FOR_STMT = "FOR_STMT"
    WHILE_STMT = "WHILE_STMT"
    RETURN_STMT = "RETURN_STMT"

    # Expressions
    EXPRESSION = "EXPRESSION"
    BINARY_EXPR = "BINARY_EXPR"
    UNARY_EXPR = "UNARY_EXPR"
    COMPARISON_EXPR = "COMPARISON_EXPR"
    BOOLEAN_EXPR = "BOOLEAN_EXPR"
    CALL_EXPR = "CALL_EXPR"

    # Variables and assignments
    ASSIGNMENT = "ASSIGNMENT"
    AUG_ASSIGNMENT = "AUG_ASSIGNMENT"
    VARIABLE = "VARIABLE"

    # Literals and data structures
    CONSTANT = "CONSTANT"
    LIST_EXPR = "LIST_EXPR"
    TUPLE_EXPR = "TUPLE_EXPR"
    SLICE = "SLICE"

    # Utility
    IDENTIFIER = "IDENTIFIER"
    OPERATOR = "OPERATOR"
    INDENT = "INDENT"
    NEWLINE = "NEWLINE"


    @classmethod
    def find(cls, value: str) -> "CFGNonTerminal | None":
        if value in cls.__members__:
            return cls.__members__[value]
        for non_terminal in cls:
            if non_terminal.value == value:
                return non_terminal
        return None

class CFGTerminal(Enum):
    """Terminal symbols (actual code tokens)"""

    # Keywords
    DEF = "def"
    IF = "if"
    ELIF = "elif"
    ELSE = "else"
    FOR = "for"
    WHILE = "while"
    IN = "in"
    RETURN = "return"
    AND = "and"
    OR = "or"
    NOT = "not"

    # Operators
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "**"

    # Assignment operators
    ASSIGN = "="
    PLUS_ASSIGN = "+="
    MINUS_ASSIGN = "-="
    MULT_ASSIGN = "*="
    DIV_ASSIGN = "/="

    # Comparison operators
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    # Punctuation
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    COLON = ":"
    COMMA = ","

    # Formatting
    SPACE = " "
    TAB = "\t"
    NEWLINE = "\n"

    @classmethod
    def find(cls, value: str) -> "CFGTerminal | None":
        if value in cls.__members__:
            return cls.__members__[value]

        for terminal in cls:
            if terminal.value == value:
                return terminal

        return None


class CFGGrammar:
    """Context-Free Grammar with production rules as simple hashmap"""

    def __init__(self):
        # Simple hashmap: symbol -> list of production rules (list of symbols)
        self.productions: Dict[CFGNonTerminal, List[List[str]]] = {}
        self._init_productions()

    def _init_productions(self):
        """Initialize all CFG production rules"""

        # Simple hashmap approach: symbol -> [list of productions]
        self.productions = {
            # Program structure
            CFGNonTerminal.PROGRAM: [
                ["FUNCTION"]
            ],

            CFGNonTerminal.FUNCTION: [
                ["FUNCTION_HEADER", "NEWLINE", "FUNCTION_BODY"]
            ],

            CFGNonTerminal.FUNCTION_HEADER: [
                ["def", "SPACE", "IDENTIFIER", "LPAREN", "RPAREN", "COLON"]
            ],

            CFGNonTerminal.FUNCTION_BODY: [
                ["INDENT", "STATEMENT_LIST"]
            ],

            # Statement lists
            CFGNonTerminal.STATEMENT_LIST: [
                ["STATEMENT"],
                ["STATEMENT", "NEWLINE", "INDENT", "STATEMENT_LIST"]
            ],

            # Statements (ordered by preference - first match wins)
            CFGNonTerminal.STATEMENT: [
                ["ASSIGNMENT"],
                ["AUG_ASSIGNMENT"],
                ["RETURN_STMT"],
                ["IF_STMT"],
                ["FOR_STMT"],
                ["WHILE_STMT"]
            ],

            # Control flow
            CFGNonTerminal.IF_STMT: [
                ["if", "SPACE", "EXPRESSION", "COLON", "NEWLINE", "STATEMENT_LIST"]
            ],

            CFGNonTerminal.FOR_STMT: [
                ["for", "SPACE", "IDENTIFIER", "SPACE", "in", "SPACE",
                 "EXPRESSION", "COLON", "NEWLINE", "STATEMENT_LIST"]
            ],

            CFGNonTerminal.WHILE_STMT: [
                ["while", "SPACE", "EXPRESSION", "COLON", "NEWLINE", "STATEMENT_LIST"]
            ],

            CFGNonTerminal.RETURN_STMT: [
                ["return", "SPACE", "EXPRESSION"]
            ],

            # Assignments
            CFGNonTerminal.ASSIGNMENT: [
                ["IDENTIFIER", "SPACE", "=", "SPACE", "EXPRESSION"]
            ],

            CFGNonTerminal.AUG_ASSIGNMENT: [
                ["IDENTIFIER", "SPACE", "OPERATOR", "SPACE", "EXPRESSION"]
            ],

            # Expressions (ordered by preference)
            CFGNonTerminal.EXPRESSION: [
                ["BINARY_EXPR"],
                ["COMPARISON_EXPR"],
                ["BOOLEAN_EXPR"],
                ["UNARY_EXPR"],
                ["CALL_EXPR"],
                ["VARIABLE"],
                ["CONSTANT"]
            ],

            CFGNonTerminal.BINARY_EXPR: [
                ["EXPRESSION", "SPACE", "OPERATOR", "SPACE", "EXPRESSION"]
            ],

            CFGNonTerminal.UNARY_EXPR: [
                ["OPERATOR", "EXPRESSION"]
            ],

            CFGNonTerminal.COMPARISON_EXPR: [
                ["EXPRESSION", "SPACE", "OPERATOR", "SPACE", "EXPRESSION"]
            ],

            CFGNonTerminal.BOOLEAN_EXPR: [
                ["EXPRESSION", "SPACE", "OPERATOR", "SPACE", "EXPRESSION"]
            ],

            CFGNonTerminal.CALL_EXPR: [
                ["IDENTIFIER", "LPAREN", "EXPRESSION", "RPAREN"],
                ["IDENTIFIER", "LPAREN", "RPAREN"]
            ],

            # Terminals and variables
            CFGNonTerminal.VARIABLE: [
                ["IDENTIFIER"]
            ],

            # Formatting
            CFGNonTerminal.INDENT: [
                ["TAB"]
            ],

            CFGNonTerminal.NEWLINE: [
                ["\\n"]
            ],

            CFGNonTerminal.LIST_EXPR: [
                ["EXPRESSION"],
                ["EXPRESSION", "COMMA", "LIST_EXPR"]
            ],

            CFGNonTerminal.TUPLE_EXPR: [
                ["EXPRESSION"],
                ["EXPRESSION", "COMMA", "TUPLE_EXPR"]
            ],

            CFGNonTerminal.SLICE: [
                ["COLON"],
                ["EXPRESSION", "COLON"],
                ["COLON", "EXPRESSION"],
                ["EXPRESSION", "COLON", "EXPRESSION"],
                ["COLON", "COLON", "EXPRESSION"],
                ["EXPRESSION", "COLON", "COLON"],
                ["EXPRESSION", "COLON", "EXPRESSION", "COLON", "EXPRESSION"]
            ]
        }

    def get_productions(self, non_terminal: CFGNonTerminal) -> List[List[str]]:
        """Get all productions for a given non-terminal"""
        return self.productions.get(non_terminal, [])


class ASTToCFGMapper:
    """Maps AST node types to CFG non-terminals"""

    # Mapping from AST node types to CFG non-terminals
    AST_TO_CFG_MAP = {
        ASTNodeType.MODULE: CFGNonTerminal.PROGRAM,
        ASTNodeType.FUNCTION_DEF: CFGNonTerminal.FUNCTION,
        ASTNodeType.RETURN: CFGNonTerminal.RETURN_STMT,
        ASTNodeType.IF: CFGNonTerminal.IF_STMT,
        ASTNodeType.FOR: CFGNonTerminal.FOR_STMT,
        ASTNodeType.WHILE: CFGNonTerminal.WHILE_STMT,
        ASTNodeType.ASSIGNMENT: CFGNonTerminal.ASSIGNMENT,
        ASTNodeType.AUGMENTED_ASSIGNMENT: CFGNonTerminal.AUG_ASSIGNMENT,
        ASTNodeType.BINARY_OPERATION: CFGNonTerminal.BINARY_EXPR,
        ASTNodeType.UNARY_OPERATION: CFGNonTerminal.UNARY_EXPR,
        ASTNodeType.COMPARISON: CFGNonTerminal.COMPARISON_EXPR,
        ASTNodeType.BOOLEAN_OPERATION: CFGNonTerminal.BOOLEAN_EXPR,
        ASTNodeType.FUNCTION_CALL: CFGNonTerminal.CALL_EXPR,
        ASTNodeType.VARIABLE: CFGNonTerminal.VARIABLE,
        ASTNodeType.CONSTANT: CFGNonTerminal.CONSTANT,
        ASTNodeType.LIST: CFGNonTerminal.LIST_EXPR,
        ASTNodeType.TUPLE: CFGNonTerminal.TUPLE_EXPR,
    }

    # Mapping from AST operators to CFG terminals
    OPERATOR_MAP = {
        "+": CFGTerminal.PLUS,
        "-": CFGTerminal.MINUS,
        "*": CFGTerminal.MULTIPLY,
        "/": CFGTerminal.DIVIDE,
        "//": CFGTerminal.DIVIDE,  # Floor division maps to same terminal for now
        "%": CFGTerminal.MODULO,
        "**": CFGTerminal.POWER,
        "+=": CFGTerminal.PLUS_ASSIGN,
        "-=": CFGTerminal.MINUS_ASSIGN,
        "*=": CFGTerminal.MULT_ASSIGN,
        "/=": CFGTerminal.DIV_ASSIGN,
        "==": CFGTerminal.EQ,
        "!=": CFGTerminal.NE,
        "<": CFGTerminal.LT,
        "<=": CFGTerminal.LE,
        ">": CFGTerminal.GT,
        ">=": CFGTerminal.GE,
        "and": CFGTerminal.AND,
        "or": CFGTerminal.OR,
        "not": CFGTerminal.NOT,
    }

    @classmethod
    def ast_to_cfg(cls, ast_node_type: ASTNodeType) -> Optional[CFGNonTerminal]:
        """Map AST node type to CFG non-terminal"""
        return cls.AST_TO_CFG_MAP.get(ast_node_type)

    @classmethod
    def operator_to_terminal(cls, operator: str) -> Optional[CFGTerminal]:
        """Map operator string to CFG terminal"""
        return cls.OPERATOR_MAP.get(operator)


class CFGCodeGenerator:
    """Generates code from AST graphs using CFG rules"""

    def __init__(self):
        self.grammar = CFGGrammar()
        self.mapper = ASTToCFGMapper()
        self.indent_level = 0

    def generate_code_from_ast_graph(self, ast_graph: Dict[str, Any]) -> str:
        """
        Generate code from an AST graph using CFG rules deterministically.

        Args:
            ast_graph: Dictionary containing nodes, edges, and root index

        Returns:
            Generated Python code as string
        """
        nodes = ast_graph["nodes"]
        edges = ast_graph["edges"]
        root_idx = ast_graph["root"]

        # Build adjacency list for traversal
        children = self._build_children_map(len(nodes), edges)

        # Start generation from root
        return self._generate_from_node(nodes, children, root_idx)

    def _build_children_map(self, num_nodes: int, edges: List[Tuple[int, int, Any]]) -> Dict[int, List[int]]:
        """Build map from parent node to ordered list of children"""
        children = {i: [] for i in range(num_nodes)}
        siblings = {}

        for src, dst, edge_type in edges:
            if hasattr(edge_type, 'value'):
                edge_val = edge_type.value
            else:
                edge_val = edge_type

            if edge_val == EdgeType.AST.value:
                children[src].append(dst)
            elif edge_val == EdgeType.NEXT_SIBLING.value:
                siblings[src] = dst

        # Order children using sibling relationships
        for parent in children:
            if len(children[parent]) > 1:
                ordered = []
                remaining = set(children[parent])

                # Find the first child (no incoming sibling edge)
                first = None
                for child in children[parent]:
                    has_incoming_sibling = any(siblings.get(s) == child for s in remaining)
                    if not has_incoming_sibling:
                        first = child
                        break

                if first is not None:
                    current = first
                    while current is not None and current in remaining:
                        ordered.append(current)
                        remaining.remove(current)
                        current = siblings.get(current)

                    # Add any remaining children
                    ordered.extend(list(remaining))
                    children[parent] = ordered

        return children

    def _get_operator_from_node(self, node: ASTNode) -> str:
        """Extract the operator from a node if it has one"""
        if isinstance(node, ASTNodeWithOp):
            return node.op
        return ""

    def _add_parentheses_if_needed(self, expr_code: str, parent_op: str, child_op: str) -> str:
        """Add parentheses around expression if operator precedence requires it"""
        # Define operator precedence (higher number = higher precedence)
        precedence = {
            "**": 4,
            "*": 3, "/": 3, "//": 3, "%": 3,
            "+": 2, "-": 2,
            "==": 1, "!=": 1, "<": 1, "<=": 1, ">": 1, ">=": 1,
            "and": 0, "or": 0
        }

        parent_prec = precedence.get(parent_op, 0)
        child_prec = precedence.get(child_op, 0)

        # If child has lower precedence than parent, add parentheses
        if child_prec < parent_prec:
            # Only add parentheses if the child expression is actually a binary operation
            # Don't add parentheses around simple variables, constants, or function calls
            if any(op in expr_code for op in precedence.keys()):
                return f"({expr_code})"

        return expr_code

    def _generate_from_node(self, nodes: List[ASTNode], children: Dict[int, List[int]],
                           node_idx: int) -> str:
        """Generate code for a specific node and its children"""
        node_base = nodes[node_idx]
        node_type = node_base.type

        if isinstance(node_type, str):
            # Convert string to enum if needed
            try:
                node_type = ASTNodeType(node_type)
            except ValueError:
                # Fallback for unknown node types
                return self._generate_children(nodes, children, node_idx)

        # Handle specific node types directly
        if node_type == ASTNodeType.MODULE:
            # For MODULE, just process its children (function definitions)
            child_results = []
            for child_idx in children.get(node_idx, []):
                child_code = self._generate_from_node(nodes, children, child_idx)
                if child_code.strip():
                    child_results.append(child_code)
            return "\n".join(child_results)

        elif node_type == ASTNodeType.FUNCTION_DEF:
            # Generate function header and body
            node = node_base.cast(ASTFunctionDef)
            func_name = node.name
            params = node.params

            # Format parameters
            if params:
                params_str = ", ".join(params)
                header = f"def {func_name}({params_str}):"
            else:
                header = f"def {func_name}():"

            # Generate function body
            body_parts = []
            for child_idx in children.get(node_idx, []):
                child_code = self._generate_from_node(nodes, children, child_idx)
                if child_code.strip():
                    # Add indentation to each line
                    indented = "\n".join(f"    {line}" for line in child_code.split("\n") if line.strip())
                    body_parts.append(indented)

            body = "\n".join(body_parts)
            return f"{header}\n{body}"

        elif node_type == ASTNodeType.ASSIGNMENT:
            # Generate assignment: target = value
            child_indices = children.get(node_idx, [])
            if len(child_indices) >= 2:
                # Handle tuple assignments: a, b = b, a % b
                # First children are targets, last child is the value
                targets = []
                for i in range(len(child_indices) - 1):
                    target_code = self._generate_from_node(nodes, children, child_indices[i])
                    if target_code.strip():
                        targets.append(target_code)

                value_code = self._generate_from_node(nodes, children, child_indices[-1])

                if len(targets) == 1:
                    return f"{targets[0]} = {value_code}"
                else:
                    targets_str = ", ".join(targets)
                    return f"{targets_str} = {value_code}"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.AUGMENTED_ASSIGNMENT:
            # Generate augmented assignment: target op= value
            child_indices = children.get(node_idx, [])
            node = node_base.cast(ASTAugmentedAssignment)
            op = node.op
            if len(child_indices) >= 2:
                target_code = self._generate_from_node(nodes, children, child_indices[0])
                value_code = self._generate_from_node(nodes, children, child_indices[1])
                return f"{target_code} {op} {value_code}"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.BINARY_OPERATION:
            # Generate binary operation: left op right
            child_indices = children.get(node_idx, [])
            node = node_base.cast(ASTBinaryOperation)
            op = node.op
            if len(child_indices) >= 2:
                left_code = self._generate_from_node(nodes, children, child_indices[0])
                right_code = self._generate_from_node(nodes, children, child_indices[1])

                # Add parentheses if needed based on operator precedence
                left_code = self._add_parentheses_if_needed(left_code, op, self._get_operator_from_node(nodes[child_indices[0]]))
                right_code = self._add_parentheses_if_needed(right_code, op, self._get_operator_from_node(nodes[child_indices[1]]))

                return f"{left_code} {op} {right_code}"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.RETURN:
            # Generate return statement
            child_indices = children.get(node_idx, [])
            node = node_base.cast(ASTReturn)
            if child_indices:
                value_code = self._generate_from_node(nodes, children, child_indices[0])
                return f"return {value_code}"
            else:
                return "return"

        elif node_type == ASTNodeType.IF:
            # Generate if statement with proper else handling
            child_indices = children.get(node_idx, [])
            node = node_base.cast(ASTIf)
            if len(child_indices) >= 2:
                # First child is the test condition
                test_code = self._generate_from_node(nodes, children, child_indices[0])
                header = f"if {test_code}:"

                # Get body_len and orelse_len from node attributes
                body_len = node.body_len
                orelse_len = node.orelse_len

                # Generate if body statements (after test, take body_len children)
                if_body_parts = []
                for i in range(1, 1 + body_len):
                    if i < len(child_indices):
                        child_code = self._generate_from_node(nodes, children, child_indices[i])
                        if child_code.strip():
                            # Add indentation
                            indented = "\n".join(f"    {line}" for line in child_code.split("\n") if line.strip())
                            if_body_parts.append(indented)

                if_body = "\n".join(if_body_parts)
                result = f"{header}\n{if_body}"

                # Generate else body if present
                if orelse_len > 0:
                    result += "\nelse:"
                    else_body_parts = []
                    for i in range(1 + body_len, 1 + body_len + orelse_len):
                        if i < len(child_indices):
                            child_code = self._generate_from_node(nodes, children, child_indices[i])
                            if child_code.strip():
                                # Add indentation
                                indented = "\n".join(f"    {line}" for line in child_code.split("\n") if line.strip())
                                else_body_parts.append(indented)

                    if else_body_parts:
                        else_body = "\n".join(else_body_parts)
                        result += f"\n{else_body}"

                return result
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.FOR:
            node = node_base.cast(ASTFor)
            # Generate for loop
            child_indices = children.get(node_idx, [])
            if len(child_indices) >= 3:
                # First child is target, second is iter, rest are body
                target_code = self._generate_from_node(nodes, children, child_indices[0])
                iter_code = self._generate_from_node(nodes, children, child_indices[1])
                header = f"for {target_code} in {iter_code}:"

                # Generate body statements
                body_parts = []
                for child_idx in child_indices[2:]:
                    child_code = self._generate_from_node(nodes, children, child_idx)
                    if child_code.strip():
                        # Add indentation
                        indented = "\n".join(f"    {line}" for line in child_code.split("\n") if line.strip())
                        body_parts.append(indented)

                body = "\n".join(body_parts)
                return f"{header}\n{body}"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.WHILE:
            node = node_base.cast(ASTWhile)
            # Generate while loop
            child_indices = children.get(node_idx, [])
            if len(child_indices) >= 2:
                # First child is test, rest are body
                test_code = self._generate_from_node(nodes, children, child_indices[0])
                header = f"while {test_code}:"

                # Generate body statements
                body_parts = []
                for child_idx in child_indices[1:]:
                    child_code = self._generate_from_node(nodes, children, child_idx)
                    if child_code.strip():
                        # Add indentation
                        indented = "\n".join(f"    {line}" for line in child_code.split("\n") if line.strip())
                        body_parts.append(indented)

                body = "\n".join(body_parts)
                return f"{header}\n{body}"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.COMPARISON:
            node = node_base.cast(ASTComparison)
            # Generate comparison: left op right
            child_indices = children.get(node_idx, [])
            op = node.op
            if len(child_indices) >= 2:
                left_code = self._generate_from_node(nodes, children, child_indices[0])
                right_code = self._generate_from_node(nodes, children, child_indices[1])

                # Add parentheses if needed based on operator precedence
                left_code = self._add_parentheses_if_needed(left_code, op, self._get_operator_from_node(nodes[child_indices[0]]))
                right_code = self._add_parentheses_if_needed(right_code, op, self._get_operator_from_node(nodes[child_indices[1]]))

                return f"{left_code} {op} {right_code}"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.BOOLEAN_OPERATION:
            node = node_base.cast(ASTBooleanOperation)
            # Generate boolean operation: left op right
            child_indices = children.get(node_idx, [])
            op = node.op
            if len(child_indices) >= 2:
                left_code = self._generate_from_node(nodes, children, child_indices[0])
                right_code = self._generate_from_node(nodes, children, child_indices[1])

                # Add parentheses if needed based on operator precedence
                left_code = self._add_parentheses_if_needed(left_code, op, self._get_operator_from_node(nodes[child_indices[0]]))
                right_code = self._add_parentheses_if_needed(right_code, op, self._get_operator_from_node(nodes[child_indices[1]]))

                return f"{left_code} {op} {right_code}"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.UNARY_OPERATION:
            # Generate unary operation: op operand
            child_indices = children.get(node_idx, [])
            node = node_base.cast(ASTUnaryOperation)
            op = node.op
            if child_indices:
                operand_code = self._generate_from_node(nodes, children, child_indices[0])
                # Add parentheses around operand if it's a binary operation
                operand_code = self._add_parentheses_if_needed(operand_code, op, self._get_operator_from_node(nodes[child_indices[0]]))
                return f"{op}{operand_code}"
            else:
                return op

        elif node_type == ASTNodeType.FUNCTION_CALL:
            node = node_base.cast(ASTFunctionCall)
            # Generate function call (including method calls)
            child_indices = children.get(node_idx, [])
            func_name = node.function

            if child_indices:
                # Check if first child is an ATTRIBUTE (method call)
                first_child = nodes[child_indices[0]]
                if first_child.type == ASTNodeType.ATTRIBUTE:
                    # This is a method call: obj.method(args)
                    method_code = self._generate_from_node(nodes, children, child_indices[0])

                    # Get arguments (skip the method/attribute)
                    args = []
                    kw_args = []
                    for child_idx in child_indices[1:]:
                        child_node = nodes[child_idx]
                        if child_node.type == ASTNodeType.KEYWORD_ARG:
                            kw_args.append(self._generate_from_node(nodes, children, child_idx))
                        else:
                            arg_code = self._generate_from_node(nodes, children, child_idx)
                            if arg_code.strip():
                                args.append(arg_code)

                    # Combine positional and keyword arguments
                    all_args = args + kw_args
                    args_str = ", ".join(all_args)
                    return f"{method_code}({args_str})"

                elif func_name:
                    # Regular function call with known name
                    # Skip first child if it's a VARIABLE with the same name as the function
                    args_start = 0
                    if child_indices and len(child_indices) > 0:
                        first_child = nodes[child_indices[0]].cast(ASTVariable)
                        if (first_child.type == ASTNodeType.VARIABLE and
                            first_child.name == func_name):
                            args_start = 1  # Skip the function name variable

                    args = []
                    kw_args = []
                    for child_idx in child_indices[args_start:]:
                        child_node = nodes[child_idx]
                        if child_node.type == ASTNodeType.KEYWORD_ARG:
                            kw_args.append(self._generate_from_node(nodes, children, child_idx))
                        else:
                            arg_code = self._generate_from_node(nodes, children, child_idx)
                            if arg_code.strip():
                                args.append(arg_code)

                    # Combine positional and keyword arguments
                    all_args = args + kw_args
                    args_str = ", ".join(all_args)
                    return f"{func_name}({args_str})"

                else:
                    # Function call where first child is the function
                    func_code = self._generate_from_node(nodes, children, child_indices[0])

                    # Get arguments (skip the function)
                    args = []
                    kw_args = []
                    for child_idx in child_indices[1:]:
                        child_node = nodes[child_idx]
                        if child_node.type == ASTNodeType.KEYWORD_ARG:
                            kw_args.append(self._generate_from_node(nodes, children, child_idx))
                        else:
                            arg_code = self._generate_from_node(nodes, children, child_idx)
                            if arg_code.strip():
                                args.append(arg_code)

                    # Combine positional and keyword arguments
                    all_args = args + kw_args
                    args_str = ", ".join(all_args)
                    return f"{func_code}({args_str})"
            else:
                # No children, use function name if available
                if func_name:
                    return f"{func_name}()"
                else:
                    return "func()"

        elif node_type == ASTNodeType.KEYWORD_ARG:
            # Generate keyword argument: name=value
            node = node_base.cast(ASTKeywordArg)
            arg_name = node.arg
            child_indices = children.get(node_idx, [])
            if child_indices:
                value_code = self._generate_from_node(nodes, children, child_indices[0])
                return f"{arg_name}={value_code}"
            else:
                return f"{arg_name}=None"

        elif node_type == ASTNodeType.VARIABLE:
            # Generate variable name
            node = node_base.cast(ASTVariable)
            return node.name

        elif node_type == ASTNodeType.CONSTANT:
            # Generate constant value
            node = node_base.cast(ASTConstant)
            value = node.value
            if isinstance(value, str):
                return f'"{value}"'
            else:
                return str(value)

        elif node_type == ASTNodeType.LIST:
            # Generate list literal
            child_indices = children.get(node_idx, [])
            if child_indices:
                # List with elements
                elements = []
                for child_idx in child_indices:
                    element_code = self._generate_from_node(nodes, children, child_idx)
                    if element_code.strip():
                        elements.append(element_code)
                return f"[{', '.join(elements)}]"
            else:
                # Empty list
                return "[]"

        elif node_type == ASTNodeType.TUPLE:
            # Generate tuple literal
            child_indices = children.get(node_idx, [])
            if child_indices:
                # Tuple with elements
                elements = []
                for child_idx in child_indices:
                    element_code = self._generate_from_node(nodes, children, child_idx)
                    if element_code.strip():
                        elements.append(element_code)
                return f"({', '.join(elements)})"
            else:
                # Empty tuple
                return "()"

        elif node_type == ASTNodeType.SLICE:
            # Generate slice operation: start:stop:step (without brackets)
            # Brackets are added by the SUBSCRIPT node that contains this slice
            child_indices = children.get(node_idx, [])
            if len(child_indices) >= 3:
                # Full slice: start:stop:step
                start_code = self._generate_from_node(nodes, children, child_indices[0]) if child_indices[0] else ""
                stop_code = self._generate_from_node(nodes, children, child_indices[1]) if child_indices[1] else ""
                step_code = self._generate_from_node(nodes, children, child_indices[2]) if child_indices[2] else ""

                # Handle the case where start and stop are None (empty slice)
                if not start_code.strip() and not stop_code.strip():
                    if step_code.strip():
                        return f"::{step_code}"
                    else:
                        return "::"
                else:
                    slice_parts = []
                    if start_code.strip():
                        slice_parts.append(start_code)
                    slice_parts.append("")  # Always include the colon
                    if stop_code.strip():
                        slice_parts.append(stop_code)
                    if step_code.strip():
                        slice_parts.append("")
                        slice_parts.append(step_code)

                    return ":".join(slice_parts)
            elif len(child_indices) == 2:
                # Partial slice: start:stop or :stop:step
                first_code = self._generate_from_node(nodes, children, child_indices[0]) if child_indices[0] else ""
                second_code = self._generate_from_node(nodes, children, child_indices[1]) if child_indices[1] else ""

                if first_code.strip() and second_code.strip():
                    return f"{first_code}:{second_code}"
                elif first_code.strip():
                    return f"{first_code}:"
                else:
                    return f":{second_code}"
            elif len(child_indices) == 1:
                # Single slice: start: or :stop
                code = self._generate_from_node(nodes, children, child_indices[0])
                if code.strip():
                    return f"{code}:"
                else:
                    return ":"
            else:
                # Empty slice: :
                return ":"

        elif node_type == ASTNodeType.SUBSCRIPT:
            # Generate subscript access: obj[index]
            child_indices = children.get(node_idx, [])
            if len(child_indices) >= 2:
                obj_code = self._generate_from_node(nodes, children, child_indices[0])
                index_code = self._generate_from_node(nodes, children, child_indices[1])
                return f"{obj_code}[{index_code}]"
            else:
                return self._generate_children(nodes, children, node_idx)

        elif node_type == ASTNodeType.ATTRIBUTE:
            # Generate attribute access: obj.attr
            child_indices = children.get(node_idx, [])
            node = node_base.cast(ASTAttribute)
            attr_name = node.attr
            if child_indices:
                obj_code = self._generate_from_node(nodes, children, child_indices[0])
                return f"{obj_code}.{attr_name}"
            else:
                return f".{attr_name}"

        elif node_type == ASTNodeType.EXPRESSION:
            # Expression wrapper - just return the child content
            child_indices = children.get(node_idx, [])
            if child_indices:
                return self._generate_from_node(nodes, children, child_indices[0])
            else:
                # Empty expression - likely a pass statement
                return "pass"

        # For other node types, try CFG mapping
        cfg_symbol = self.mapper.ast_to_cfg(node_type)
        if cfg_symbol is not None:
            production = self._select_production(cfg_symbol, node_base, children.get(node_idx, []))
            if production is not None:
                return self._apply_production(production, node_base, nodes, children, node_idx)

        # Fallback: generate children directly
        return self._generate_children(nodes, children, node_idx)

    def _select_production(self, non_terminal: CFGNonTerminal, node: ASTNode,
                          child_indices: List[int]) -> Optional[List[str]]:
        """Select the best production rule for a given non-terminal"""
        productions = self.grammar.get_productions(non_terminal)
        if not productions:
            return None

        # Find first production that structurally matches
        for production in productions:
            if self._production_matches_structure(production, node, child_indices):
                return production

        # Fallback: return first production
        return productions[0]

    def _production_matches_structure(self, production: List[str], node: ASTNode,
                                    child_indices: List[int]) -> bool:
        """Check if a production matches the AST node structure"""
        # Count expected non-terminals in production
        expected_non_terminals = sum(1 for symbol in production
                                   if symbol.isupper() and not symbol.startswith("\\"))
        actual_children = len(child_indices)

        # Production matches if child count is compatible
        if expected_non_terminals == actual_children:
            return True

        # Also check if operator matches for expression productions
        if isinstance(node, ASTNodeWithOp):
            operator = node.op
            terminal = self.mapper.operator_to_terminal(operator)
            if operator in production or (terminal and terminal.value in production):
                return True

        return False

    def _apply_production(self, production: List[str], node: ASTNode,
                         nodes: List[ASTNode], children: Dict[int, List[int]],
                         node_idx: int) -> str:
        """Apply a production rule to generate code"""
        result = []
        child_indices = children.get(node_idx, [])
        child_idx = 0

        for symbol in production:
            if symbol == "SPACE":
                result.append(" ")
            elif symbol == "TAB":
                result.append("\t")
            elif symbol == "\\n":
                result.append("\n")
            elif symbol == "NEWLINE":
                result.append("\n")
            elif symbol == "INDENT":
                result.append("    ")  # Use 4 spaces for indentation
            elif symbol.startswith("\\"):
                # Escaped characters
                result.append(symbol[1:])
            elif symbol.islower() or symbol in ["(", ")", "[", "]", ":", ",", "=", "+", "-", "*", "/",
                                               "==", "!=", "<", "<=", ">", ">="]:
                # Terminal symbols
                result.append(symbol)
            elif symbol == "IDENTIFIER":
                # Use node's name attribute or generate from variable info
                if isinstance(node, ASTNodeWithName):
                    result.append(node.name)
                elif isinstance(node, ASTVariable):
                    result.append(f"var_{node.var_id}")
                else:
                    result.append("x")
            elif symbol == "OPERATOR":
                # Use node's operator attribute
                if isinstance(node, ASTNodeWithOp):
                    result.append(node.op)
                else:
                    result.append("+")  # fallback
            elif symbol == "CONSTANT":
                # Use node's value
                if isinstance(node, ASTConstant):
                    value = node.value
                    if isinstance(value, str):
                        result.append(f'"{value}"')
                    else:
                        result.append(str(value))
                else:
                    result.append("0")
            elif symbol.isupper():
                # Non-terminal, recursively generate from child
                if child_idx < len(child_indices):
                    child_code = self._generate_from_node(nodes, children, child_indices[child_idx])
                    result.append(child_code)
                    child_idx += 1
                else:
                    # No more children available, use empty string or default
                    result.append("")
            else:
                result.append(symbol)

        return "".join(result)

    def _generate_children(self, nodes: List[ASTNode], children: Dict[int, List[int]],
                          node_idx: int) -> str:
        """Fallback: generate code by traversing children directly"""
        child_indices = children.get(node_idx, [])
        results = []

        for child_idx in child_indices:
            child_code = self._generate_from_node(nodes, children, child_idx)
            if child_code.strip():
                results.append(child_code)

        return " ".join(results)


def generate_code_from_ast(code: str) -> str:
    """
    Convenience function to parse code, convert to AST graph, and regenerate code.

    Args:
        code: Input Python code string

    Returns:
        Regenerated Python code using CFG rules
    """
    from .ast import ASTSimplifier

    # Parse code to AST graph
    ast_graph = ASTSimplifier.ast_to_graph(code)

    # Generate code using CFG
    generator = CFGCodeGenerator()
    return generator.generate_code_from_ast_graph(ast_graph)


if __name__ == "__main__":
    # Test the CFG system
    test_code = """
def test_function():
    x = 5
    y = x + 3
    return y
"""

    regenerated = generate_code_from_ast(test_code)
    print("Original:")
    print(test_code)
    print("\nRegenerated:")
    print(regenerated)
