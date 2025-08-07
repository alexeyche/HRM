import ast
from typing import Dict, Any
from enum import Enum


class ASTNodeType(Enum):
    """Enumeration of AST node types for minimal language graph neural network modeling"""

    # Control Flow
    FOR = "for"                     # For loop
    WHILE = "while"                 # While loop
    IF = "if"                       # If statement

    # Variables and Names
    VARIABLE = "variable"           # Variable reference (Name node)
    ASSIGNMENT = "assignment"       # Assignment statement
    AUGMENTED_ASSIGNMENT = "augmented_assignment"  # +=, -=, etc.

    # Expressions
    BINARY_OPERATION = "binary_operation"  # +, -, *, /, %, **, etc.
    UNARY_OPERATION = "unary_operation"    # +, -, not
    COMPARISON = "comparison"       # ==, !=, <, <=, >, >=
    BOOLEAN_OPERATION = "boolean_operation"  # and, or
    FUNCTION_CALL = "function_call" # Function call
    ATTRIBUTE = "attribute"         # Object attribute access
    SUBSCRIPT = "subscript"         # Indexing/slicing
    SLICE = "slice"                 # Slice operation

    # Data Structures
    LIST = "list"                   # List literal

    # Literals
    CONSTANT = "constant"           # Literal values (numbers, strings, etc.)
    EXPRESSION = "expression"       # Expression wrapper


class ASTSimplifier:
    """Converts Python AST to simplified graph representation"""

    @staticmethod
    def python_to_ast(code: str) -> ast.AST:
        """Parse Python code to AST"""
        return ast.parse(code)

    @staticmethod
    def ast_to_simplified_json(code: str) -> Dict[str, Any]:
        """Convert Python code to human-readable JSON AST"""
        tree = ASTSimplifier.python_to_ast(code)

        def traverse_to_readable(node) -> Any:
            """Convert AST node to human-readable format"""
            if isinstance(node, ast.FunctionDef):
                return {
                    "function": {
                        "name": node.name,
                        "params": [arg.arg for arg in node.args.args],
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
            elif isinstance(node, ast.Return):
                return {"return": traverse_to_readable(node.value)}
            elif isinstance(node, ast.For):
                return {
                    "for": {
                        "target": traverse_to_readable(node.target),
                        "iter": traverse_to_readable(node.iter),
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
            elif isinstance(node, ast.While):
                return {
                    "while": {
                        "test": traverse_to_readable(node.test),
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
            elif isinstance(node, ast.If):
                result = {
                    "if": {
                        "test": traverse_to_readable(node.test),
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
                if node.orelse:
                    result["if"]["else"] = [traverse_to_readable(stmt) for stmt in node.orelse]
                return result
            elif isinstance(node, ast.Name):
                return {"var": node.id}
            elif isinstance(node, ast.BinOp):
                op_map: Dict[Any, str] = {
                    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                    ast.Mod: "%", ast.Pow: "**"
                }
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {op_name: [traverse_to_readable(node.left), traverse_to_readable(node.right)]}
            elif isinstance(node, ast.UnaryOp):
                op_map = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not"}
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {op_name: traverse_to_readable(node.operand)}
            elif isinstance(node, ast.Compare):
                op_map = {
                    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                    ast.Gt: ">", ast.GtE: ">="
                }
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op_name = op_map.get(type(node.ops[0]), str(type(node.ops[0]).__name__))
                    return {op_name: [traverse_to_readable(node.left), traverse_to_readable(node.comparators[0])]}
                else:
                    # Multiple comparisons
                    return {
                        "compare": {
                            "left": traverse_to_readable(node.left),
                            "ops": [op_map.get(type(op), str(type(op).__name__)) for op in node.ops],
                            "comparators": [traverse_to_readable(comp) for comp in node.comparators]
                        }
                    }
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    return {
                        "call": {
                            "function": node.func.id,
                            "args": [traverse_to_readable(arg) for arg in node.args]
                        }
                    }
                else:
                    return {
                        "call": {
                            "function": traverse_to_readable(node.func),
                            "args": [traverse_to_readable(arg) for arg in node.args]
                        }
                    }
            elif isinstance(node, ast.Constant):
                return {"const": node.value}
            elif isinstance(node, ast.List):
                return {"list": [traverse_to_readable(elt) for elt in node.elts]}
            elif isinstance(node, ast.Tuple):
                return {"list": [traverse_to_readable(elt) for elt in node.elts]}
            elif isinstance(node, ast.AugAssign):
                op_map = {
                    ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/=",
                    ast.Mod: "%=", ast.Pow: "**="
                }
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {
                    "aug_assign": {
                        "target": traverse_to_readable(node.target),
                        "op": op_name,
                        "value": traverse_to_readable(node.value)
                    }
                }
            elif isinstance(node, ast.Assign):
                return {"assign": [traverse_to_readable(elt) for elt in node.targets]}
            elif isinstance(node, ast.Expr):
                return {"expr": traverse_to_readable(node.value)}
            elif isinstance(node, ast.Attribute):
                return {
                    "attribute": {
                        "value": traverse_to_readable(node.value),
                        "attr": node.attr
                    }
                }
            elif isinstance(node, ast.Subscript):
                return {
                    "subscript": {
                        "value": traverse_to_readable(node.value),
                        "slice": traverse_to_readable(node.slice)
                    }
                }
            elif isinstance(node, ast.Slice):
                return {
                    "slice": {
                        "lower": traverse_to_readable(node.lower) if node.lower else None,
                        "upper": traverse_to_readable(node.upper) if node.upper else None,
                        "step": traverse_to_readable(node.step) if node.step else None
                    }
                }
            elif isinstance(node, ast.BoolOp):
                op_map: Dict[Any, str] = {ast.And: "and", ast.Or: "or"}
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {
                    "bool_op": {
                        "op": op_name,
                        "values": [traverse_to_readable(val) for val in node.values]
                    }
                }
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")

        # Start traversal from the function definition
        result = traverse_to_readable(tree.body[0])  # type: ignore

        return result

