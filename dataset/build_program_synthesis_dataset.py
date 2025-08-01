from typing import List, Dict, Any, Tuple
import os
import json
import ast
import yaml
import numpy as np
from dataclasses import dataclass

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_name: str = "simple_math"
    output_dir: str = "data/program-synthesis-simple-math"
    num_samples: int = 100
    seed: int = 42

    # AST processing config
    max_nodes: int = 50
    examples_per_program: int = 20


# AST Node Type Vocabulary (30 types)
class NodeType:
    # Control Flow (6 types)
    FUNC_DEF = 0
    RETURN = 1
    FOR_LOOP = 2
    WHILE_LOOP = 3
    IF_STMT = 4
    ELSE_STMT = 5

    # Variables & Parameters (4 types)
    VAR_PARAM = 6
    VAR_LOCAL = 7
    VAR_ITER = 8
    VAR_TEMP = 9

    # Mathematical Operations (8 types)
    OP_ADD = 10
    OP_SUB = 11
    OP_MUL = 12
    OP_DIV = 13
    OP_MOD = 14
    OP_POW = 15
    OP_NEG = 16
    OP_ABS = 17

    # Comparison Operations (6 types)
    OP_EQ = 18
    OP_NE = 19
    OP_LT = 20
    OP_LE = 21
    OP_GT = 22
    OP_GE = 23

    # Built-in Functions (4 types)
    OP_BUILTIN_SUM = 24
    OP_BUILTIN_RANGE = 25
    OP_BUILTIN_LEN = 26
    OP_BUILTIN_MIN = 27

    # Constants (2 types)
    CONST_INT = 28
    CONST_BOOL = 29


@dataclass
class ProgramSpec:
    name: str
    description: str
    inputs: List[Dict[str, str]]
    outputs: List[Dict[str, str]]
    examples: List[Dict[str, Any]]
    reference_impl: str


# Program Templates for Generation
PROGRAM_TEMPLATES = {
    "sum_up_to_n": {
        "description": "Sum up all numbers up to the input number N",
        "inputs": [{"type": "int", "description": "The input number N"}],
        "outputs": [{"type": "int", "description": "The sum of all numbers up to N"}],
        "base_examples": [
            {"input": 5, "output": 15},
            {"input": 10, "output": 55},
            {"input": 3, "output": 6},
            {"input": 1, "output": 1},
            {"input": 0, "output": 0}
        ],
        "implementation": """def program(n):
    return sum(range(1, n + 1))"""
    },

    "max_of_two": {
        "description": "Return the larger of two numbers",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"}
        ],
        "outputs": [{"type": "int", "description": "The larger number"}],
        "base_examples": [
            {"input": [5, 3], "output": 5},
            {"input": [10, 15], "output": 15},
            {"input": [-2, -5], "output": -2},
            {"input": [0, 0], "output": 0},
            {"input": [7, 7], "output": 7}
        ],
        "implementation": """def program(a, b):
    return a if a > b else b"""
    },

    "absolute_value": {
        "description": "Return the absolute value of a number",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The absolute value"}],
        "base_examples": [
            {"input": -5, "output": 5},
            {"input": 3, "output": 3},
            {"input": 0, "output": 0},
            {"input": -100, "output": 100},
            {"input": 42, "output": 42}
        ],
        "implementation": """def program(n):
    return n if n >= 0 else -n"""
    },

    "is_even": {
        "description": "Check if a number is even",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "bool", "description": "True if even, False if odd"}],
        "base_examples": [
            {"input": 4, "output": True},
            {"input": 7, "output": False},
            {"input": 0, "output": True},
            {"input": -2, "output": True},
            {"input": -3, "output": False}
        ],
        "implementation": """def program(n):
    return n % 2 == 0"""
    },

    "factorial": {
        "description": "Calculate the factorial of a number",
        "inputs": [{"type": "int", "description": "The input number (non-negative)"}],
        "outputs": [{"type": "int", "description": "The factorial"}],
        "base_examples": [
            {"input": 0, "output": 1},
            {"input": 1, "output": 1},
            {"input": 3, "output": 6},
            {"input": 4, "output": 24},
            {"input": 5, "output": 120}
        ],
        "implementation": """def program(n):
    if n <= 1:
        return 1
    return n * program(n - 1)"""
    },

    "power_of_two": {
        "description": "Calculate 2 to the power of n",
        "inputs": [{"type": "int", "description": "The exponent"}],
        "outputs": [{"type": "int", "description": "2^n"}],
        "base_examples": [
            {"input": 0, "output": 1},
            {"input": 1, "output": 2},
            {"input": 3, "output": 8},
            {"input": 4, "output": 16},
            {"input": 5, "output": 32}
        ],
        "implementation": """def program(n):
    return 2 ** n"""
    },

    "square": {
        "description": "Calculate the square of a number",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The square of the number"}],
        "base_examples": [
            {"input": 0, "output": 0},
            {"input": 1, "output": 1},
            {"input": 3, "output": 9},
            {"input": -4, "output": 16},
            {"input": 5, "output": 25}
        ],
        "implementation": """def program(n):
    return n * n"""
    },

    "cube": {
        "description": "Calculate the cube of a number",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The cube of the number"}],
        "base_examples": [
            {"input": 0, "output": 0},
            {"input": 1, "output": 1},
            {"input": 2, "output": 8},
            {"input": -3, "output": -27},
            {"input": 4, "output": 64}
        ],
        "implementation": """def program(n):
    return n * n * n"""
    },

    "min_of_two": {
        "description": "Return the smaller of two numbers",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"}
        ],
        "outputs": [{"type": "int", "description": "The smaller number"}],
        "base_examples": [
            {"input": [5, 3], "output": 3},
            {"input": [10, 15], "output": 10},
            {"input": [-2, -5], "output": -5},
            {"input": [0, 0], "output": 0},
            {"input": [7, 7], "output": 7}
        ],
        "implementation": """def program(a, b):
    return a if a < b else b"""
    },

    "is_positive": {
        "description": "Check if a number is positive",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "bool", "description": "True if positive, False otherwise"}],
        "base_examples": [
            {"input": 5, "output": True},
            {"input": -3, "output": False},
            {"input": 0, "output": False},
            {"input": 10, "output": True},
            {"input": -1, "output": False}
        ],
        "implementation": """def program(n):
    return n > 0"""
    },

    "is_negative": {
        "description": "Check if a number is negative",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "bool", "description": "True if negative, False otherwise"}],
        "base_examples": [
            {"input": -5, "output": True},
            {"input": 3, "output": False},
            {"input": 0, "output": False},
            {"input": -10, "output": True},
            {"input": 1, "output": False}
        ],
        "implementation": """def program(n):
    return n < 0"""
    },

    "double": {
        "description": "Double a number (multiply by 2)",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The number multiplied by 2"}],
        "base_examples": [
            {"input": 0, "output": 0},
            {"input": 3, "output": 6},
            {"input": -4, "output": -8},
            {"input": 10, "output": 20},
            {"input": -1, "output": -2}
        ],
        "implementation": """def program(n):
    return n * 2"""
    },

    "half": {
        "description": "Divide a number by 2 (integer division)",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The number divided by 2"}],
        "base_examples": [
            {"input": 0, "output": 0},
            {"input": 6, "output": 3},
            {"input": -8, "output": -4},
            {"input": 10, "output": 5},
            {"input": 5, "output": 2}
        ],
        "implementation": """def program(n):
    return n // 2"""
    },

    "add_ten": {
        "description": "Add 10 to a number",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The number plus 10"}],
        "base_examples": [
            {"input": 0, "output": 10},
            {"input": 5, "output": 15},
            {"input": -3, "output": 7},
            {"input": 10, "output": 20},
            {"input": -15, "output": -5}
        ],
        "implementation": """def program(n):
    return n + 10"""
    },

    "subtract_five": {
        "description": "Subtract 5 from a number",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The number minus 5"}],
        "base_examples": [
            {"input": 10, "output": 5},
            {"input": 5, "output": 0},
            {"input": 0, "output": -5},
            {"input": -3, "output": -8},
            {"input": 15, "output": 10}
        ],
        "implementation": """def program(n):
    return n - 5"""
    },

    "is_zero": {
        "description": "Check if a number is zero",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "bool", "description": "True if zero, False otherwise"}],
        "base_examples": [
            {"input": 0, "output": True},
            {"input": 1, "output": False},
            {"input": -1, "output": False},
            {"input": 10, "output": False},
            {"input": -5, "output": False}
        ],
        "implementation": """def program(n):
    return n == 0"""
    },

    "sum_of_two": {
        "description": "Add two numbers together",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"}
        ],
        "outputs": [{"type": "int", "description": "The sum of the two numbers"}],
        "base_examples": [
            {"input": [3, 5], "output": 8},
            {"input": [0, 0], "output": 0},
            {"input": [-2, 7], "output": 5},
            {"input": [10, -3], "output": 7},
            {"input": [-5, -3], "output": -8}
        ],
        "implementation": """def program(a, b):
    return a + b"""
    },

    "difference": {
        "description": "Calculate the difference between two numbers (a - b)",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"}
        ],
        "outputs": [{"type": "int", "description": "The difference (a - b)"}],
        "base_examples": [
            {"input": [8, 3], "output": 5},
            {"input": [5, 5], "output": 0},
            {"input": [2, 7], "output": -5},
            {"input": [10, -3], "output": 13},
            {"input": [-5, -2], "output": -3}
        ],
        "implementation": """def program(a, b):
    return a - b"""
    },

    "product": {
        "description": "Multiply two numbers together",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"}
        ],
        "outputs": [{"type": "int", "description": "The product of the two numbers"}],
        "base_examples": [
            {"input": [3, 4], "output": 12},
            {"input": [0, 5], "output": 0},
            {"input": [-2, 3], "output": -6},
            {"input": [7, -2], "output": -14},
            {"input": [-3, -4], "output": 12}
        ],
        "implementation": """def program(a, b):
    return a * b"""
    },

    "count_up_to_n": {
        "description": "Count the number of integers from 1 to n (inclusive)",
        "inputs": [{"type": "int", "description": "The upper limit"}],
        "outputs": [{"type": "int", "description": "The count (which is just n)"}],
        "base_examples": [
            {"input": 1, "output": 1},
            {"input": 5, "output": 5},
            {"input": 10, "output": 10},
            {"input": 0, "output": 0},
            {"input": 3, "output": 3}
        ],
        "implementation": """def program(n):
    return n if n >= 0 else 0"""
    },

    "is_greater_than_ten": {
        "description": "Check if a number is greater than 10",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "bool", "description": "True if greater than 10, False otherwise"}],
        "base_examples": [
            {"input": 15, "output": True},
            {"input": 10, "output": False},
            {"input": 5, "output": False},
            {"input": 11, "output": True},
            {"input": -5, "output": False}
        ],
        "implementation": """def program(n):
    return n > 10"""
    },

    "remainder_by_three": {
        "description": "Calculate the remainder when dividing by 3",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The remainder when divided by 3"}],
        "base_examples": [
            {"input": 9, "output": 0},
            {"input": 10, "output": 1},
            {"input": 11, "output": 2},
            {"input": 7, "output": 1},
            {"input": 6, "output": 0}
        ],
        "implementation": """def program(n):
    return n % 3"""
    },

    "is_divisible_by_five": {
        "description": "Check if a number is divisible by 5",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "bool", "description": "True if divisible by 5, False otherwise"}],
        "base_examples": [
            {"input": 10, "output": True},
            {"input": 15, "output": True},
            {"input": 7, "output": False},
            {"input": 0, "output": True},
            {"input": -5, "output": True}
        ],
        "implementation": """def program(n):
    return n % 5 == 0"""
    },

    "triple": {
        "description": "Multiply a number by 3",
        "inputs": [{"type": "int", "description": "The input number"}],
        "outputs": [{"type": "int", "description": "The number multiplied by 3"}],
        "base_examples": [
            {"input": 0, "output": 0},
            {"input": 2, "output": 6},
            {"input": -3, "output": -9},
            {"input": 5, "output": 15},
            {"input": -1, "output": -3}
        ],
        "implementation": """def program(n):
    return n * 3"""
    }
}


class ASTSimplifier:
    """Converts Python AST to simplified graph representation"""

    def __init__(self, max_nodes: int = 50):
        self.max_nodes = max_nodes

    def python_to_ast(self, code: str) -> ast.AST:
        """Parse Python code to AST"""
        return ast.parse(code)

    def simplify_ast_node(self, node: ast.AST) -> Dict[str, Any]:
        """Convert AST node to simplified representation"""
        if isinstance(node, ast.FunctionDef):
            return {"type": NodeType.FUNC_DEF, "params": len(node.args.args)}
        elif isinstance(node, ast.Return):
            return {"type": NodeType.RETURN}
        elif isinstance(node, ast.For):
            return {"type": NodeType.FOR_LOOP}
        elif isinstance(node, ast.While):
            return {"type": NodeType.WHILE_LOOP}
        elif isinstance(node, ast.If):
            return {"type": NodeType.IF_STMT}
        elif hasattr(ast, 'Else') and isinstance(node, ast.Else):
            return {"type": NodeType.ELSE_STMT}
        elif isinstance(node, ast.Name):
            # Simplified variable handling
            return {"type": NodeType.VAR_PARAM, "index": 0}
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return {"type": NodeType.OP_ADD}
            elif isinstance(node.op, ast.Sub):
                return {"type": NodeType.OP_SUB}
            elif isinstance(node.op, ast.Mult):
                return {"type": NodeType.OP_MUL}
            elif isinstance(node.op, ast.Div):
                return {"type": NodeType.OP_DIV}
            elif isinstance(node.op, ast.Mod):
                return {"type": NodeType.OP_MOD}
            elif isinstance(node.op, ast.Pow):
                return {"type": NodeType.OP_POW}
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return {"type": NodeType.OP_NEG}
        elif isinstance(node, ast.Compare):
            if len(node.ops) > 0:
                op = node.ops[0]
                if isinstance(op, ast.Eq):
                    return {"type": NodeType.OP_EQ}
                elif isinstance(op, ast.NotEq):
                    return {"type": NodeType.OP_NE}
                elif isinstance(op, ast.Lt):
                    return {"type": NodeType.OP_LT}
                elif isinstance(op, ast.LtE):
                    return {"type": NodeType.OP_LE}
                elif isinstance(op, ast.Gt):
                    return {"type": NodeType.OP_GT}
                elif isinstance(op, ast.GtE):
                    return {"type": NodeType.OP_GE}
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "sum":
                    return {"type": NodeType.OP_BUILTIN_SUM}
                elif node.func.id == "range":
                    return {"type": NodeType.OP_BUILTIN_RANGE}
                elif node.func.id == "len":
                    return {"type": NodeType.OP_BUILTIN_LEN}
                elif node.func.id == "min":
                    return {"type": NodeType.OP_BUILTIN_MIN}
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return {"type": NodeType.CONST_INT, "value": node.value}
            elif isinstance(node.value, bool):
                return {"type": NodeType.CONST_BOOL, "value": node.value}

        # Default fallback
        return {"type": NodeType.VAR_TEMP}

    def ast_to_simplified_json(self, code: str) -> Dict[str, Any]:
        """Convert Python code to human-readable JSON AST"""
        tree = self.python_to_ast(code)

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
                op_map = {
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
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op_map = {
                        ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                        ast.Gt: ">", ast.GtE: ">="
                    }
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
            elif isinstance(node, ast.IfExp):  # Ternary operator: a if condition else b
                return {
                    "ternary": {
                        "test": traverse_to_readable(node.test),
                        "body": traverse_to_readable(node.body),
                        "orelse": traverse_to_readable(node.orelse)
                    }
                }
            elif isinstance(node, ast.List):
                return {"list": [traverse_to_readable(elt) for elt in node.elts]}
            elif isinstance(node, ast.Tuple):
                return {"tuple": [traverse_to_readable(elt) for elt in node.elts]}
            else:
                # Fallback for unknown node types
                return {"unknown": str(type(node).__name__)}

        # Start traversal from the function definition
        result = traverse_to_readable(tree.body[0])

        return result

    def ast_to_graph_arrays(self, code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert Python code to graph representation arrays"""
        # Use the old method for generating graph arrays
        tree = self.python_to_ast(code)

        nodes = []
        edges = []
        node_counter = 0

        def traverse(node, parent_id=None):
            nonlocal node_counter
            current_id = node_counter
            node_counter += 1

            # Add simplified node
            simplified = self.simplify_ast_node(node)
            nodes.append(simplified)

            # Add edge from parent
            if parent_id is not None:
                edges.append((parent_id, current_id))

            # Traverse children
            for child in ast.iter_child_nodes(node):
                if node_counter < self.max_nodes:
                    traverse(child, current_id)

        # Start traversal
        traverse(tree.body[0])  # Assume single function definition

        # Build arrays
        num_nodes = min(len(nodes), self.max_nodes)

        # Node existence mask
        node_exists = np.zeros(self.max_nodes, dtype=bool)
        node_exists[:num_nodes] = True

        # Adjacency matrix
        adjacency = np.zeros((self.max_nodes, self.max_nodes), dtype=bool)
        for parent, child in edges:
            if parent < self.max_nodes and child < self.max_nodes:
                adjacency[parent, child] = True

        # Node types
        node_types = np.zeros(self.max_nodes, dtype=np.int32)
        for i, node in enumerate(nodes[:num_nodes]):
            node_types[i] = node["type"]

        # Node values (for constants)
        node_values = np.zeros(self.max_nodes, dtype=np.int32)
        for i, node in enumerate(nodes[:num_nodes]):
            if "value" in node:
                node_values[i] = node["value"]
            elif "params" in node:
                node_values[i] = node["params"]

        return node_exists, adjacency, node_types, node_values


def generate_program_examples(template_name: str, template: Dict[str, Any], num_examples: int) -> List[Dict[str, Any]]:
    """Generate additional examples for a program template"""
    examples = template["base_examples"].copy()

    # Generate more examples based on the pattern
    np.random.seed(42)  # Deterministic for reproducibility

    if template_name == "sum_up_to_n":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(1, 20)
            output = sum(range(1, n + 1))
            examples.append({"input": n, "output": output})

    elif template_name == "max_of_two":
        for _ in range(num_examples - len(examples)):
            a, b = np.random.randint(-10, 20, 2)
            output = max(a, b)
            examples.append({"input": [a, b], "output": output})

    elif template_name == "absolute_value":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-20, 20)
            output = abs(n)
            examples.append({"input": n, "output": output})

    elif template_name == "is_even":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-10, 20)
            output = n % 2 == 0
            examples.append({"input": n, "output": output})

    elif template_name == "factorial":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(0, 8)  # Keep small for factorial
            output = 1
            for i in range(1, n + 1):
                output *= i
            examples.append({"input": n, "output": output})

    elif template_name == "power_of_two":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(0, 10)  # Keep small for powers
            output = 2 ** n
            examples.append({"input": n, "output": output})

    elif template_name == "square":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-10, 11)
            output = n * n
            examples.append({"input": n, "output": output})

    elif template_name == "cube":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-5, 6)  # Keep smaller range for cubes
            output = n * n * n
            examples.append({"input": n, "output": output})

    elif template_name == "min_of_two":
        for _ in range(num_examples - len(examples)):
            a, b = np.random.randint(-10, 20, 2)
            output = min(a, b)
            examples.append({"input": [a, b], "output": output})

    elif template_name == "is_positive":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-10, 20)
            output = n > 0
            examples.append({"input": n, "output": output})

    elif template_name == "is_negative":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-10, 20)
            output = n < 0
            examples.append({"input": n, "output": output})

    elif template_name == "double":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-20, 21)
            output = n * 2
            examples.append({"input": n, "output": output})

    elif template_name == "half":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-20, 21)
            output = n // 2
            examples.append({"input": n, "output": output})

    elif template_name == "add_ten":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-20, 21)
            output = n + 10
            examples.append({"input": n, "output": output})

    elif template_name == "subtract_five":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-15, 26)
            output = n - 5
            examples.append({"input": n, "output": output})

    elif template_name == "is_zero":
        for _ in range(num_examples - len(examples)):
            # Include zero more frequently for this test
            if np.random.random() < 0.3:
                n = 0
            else:
                n = np.random.randint(-10, 11)
                if n == 0:
                    n = np.random.choice([-1, 1])  # Avoid zero
            output = n == 0
            examples.append({"input": n, "output": output})

    elif template_name == "sum_of_two":
        for _ in range(num_examples - len(examples)):
            a, b = np.random.randint(-10, 21, 2)
            output = a + b
            examples.append({"input": [a, b], "output": output})

    elif template_name == "difference":
        for _ in range(num_examples - len(examples)):
            a, b = np.random.randint(-10, 21, 2)
            output = a - b
            examples.append({"input": [a, b], "output": output})

    elif template_name == "product":
        for _ in range(num_examples - len(examples)):
            a, b = np.random.randint(-8, 9, 2)  # Keep products reasonable
            output = a * b
            examples.append({"input": [a, b], "output": output})

    elif template_name == "count_up_to_n":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(0, 21)
            output = n if n >= 0 else 0
            examples.append({"input": n, "output": output})

    elif template_name == "is_greater_than_ten":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-5, 26)
            output = n > 10
            examples.append({"input": n, "output": output})

    elif template_name == "remainder_by_three":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-10, 21)
            output = n % 3
            examples.append({"input": n, "output": output})

    elif template_name == "is_divisible_by_five":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-20, 21)
            output = n % 5 == 0
            examples.append({"input": n, "output": output})

    elif template_name == "triple":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-10, 11)
            output = n * 3
            examples.append({"input": n, "output": output})

    return examples[:num_examples]


def encode_program_specification(spec: Dict[str, Any]) -> np.ndarray:
    """Encode program specification as fixed-size vector"""
    spec_vector = np.zeros(512, dtype=np.float32)  # Fixed size

    # Encode basic info (simplified)
    spec_vector[0] = len(spec["inputs"])
    spec_vector[1] = len(spec["outputs"])
    spec_vector[2] = len(spec["examples"])

    # Encode examples (first 5 examples, simplified)
    for i, example in enumerate(spec["examples"][:5]):
        base_idx = 10 + i * 10
        if isinstance(example["input"], list):
            for j, inp in enumerate(example["input"][:5]):
                spec_vector[base_idx + j] = float(inp)
        else:
            spec_vector[base_idx] = float(example["input"])

        spec_vector[base_idx + 5] = float(example["output"])

    return spec_vector


def generate_program_files(program_id: int, _template_name: str, template: Dict[str, Any],
                          examples: List[Dict[str, Any]], ast_processor: ASTSimplifier,
                          output_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generate individual program files and return arrays"""


    # Create program specification
    spec = {
        "description": template["description"],
        "inputs": template["inputs"],
        "outputs": template["outputs"],
        "examples": examples
    }

    # Generate Python implementation
    python_code = template["implementation"]

    # Generate simplified AST
    simplified_ast = ast_processor.ast_to_simplified_json(python_code)

    # Generate arrays for model training
    node_exists, adjacency, node_types, node_values = ast_processor.ast_to_graph_arrays(python_code)

    # Create flattened AST tensor
    ast_tensor = np.concatenate([
        node_exists.astype(np.int32),
        adjacency.flatten().astype(np.int32),
        node_types,
        node_values
    ])

    # Encode specification
    spec_encoding = encode_program_specification(spec)

    # Write files
    base_filename = f"program{program_id:04d}"

    # Write YAML specification
    with open(os.path.join(output_dir, f"{base_filename}.yaml"), "w") as f:
        yaml.dump(spec, f, default_flow_style=False, indent=2)

    # Write simplified AST JSON
    with open(os.path.join(output_dir, f"{base_filename}.json"), "w") as f:
        json.dump(simplified_ast, f, indent=2)

    # Write Python program
    with open(os.path.join(output_dir, f"{base_filename}.py"), "w") as f:
        f.write(python_code)

    return spec_encoding, ast_tensor


def validate_ast_graph(node_exists: np.ndarray, adjacency: np.ndarray, node_types: np.ndarray) -> bool:
    """Basic validation of AST graph structure"""
    # Check that we have at least one node (function definition)
    if not np.any(node_exists):
        return False

    # Check that first node is FUNC_DEF
    if node_types[0] != NodeType.FUNC_DEF:
        return False

    # Check adjacency matrix is valid
    num_nodes = np.sum(node_exists)
    if np.any(adjacency[num_nodes:, :]) or np.any(adjacency[:, num_nodes:]):
        return False

    return True


def build_dataset(config: DataProcessConfig):
    """Main dataset building function"""
    np.random.seed(config.seed)

    print(f"Building {config.dataset_name} dataset with {config.num_samples} samples")

    # Initialize AST processor
    ast_processor = ASTSimplifier(max_nodes=config.max_nodes)

    # Create output directories
    train_dir = os.path.join(config.output_dir, "train")
    test_dir = os.path.join(config.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Prepare results storage for numpy arrays
    train_results = {"inputs": [], "labels": [], "program_identifiers": [], "program_indices": [], "group_indices": []}
    test_results = {"inputs": [], "labels": [], "program_identifiers": [], "program_indices": [], "group_indices": []}

    train_results["program_indices"].append(0)
    train_results["group_indices"].append(0)
    test_results["program_indices"].append(0)
    test_results["group_indices"].append(0)

    program_id = 0
    train_example_id = 0
    test_example_id = 0
    train_program_id = 0
    test_program_id = 0

    # Calculate program files per template (each file contains 3 examples)
    templates_per_split = len(PROGRAM_TEMPLATES)
    examples_per_template = config.examples_per_program
    examples_per_program_file = 3
    program_files_per_template = (examples_per_template + examples_per_program_file - 1) // examples_per_program_file
    total_program_files = templates_per_split * program_files_per_template
    train_size = int(0.8 * total_program_files)

    current_example_count = 0

    # Process each program template
    for template_idx, (template_name, template) in enumerate(PROGRAM_TEMPLATES.items()):
        print(f"Processing template: {template_name}")

        # Generate examples for this template
        all_examples = generate_program_examples(template_name, template, examples_per_template)

        # Process examples in groups of 3 as separate programs
        examples_per_program_file = 3
        num_program_files = (len(all_examples) + examples_per_program_file - 1) // examples_per_program_file

        for program_file_idx in range(num_program_files):
            start_idx = program_file_idx * examples_per_program_file
            end_idx = min(start_idx + examples_per_program_file, len(all_examples))
            examples_subset = all_examples[start_idx:end_idx]
            # Determine if this goes to train or test
            is_train = current_example_count < train_size
            current_example_count += 1

            output_dir = train_dir if is_train else test_dir
            results = train_results if is_train else test_results

            # Generate program files and arrays
            spec_encoding, ast_tensor = generate_program_files(
                program_id, template_name, template, examples_subset,
                ast_processor, output_dir
            )

            # Validate AST
            node_exists = ast_tensor[:config.max_nodes].astype(bool)
            adjacency = ast_tensor[config.max_nodes:config.max_nodes + config.max_nodes*config.max_nodes].reshape(config.max_nodes, config.max_nodes).astype(bool)
            node_types = ast_tensor[config.max_nodes + config.max_nodes*config.max_nodes:config.max_nodes + config.max_nodes*config.max_nodes + config.max_nodes].astype(int)

            if not validate_ast_graph(node_exists, adjacency, node_types):
                print(f"Warning: Invalid AST graph for program {program_id}")

            # Store arrays
            results["inputs"].append(spec_encoding)
            results["labels"].append(ast_tensor)

            if is_train:
                train_example_id += 1
                results["program_indices"].append(train_example_id)
                results["program_identifiers"].append(template_idx + 1)  # 1-indexed
                train_program_id += 1
            else:
                test_example_id += 1
                results["program_indices"].append(test_example_id)
                results["program_identifiers"].append(template_idx + 1)  # 1-indexed
                test_program_id += 1

            program_id += 1

        # Update group indices
        train_results["group_indices"].append(train_program_id)
        test_results["group_indices"].append(test_program_id)

    # Convert to numpy arrays and save
    for split_name, results, split_dir in [("train", train_results, train_dir), ("test", test_results, test_dir)]:
        if len(results["inputs"]) == 0:
            continue
        results_np = {}
        for key in ["inputs", "labels"]:
            results_np[key] = np.stack(results[key], axis=0)

        for key in ["program_identifiers", "program_indices", "group_indices"]:
            results_np[key] = np.array(results[key], dtype=np.int32)

        # Save numpy arrays
        for key, data in results_np.items():
            np.save(os.path.join(split_dir, f"all__{key}.npy"), data)

        # Create metadata
        metadata = PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,

            vocab_size=30,  # AST node types
            seq_len=512 + config.max_nodes * (1 + config.max_nodes + 2),  # spec + flattened AST
            num_puzzle_identifiers=len(PROGRAM_TEMPLATES) + 1,

            total_groups=len(PROGRAM_TEMPLATES),
            mean_puzzle_examples=examples_per_template,
            sets=["all"]
        )

        # Save metadata
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        print(f"{split_name.capitalize()} set:")
        print(f"  Total examples: {len(results_np['inputs'])}")
        print(f"  Input shape: {results_np['inputs'].shape}")
        print(f"  Label shape: {results_np['labels'].shape}")

    print(f"Dataset saved to: {config.output_dir}")
    print(f"Generated {program_id} individual program files")


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    """Main entry point"""
    print(f"Building program synthesis dataset: {config.dataset_name}")
    build_dataset(config)
    print("Dataset generation completed successfully!")


if __name__ == "__main__":
    cli()