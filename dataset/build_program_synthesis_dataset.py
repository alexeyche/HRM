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


# Type vocabulary for program specifications
TYPE_VOCAB = {
    # Basic types (4 primitives)
    'int': 1, 'float': 2, 'str': 3, 'bool': 4,

    # Container types (3 containers + generics)
    'List[int]': 5, 'List[float]': 6, 'List[str]': 7, 'List[bool]': 8,
    'Dict[str,int]': 9, 'Dict[str,str]': 10, 'Dict[int,str]': 11,
    'Set[int]': 12, 'Set[str]': 13, 'Set[float]': 14,

    # Arrays (common alias)
    'Array[int]': 5, 'Array[float]': 6, 'Array[str]': 7
}


cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_name: str = "simple_math"
    output_dir: str = "data/program-synthesis-simple-math"
    num_samples: int = 100
    seed: int = 42

    # AST processing config
    max_nodes: int = 30  # Reduced from 50 for efficiency
    max_edges: int = 25  # New: explicit edge limit
    examples_per_program: int = 20  # Increased for more data per template


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
    },

    # Mathematical Operations (20 templates)
    "gcd": {
        "description": "Calculate greatest common divisor of two numbers",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"}
        ],
        "outputs": [{"type": "int", "description": "Greatest common divisor"}],
        "base_examples": [
            {"input": [12, 8], "output": 4},
            {"input": [15, 10], "output": 5},
            {"input": [7, 3], "output": 1},
            {"input": [24, 18], "output": 6},
            {"input": [100, 25], "output": 25}
        ],
        "implementation": """def program(a, b):
    while b:
        a, b = b, a % b
    return a"""
    },

    "lcm": {
        "description": "Calculate least common multiple of two numbers",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"}
        ],
        "outputs": [{"type": "int", "description": "Least common multiple"}],
        "base_examples": [
            {"input": [4, 6], "output": 12},
            {"input": [3, 5], "output": 15},
            {"input": [8, 12], "output": 24},
            {"input": [7, 14], "output": 14},
            {"input": [15, 25], "output": 75}
        ],
        "implementation": """def program(a, b):
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return (a * b) // gcd(a, b)"""
    },

    "fibonacci": {
        "description": "Generate nth Fibonacci number",
        "inputs": [{"type": "int", "description": "Position in Fibonacci sequence"}],
        "outputs": [{"type": "int", "description": "Fibonacci number"}],
        "base_examples": [
            {"input": 0, "output": 0},
            {"input": 1, "output": 1},
            {"input": 5, "output": 5},
            {"input": 8, "output": 21},
            {"input": 10, "output": 55}
        ],
        "implementation": """def program(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""
    },

    "prime_check": {
        "description": "Check if a number is prime",
        "inputs": [{"type": "int", "description": "Number to check"}],
        "outputs": [{"type": "bool", "description": "True if prime, False otherwise"}],
        "base_examples": [
            {"input": 2, "output": True},
            {"input": 17, "output": True},
            {"input": 4, "output": False},
            {"input": 9, "output": False},
            {"input": 13, "output": True}
        ],
        "implementation": """def program(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True"""
    },

    "digit_sum": {
        "description": "Calculate sum of digits in a number",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "int", "description": "Sum of digits"}],
        "base_examples": [
            {"input": 123, "output": 6},
            {"input": 456, "output": 15},
            {"input": 789, "output": 24},
            {"input": 1000, "output": 1},
            {"input": 99, "output": 18}
        ],
        "implementation": """def program(n):
    return sum(int(digit) for digit in str(abs(n)))"""
    },

    "reverse_number": {
        "description": "Reverse the digits of a number",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "int", "description": "Number with reversed digits"}],
        "base_examples": [
            {"input": 123, "output": 321},
            {"input": 456, "output": 654},
            {"input": 1000, "output": 1},
            {"input": 5040, "output": 405},
            {"input": 12, "output": 21}
        ],
        "implementation": """def program(n):
    return int(str(abs(n))[::-1]) * (1 if n >= 0 else -1)"""
    },

    "perfect_square": {
        "description": "Check if a number is a perfect square",
        "inputs": [{"type": "int", "description": "Number to check"}],
        "outputs": [{"type": "bool", "description": "True if perfect square"}],
        "base_examples": [
            {"input": 16, "output": True},
            {"input": 25, "output": True},
            {"input": 15, "output": False},
            {"input": 36, "output": True},
            {"input": 50, "output": False}
        ],
        "implementation": """def program(n):
    if n < 0:
        return False
    root = int(n ** 0.5)
    return root * root == n"""
    },

    "factorial_mod": {
        "description": "Calculate factorial modulo a number",
        "inputs": [
            {"type": "int", "description": "Number for factorial"},
            {"type": "int", "description": "Modulo value"}
        ],
        "outputs": [{"type": "int", "description": "Factorial mod p"}],
        "base_examples": [
            {"input": [5, 7], "output": 1},
            {"input": [4, 10], "output": 4},
            {"input": [6, 13], "output": 5},
            {"input": [3, 5], "output": 1},
            {"input": [7, 11], "output": 2}
        ],
        "implementation": """def program(n, p):
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % p
    return result"""
    },

    "power_mod": {
        "description": "Calculate (base^exp) mod m efficiently",
        "inputs": [
            {"type": "int", "description": "Base"},
            {"type": "int", "description": "Exponent"},
            {"type": "int", "description": "Modulo"}
        ],
        "outputs": [{"type": "int", "description": "Result of (base^exp) mod m"}],
        "base_examples": [
            {"input": [2, 3, 5], "output": 3},
            {"input": [3, 4, 7], "output": 4},
            {"input": [5, 2, 11], "output": 3},
            {"input": [2, 10, 1000], "output": 24},
            {"input": [7, 3, 13], "output": 5}
        ],
        "implementation": """def program(base, exp, m):
    result = 1
    base = base % m
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % m
        exp = exp >> 1
        base = (base * base) % m
    return result"""
    },

    "divisor_count": {
        "description": "Count the number of divisors of a number",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "int", "description": "Number of divisors"}],
        "base_examples": [
            {"input": 12, "output": 6},
            {"input": 16, "output": 5},
            {"input": 7, "output": 2},
            {"input": 24, "output": 8},
            {"input": 1, "output": 1}
        ],
        "implementation": """def program(n):
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
    return count"""
    },

    "collatz_steps": {
        "description": "Count steps in Collatz sequence to reach 1",
        "inputs": [{"type": "int", "description": "Starting number"}],
        "outputs": [{"type": "int", "description": "Number of steps"}],
        "base_examples": [
            {"input": 1, "output": 0},
            {"input": 2, "output": 1},
            {"input": 3, "output": 7},
            {"input": 4, "output": 2},
            {"input": 5, "output": 5}
        ],
        "implementation": """def program(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps"""
    },

    "sum_of_squares": {
        "description": "Calculate sum of squares from 1 to n",
        "inputs": [{"type": "int", "description": "Upper limit"}],
        "outputs": [{"type": "int", "description": "Sum of squares"}],
        "base_examples": [
            {"input": 3, "output": 14},
            {"input": 4, "output": 30},
            {"input": 5, "output": 55},
            {"input": 1, "output": 1},
            {"input": 2, "output": 5}
        ],
        "implementation": """def program(n):
    return sum(i * i for i in range(1, n + 1))"""
    },

    "binomial_coeff": {
        "description": "Calculate binomial coefficient C(n,k)",
        "inputs": [
            {"type": "int", "description": "n value"},
            {"type": "int", "description": "k value"}
        ],
        "outputs": [{"type": "int", "description": "Binomial coefficient"}],
        "base_examples": [
            {"input": [5, 2], "output": 10},
            {"input": [4, 1], "output": 4},
            {"input": [6, 3], "output": 20},
            {"input": [3, 0], "output": 1},
            {"input": [7, 4], "output": 35}
        ],
        "implementation": """def program(n, k):
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result"""
    },

    "next_prime": {
        "description": "Find the next prime number after n",
        "inputs": [{"type": "int", "description": "Starting number"}],
        "outputs": [{"type": "int", "description": "Next prime number"}],
        "base_examples": [
            {"input": 2, "output": 3},
            {"input": 7, "output": 11},
            {"input": 13, "output": 17},
            {"input": 20, "output": 23},
            {"input": 30, "output": 31}
        ],
        "implementation": """def program(n):
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x ** 0.5) + 1):
            if x % i == 0:
                return False
        return True

    candidate = n + 1
    while not is_prime(candidate):
        candidate += 1
    return candidate"""
    },

    "sum_divisors": {
        "description": "Calculate sum of proper divisors of a number",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "int", "description": "Sum of proper divisors"}],
        "base_examples": [
            {"input": 12, "output": 16},
            {"input": 6, "output": 6},
            {"input": 28, "output": 28},
            {"input": 8, "output": 7},
            {"input": 1, "output": 0}
        ],
        "implementation": """def program(n):
    return sum(i for i in range(1, n) if n % i == 0)"""
    },

    "digit_product": {
        "description": "Calculate product of digits in a number",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "int", "description": "Product of digits"}],
        "base_examples": [
            {"input": 123, "output": 6},
            {"input": 456, "output": 120},
            {"input": 1000, "output": 0},
            {"input": 789, "output": 504},
            {"input": 25, "output": 10}
        ],
        "implementation": """def program(n):
    product = 1
    for digit in str(abs(n)):
        product *= int(digit)
    return product"""
    },

    "is_perfect": {
        "description": "Check if a number is perfect (equals sum of proper divisors)",
        "inputs": [{"type": "int", "description": "Number to check"}],
        "outputs": [{"type": "bool", "description": "True if perfect number"}],
        "base_examples": [
            {"input": 6, "output": True},
            {"input": 28, "output": True},
            {"input": 12, "output": False},
            {"input": 8, "output": False},
            {"input": 496, "output": True}
        ],
        "implementation": """def program(n):
    divisor_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisor_sum == n"""
    },

    "greatest_digit": {
        "description": "Find the greatest digit in a number",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "int", "description": "Greatest digit"}],
        "base_examples": [
            {"input": 1234, "output": 4},
            {"input": 9876, "output": 9},
            {"input": 5000, "output": 5},
            {"input": 111, "output": 1},
            {"input": 987, "output": 9}
        ],
        "implementation": """def program(n):
    return max(int(digit) for digit in str(abs(n)))"""
    },

    # Array/List Operations (15 templates)
    "array_sum": {
        "description": "Sum all elements in an array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Sum of all elements"}],
        "base_examples": [
            {"input": [[1, 2, 3, 4]], "output": 10},
            {"input": [[5, -2, 8]], "output": 11},
            {"input": [[-1, -2, -3]], "output": -6},
            {"input": [[0, 0, 0]], "output": 0},
            {"input": [[42]], "output": 42}
        ],
        "implementation": """def program(arr):
    return sum(arr)"""
    },

    "array_max": {
        "description": "Find maximum element in array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Maximum element"}],
        "base_examples": [
            {"input": [[1, 5, 3, 9, 2]], "output": 9},
            {"input": [[-1, -5, -2]], "output": -1},
            {"input": [[100]], "output": 100},
            {"input": [[0, 0, 1]], "output": 1},
            {"input": [[7, 3, 7, 1]], "output": 7}
        ],
        "implementation": """def program(arr):
    return max(arr)"""
    },

    "array_min": {
        "description": "Find minimum element in array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Minimum element"}],
        "base_examples": [
            {"input": [[1, 5, 3, 9, 2]], "output": 1},
            {"input": [[-1, -5, -2]], "output": -5},
            {"input": [[100]], "output": 100},
            {"input": [[0, 0, -1]], "output": -1},
            {"input": [[7, 3, 7, 1]], "output": 1}
        ],
        "implementation": """def program(arr):
    return min(arr)"""
    },

    "array_length": {
        "description": "Get length of array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Length of array"}],
        "base_examples": [
            {"input": [[1, 2, 3]], "output": 3},
            {"input": [[]], "output": 0},
            {"input": [[42]], "output": 1},
            {"input": [[1, 2, 3, 4, 5]], "output": 5},
            {"input": [[-1, 0, 1, 2]], "output": 4}
        ],
        "implementation": """def program(arr):
    return len(arr)"""
    },

    "array_reverse": {
        "description": "Reverse array elements",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "List[int]", "description": "Reversed array"}],
        "base_examples": [
            {"input": [[1, 2, 3]], "output": [3, 2, 1]},
            {"input": [[5, 4]], "output": [4, 5]},
            {"input": [[42]], "output": [42]},
            {"input": [[]], "output": []},
            {"input": [[1, 2, 3, 4]], "output": [4, 3, 2, 1]}
        ],
        "implementation": """def program(arr):
    return arr[::-1]"""
    },

    "array_contains": {
        "description": "Check if array contains a specific value",
        "inputs": [
            {"type": "List[int]", "description": "Input array"},
            {"type": "int", "description": "Value to search for"}
        ],
        "outputs": [{"type": "bool", "description": "True if value found"}],
        "base_examples": [
            {"input": [[1, 2, 3], 2], "output": True},
            {"input": [[1, 2, 3], 5], "output": False},
            {"input": [[], 1], "output": False},
            {"input": [[0, -1, 2], -1], "output": True},
            {"input": [[5, 5, 5], 5], "output": True}
        ],
        "implementation": """def program(arr, value):
    return value in arr"""
    },

    "array_first": {
        "description": "Get first element of array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "First element"}],
        "base_examples": [
            {"input": [[1, 2, 3]], "output": 1},
            {"input": [[42]], "output": 42},
            {"input": [[-5, 0, 10]], "output": -5},
            {"input": [[100, 200]], "output": 100},
            {"input": [[7, 7, 7]], "output": 7}
        ],
        "implementation": """def program(arr):
    return arr[0] if arr else 0"""
    },

    "array_last": {
        "description": "Get last element of array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Last element"}],
        "base_examples": [
            {"input": [[1, 2, 3]], "output": 3},
            {"input": [[42]], "output": 42},
            {"input": [[-5, 0, 10]], "output": 10},
            {"input": [[100, 200]], "output": 200},
            {"input": [[7, 7, 7]], "output": 7}
        ],
        "implementation": """def program(arr):
    return arr[-1] if arr else 0"""
    },

    "array_product": {
        "description": "Calculate product of all elements in array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Product of all elements"}],
        "base_examples": [
            {"input": [[2, 3, 4]], "output": 24},
            {"input": [[1, 5, 2]], "output": 10},
            {"input": [[0, 5, 3]], "output": 0},
            {"input": [[-2, 3]], "output": -6},
            {"input": [[1]], "output": 1}
        ],
        "implementation": """def program(arr):
    result = 1
    for x in arr:
        result *= x
    return result"""
    },

    "array_count_positive": {
        "description": "Count positive numbers in array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Count of positive numbers"}],
        "base_examples": [
            {"input": [[1, -2, 3, -4, 5]], "output": 3},
            {"input": [[-1, -2, -3]], "output": 0},
            {"input": [[1, 2, 3]], "output": 3},
            {"input": [[0, 0, 0]], "output": 0},
            {"input": [[5]], "output": 1}
        ],
        "implementation": """def program(arr):
    return sum(1 for x in arr if x > 0)"""
    },

    "array_count_even": {
        "description": "Count even numbers in array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Count of even numbers"}],
        "base_examples": [
            {"input": [[1, 2, 3, 4, 5]], "output": 2},
            {"input": [[2, 4, 6]], "output": 3},
            {"input": [[1, 3, 5]], "output": 0},
            {"input": [[0, 2, 4]], "output": 3},
            {"input": [[-2, -1, 0]], "output": 2}
        ],
        "implementation": """def program(arr):
    return sum(1 for x in arr if x % 2 == 0)"""
    },

    "array_second_largest": {
        "description": "Find second largest element in array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Second largest element"}],
        "base_examples": [
            {"input": [[1, 5, 3, 9, 2]], "output": 5},
            {"input": [[10, 20, 30]], "output": 20},
            {"input": [[5, 5, 4]], "output": 4},
            {"input": [[1, 2]], "output": 1},
            {"input": [[7, 1, 9, 3]], "output": 7}
        ],
        "implementation": """def program(arr):
    unique_sorted = sorted(set(arr), reverse=True)
    return unique_sorted[1] if len(unique_sorted) > 1 else unique_sorted[0]"""
    },

    "array_range": {
        "description": "Calculate range (max - min) of array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Range of values"}],
        "base_examples": [
            {"input": [[1, 5, 3, 9, 2]], "output": 8},
            {"input": [[10, 10, 10]], "output": 0},
            {"input": [[-5, 5]], "output": 10},
            {"input": [[100]], "output": 0},
            {"input": [[0, 1, 2, 3]], "output": 3}
        ],
        "implementation": """def program(arr):
    return max(arr) - min(arr) if arr else 0"""
    },

    "array_median": {
        "description": "Find median value of array (middle element when sorted)",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Median value"}],
        "base_examples": [
            {"input": [[1, 3, 5]], "output": 3},
            {"input": [[1, 2, 3, 4]], "output": 2},
            {"input": [[5, 1, 9, 3, 7]], "output": 5},
            {"input": [[10]], "output": 10},
            {"input": [[2, 1]], "output": 1}
        ],
        "implementation": """def program(arr):
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return sorted_arr[n // 2 - 1]"""
    },

    "array_unique_count": {
        "description": "Count unique elements in array",
        "inputs": [{"type": "List[int]", "description": "Input array"}],
        "outputs": [{"type": "int", "description": "Number of unique elements"}],
        "base_examples": [
            {"input": [[1, 2, 3, 2, 1]], "output": 3},
            {"input": [[5, 5, 5]], "output": 1},
            {"input": [[1, 2, 3, 4]], "output": 4},
            {"input": [[]], "output": 0},
            {"input": [[0, 0, 1, 1, 2]], "output": 3}
        ],
        "implementation": """def program(arr):
    return len(set(arr))"""
    },

    # String Operations (10 templates)
    "string_length": {
        "description": "Get length of string",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "int", "description": "Length of string"}],
        "base_examples": [
            {"input": "hello", "output": 5},
            {"input": "world", "output": 5},
            {"input": "", "output": 0},
            {"input": "a", "output": 1},
            {"input": "python", "output": 6}
        ],
        "implementation": """def program(s):
    return len(s)"""
    },

    "string_reverse": {
        "description": "Reverse a string",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "str", "description": "Reversed string"}],
        "base_examples": [
            {"input": "hello", "output": "olleh"},
            {"input": "world", "output": "dlrow"},
            {"input": "a", "output": "a"},
            {"input": "", "output": ""},
            {"input": "abc", "output": "cba"}
        ],
        "implementation": """def program(s):
    return s[::-1]"""
    },

    "string_uppercase": {
        "description": "Convert string to uppercase",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "str", "description": "Uppercase string"}],
        "base_examples": [
            {"input": "hello", "output": "HELLO"},
            {"input": "World", "output": "WORLD"},
            {"input": "ABC", "output": "ABC"},
            {"input": "", "output": ""},
            {"input": "python", "output": "PYTHON"}
        ],
        "implementation": """def program(s):
    return s.upper()"""
    },

    "string_lowercase": {
        "description": "Convert string to lowercase",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "str", "description": "Lowercase string"}],
        "base_examples": [
            {"input": "HELLO", "output": "hello"},
            {"input": "World", "output": "world"},
            {"input": "abc", "output": "abc"},
            {"input": "", "output": ""},
            {"input": "PYTHON", "output": "python"}
        ],
        "implementation": """def program(s):
    return s.lower()"""
    },

    "string_count_vowels": {
        "description": "Count vowel characters in string",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "int", "description": "Number of vowels"}],
        "base_examples": [
            {"input": "hello", "output": 2},
            {"input": "python", "output": 1},
            {"input": "aeiou", "output": 5},
            {"input": "bcdfg", "output": 0},
            {"input": "", "output": 0}
        ],
        "implementation": """def program(s):
    return sum(1 for c in s.lower() if c in 'aeiou')"""
    },

    "string_palindrome": {
        "description": "Check if string is a palindrome",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "bool", "description": "True if palindrome"}],
        "base_examples": [
            {"input": "racecar", "output": True},
            {"input": "hello", "output": False},
            {"input": "a", "output": True},
            {"input": "", "output": True},
            {"input": "aba", "output": True}
        ],
        "implementation": """def program(s):
    return s == s[::-1]"""
    },

    "string_concat": {
        "description": "Concatenate two strings",
        "inputs": [
            {"type": "str", "description": "First string"},
            {"type": "str", "description": "Second string"}
        ],
        "outputs": [{"type": "str", "description": "Concatenated string"}],
        "base_examples": [
            {"input": ["hello", "world"], "output": "helloworld"},
            {"input": ["py", "thon"], "output": "python"},
            {"input": ["", "test"], "output": "test"},
            {"input": ["a", ""], "output": "a"},
            {"input": ["foo", "bar"], "output": "foobar"}
        ],
        "implementation": """def program(s1, s2):
    return s1 + s2"""
    },

    "string_first_char": {
        "description": "Get first character of string",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "str", "description": "First character"}],
        "base_examples": [
            {"input": "hello", "output": "h"},
            {"input": "world", "output": "w"},
            {"input": "a", "output": "a"},
            {"input": "python", "output": "p"},
            {"input": "test", "output": "t"}
        ],
        "implementation": """def program(s):
    return s[0] if s else ''"""
    },

    "string_last_char": {
        "description": "Get last character of string",
        "inputs": [{"type": "str", "description": "Input string"}],
        "outputs": [{"type": "str", "description": "Last character"}],
        "base_examples": [
            {"input": "hello", "output": "o"},
            {"input": "world", "output": "d"},
            {"input": "a", "output": "a"},
            {"input": "python", "output": "n"},
            {"input": "test", "output": "t"}
        ],
        "implementation": """def program(s):
    return s[-1] if s else ''"""
    },

    "string_contains": {
        "description": "Check if string contains substring",
        "inputs": [
            {"type": "str", "description": "Main string"},
            {"type": "str", "description": "Substring to search"}
        ],
        "outputs": [{"type": "bool", "description": "True if substring found"}],
        "base_examples": [
            {"input": ["hello", "ell"], "output": True},
            {"input": ["world", "or"], "output": True},
            {"input": ["python", "java"], "output": False},
            {"input": ["test", ""], "output": True},
            {"input": ["", "a"], "output": False}
        ],
        "implementation": """def program(s, sub):
    return sub in s"""
    },

    # Conditional Logic (10 templates)
    "max_of_three": {
        "description": "Find maximum of three numbers",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"},
            {"type": "int", "description": "Third number"}
        ],
        "outputs": [{"type": "int", "description": "Maximum of three"}],
        "base_examples": [
            {"input": [1, 5, 3], "output": 5},
            {"input": [10, 2, 15], "output": 15},
            {"input": [-1, -5, -2], "output": -1},
            {"input": [7, 7, 7], "output": 7},
            {"input": [0, 1, -1], "output": 1}
        ],
        "implementation": """def program(a, b, c):
    return max(a, b, c)"""
    },

    "min_of_three": {
        "description": "Find minimum of three numbers",
        "inputs": [
            {"type": "int", "description": "First number"},
            {"type": "int", "description": "Second number"},
            {"type": "int", "description": "Third number"}
        ],
        "outputs": [{"type": "int", "description": "Minimum of three"}],
        "base_examples": [
            {"input": [1, 5, 3], "output": 1},
            {"input": [10, 2, 15], "output": 2},
            {"input": [-1, -5, -2], "output": -5},
            {"input": [7, 7, 7], "output": 7},
            {"input": [0, 1, -1], "output": -1}
        ],
        "implementation": """def program(a, b, c):
    return min(a, b, c)"""
    },

    "grade_classifier": {
        "description": "Classify grade (A: 90+, B: 80+, C: 70+, D: 60+, F: <60)",
        "inputs": [{"type": "int", "description": "Grade score"}],
        "outputs": [{"type": "str", "description": "Letter grade"}],
        "base_examples": [
            {"input": 95, "output": "A"},
            {"input": 85, "output": "B"},
            {"input": 75, "output": "C"},
            {"input": 65, "output": "D"},
            {"input": 55, "output": "F"}
        ],
        "implementation": """def program(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'"""
    },

    "sign_function": {
        "description": "Return sign of number (-1, 0, or 1)",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "int", "description": "Sign (-1, 0, 1)"}],
        "base_examples": [
            {"input": 5, "output": 1},
            {"input": -3, "output": -1},
            {"input": 0, "output": 0},
            {"input": 100, "output": 1},
            {"input": -50, "output": -1}
        ],
        "implementation": """def program(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0"""
    },

    "triangle_type": {
        "description": "Classify triangle type (equilateral, isosceles, scalene)",
        "inputs": [
            {"type": "int", "description": "Side a"},
            {"type": "int", "description": "Side b"},
            {"type": "int", "description": "Side c"}
        ],
        "outputs": [{"type": "str", "description": "Triangle type"}],
        "base_examples": [
            {"input": [3, 3, 3], "output": "e"},
            {"input": [3, 3, 4], "output": "i"},
            {"input": [3, 4, 5], "output": "s"},
            {"input": [5, 5, 8], "output": "i"},
            {"input": [2, 3, 4], "output": "s"}
        ],
        "implementation": """def program(a, b, c):
    if a == b == c:
        return 'e'
    elif a == b or b == c or a == c:
        return 'i'
    else:
        return 's'"""
    },

    "leap_year": {
        "description": "Check if year is a leap year",
        "inputs": [{"type": "int", "description": "Year"}],
        "outputs": [{"type": "bool", "description": "True if leap year"}],
        "base_examples": [
            {"input": 2020, "output": True},
            {"input": 2021, "output": False},
            {"input": 2000, "output": True},
            {"input": 1900, "output": False},
            {"input": 2024, "output": True}
        ],
        "implementation": """def program(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"""
    },

    "quadrant": {
        "description": "Determine quadrant of point (x, y)",
        "inputs": [
            {"type": "int", "description": "X coordinate"},
            {"type": "int", "description": "Y coordinate"}
        ],
        "outputs": [{"type": "int", "description": "Quadrant number (1-4)"}],
        "base_examples": [
            {"input": [1, 1], "output": 1},
            {"input": [-1, 1], "output": 2},
            {"input": [-1, -1], "output": 3},
            {"input": [1, -1], "output": 4},
            {"input": [5, 3], "output": 1}
        ],
        "implementation": """def program(x, y):
    if x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:
        return 4"""
    },

    "number_category": {
        "description": "Categorize number (negative, zero, single, double, triple digit)",
        "inputs": [{"type": "int", "description": "Input number"}],
        "outputs": [{"type": "str", "description": "Category"}],
        "base_examples": [
            {"input": -5, "output": "negative"},
            {"input": 0, "output": "zero"},
            {"input": 7, "output": "single"},
            {"input": 42, "output": "double"},
            {"input": 123, "output": "triple"}
        ],
        "implementation": """def program(n):
            if n < 0:
                return 'negative'
            elif n == 0:
                return 'zero'
            elif n < 10:
                return 'single'
            elif n < 100:
                return 'double'
            else:
                return 'triple'"""
    },

    "compare_strings": {
        "description": "Compare two strings lexicographically (-1, 0, 1)",
        "inputs": [
            {"type": "str", "description": "First string"},
            {"type": "str", "description": "Second string"}
        ],
        "outputs": [{"type": "int", "description": "Comparison result"}],
        "base_examples": [
            {"input": ["apple", "banana"], "output": -1},
            {"input": ["hello", "hello"], "output": 0},
            {"input": ["zebra", "apple"], "output": 1},
            {"input": ["a", "b"], "output": -1},
            {"input": ["test", "test"], "output": 0}
        ],
        "implementation": """def program(s1, s2):
            if s1 < s2:
                return -1
            elif s1 > s2:
                return 1
            else:
                return 0"""
    },

    "season_from_month": {
        "description": "Determine season from month number (1-12)",
        "inputs": [{"type": "int", "description": "Month number (1-12)"}],
        "outputs": [{"type": "str", "description": "Season name"}],
        "base_examples": [
            {"input": 1, "output": "winter"},
            {"input": 4, "output": "spring"},
            {"input": 7, "output": "summer"},
            {"input": 10, "output": "fall"},
            {"input": 12, "output": "winter"}
        ],
        "implementation": """def program(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'"""
    }
}


class ASTSimplifier:
    """Converts Python AST to simplified graph representation"""

    def __init__(self, max_nodes: int = 30, max_edges: int = 25):
        self.max_nodes = max_nodes
        self.max_edges = max_edges

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

    def ast_to_sparse_representation(self, code: str) -> Dict[str, Any]:
        """Convert Python code to efficient sparse representation"""
        tree = self.python_to_ast(code)

        nodes = []
        edges = []
        node_counter = 0

        def traverse(node, parent_id=None):
            nonlocal node_counter
            current_id = node_counter
            node_counter += 1

            # Store node info compactly
            simplified = self.simplify_ast_node(node)
            nodes.append({
                'type': simplified['type'],
                'value': simplified.get('value', 0),
                'params': simplified.get('params', 0)
            })

            # Store edge (parent -> child)
            if parent_id is not None:
                edges.append((parent_id, current_id))

            # Traverse children
            for child in ast.iter_child_nodes(node):
                if node_counter < self.max_nodes:
                    traverse(child, current_id)

        traverse(tree.body[0])  # Start from function definition

        # Convert to efficient arrays
        num_nodes = len(nodes)

        return {
            'num_nodes': num_nodes,
            'node_types': np.array([n['type'] for n in nodes], dtype=np.int32),
            'node_values': np.array([n['value'] for n in nodes], dtype=np.int32),
            'node_params': np.array([n['params'] for n in nodes], dtype=np.int32),
            'edge_list': np.array(edges, dtype=np.int32) if edges else np.array([], dtype=np.int32).reshape(0, 2),
        }

    def flatten_sparse_ast(self, sparse_ast: Dict[str, Any]) -> np.ndarray:
        """Flatten sparse AST to fixed-size tensor for training"""
        components = []

        # Number of actual nodes/edges
        num_nodes = sparse_ast['num_nodes']
        num_edges = len(sparse_ast['edge_list'])
        components.extend([num_nodes, num_edges])  # 2 dims

        # Node features (padded to max_nodes)
        for key in ['node_types', 'node_values', 'node_params']:
            arr = sparse_ast[key]
            pad_width = max(0, self.max_nodes - len(arr))
            padded = np.pad(arr, (0, pad_width))[:self.max_nodes]
            components.extend(padded.tolist())  # 3 × max_nodes = 90 dims

        # Edge list (padded to max_edges)
        edges_flat = sparse_ast['edge_list'].flatten()
        pad_width = max(0, 2*self.max_edges - len(edges_flat))
        edges_padded = np.pad(edges_flat, (0, pad_width))[:2*self.max_edges]
        components.extend(edges_padded.tolist())  # 2 × max_edges = 50 dims

        return np.array(components, dtype=np.int32)
        # Total: 2 + 90 + 50 = 142 dims (vs 2650)


def convert_numpy_to_python(obj):
    """Recursively convert numpy types to Python types to avoid binary serialization in YAML"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


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

    # Mathematical Operations Augmentation
    elif template_name == "gcd":
        for _ in range(num_examples - len(examples)):
            a, b = np.random.randint(1, 50, 2)
            def gcd(x, y):
                while y:
                    x, y = y, x % y
                return x
            output = gcd(a, b)
            examples.append({"input": [a, b], "output": output})

    elif template_name == "lcm":
        for _ in range(num_examples - len(examples)):
            a, b = np.random.randint(1, 20, 2)
            def gcd(x, y):
                while y:
                    x, y = y, x % y
                return x
            output = (a * b) // gcd(a, b)
            examples.append({"input": [a, b], "output": output})

    elif template_name == "fibonacci":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(0, 15)
            if n <= 1:
                output = n
            else:
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                output = b
            examples.append({"input": n, "output": output})

    elif template_name == "prime_check":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(1, 50)
            def is_prime(x):
                if x < 2:
                    return False
                for i in range(2, int(x ** 0.5) + 1):
                    if x % i == 0:
                        return False
                return True
            output = is_prime(n)
            examples.append({"input": n, "output": output})

    elif template_name == "digit_sum":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(1, 10000)
            output = sum(int(digit) for digit in str(n))
            examples.append({"input": n, "output": output})

    elif template_name == "reverse_number":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(10, 10000)
            output = int(str(n)[::-1])
            examples.append({"input": n, "output": output})

    elif template_name == "perfect_square":
        for _ in range(num_examples - len(examples)):
            if np.random.random() < 0.3:  # 30% perfect squares
                root = np.random.randint(1, 10)
                n = root * root
                output = True
            else:
                n = np.random.randint(1, 100)
                root = int(n ** 0.5)
                if root * root == n:
                    continue  # Skip actual perfect squares
                output = False
            examples.append({"input": n, "output": output})

    elif template_name == "collatz_steps":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(1, 30)
            steps = 0
            temp_n = n
            while temp_n != 1:
                if temp_n % 2 == 0:
                    temp_n = temp_n // 2
                else:
                    temp_n = 3 * temp_n + 1
                steps += 1
                if steps > 500:  # Safety limit (increased for numbers like 27)
                    break
            examples.append({"input": n, "output": steps})

    # Array/List Operations Augmentation
    elif template_name == "array_sum":
        for _ in range(num_examples - len(examples)):
            arr_len = np.random.randint(1, 8)
            arr = np.random.randint(-10, 20, arr_len).tolist()
            output = sum(arr)
            examples.append({"input": [arr], "output": output})

    elif template_name == "array_max":
        for _ in range(num_examples - len(examples)):
            arr_len = np.random.randint(1, 8)
            arr = np.random.randint(-10, 20, arr_len).tolist()
            output = max(arr)
            examples.append({"input": [arr], "output": output})

    elif template_name == "array_min":
        for _ in range(num_examples - len(examples)):
            arr_len = np.random.randint(1, 8)
            arr = np.random.randint(-10, 20, arr_len).tolist()
            output = min(arr)
            examples.append({"input": [arr], "output": output})

    elif template_name == "array_length":
        for _ in range(num_examples - len(examples)):
            arr_len = np.random.randint(0, 10)
            arr = np.random.randint(-5, 10, arr_len).tolist()
            output = len(arr)
            examples.append({"input": [arr], "output": output})

    elif template_name == "array_reverse":
        for _ in range(num_examples - len(examples)):
            arr_len = np.random.randint(1, 6)
            arr = np.random.randint(-5, 10, arr_len).tolist()
            output = arr[::-1]
            examples.append({"input": [arr], "output": output})

    elif template_name == "array_contains":
        for _ in range(num_examples - len(examples)):
            arr_len = np.random.randint(2, 8)
            arr = np.random.randint(-5, 10, arr_len).tolist()
            if np.random.random() < 0.5:
                # 50% chance value is in array
                value = arr[np.random.randint(0, len(arr))]
                output = True
            else:
                # 50% chance value is not in array
                value = np.random.randint(50, 100)  # Unlikely to be in small array
                output = value in arr
            examples.append({"input": [arr, value], "output": output})

    elif template_name == "array_count_positive":
        for _ in range(num_examples - len(examples)):
            arr_len = np.random.randint(1, 8)
            arr = np.random.randint(-10, 10, arr_len).tolist()
            output = sum(1 for x in arr if x > 0)
            examples.append({"input": [arr], "output": output})

    # String Operations Augmentation
    elif template_name == "string_length":
        for _ in range(num_examples - len(examples)):
            length = np.random.randint(0, 15)
            s = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
            output = len(s)
            examples.append({"input": s, "output": output})

    elif template_name == "string_reverse":
        for _ in range(num_examples - len(examples)):
            length = np.random.randint(1, 10)
            s = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
            output = s[::-1]
            examples.append({"input": s, "output": output})

    elif template_name == "string_count_vowels":
        for _ in range(num_examples - len(examples)):
            length = np.random.randint(1, 12)
            s = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
            output = sum(1 for c in s.lower() if c in 'aeiou')
            examples.append({"input": s, "output": output})

    elif template_name == "string_palindrome":
        for _ in range(num_examples - len(examples)):
            if np.random.random() < 0.3:  # 30% palindromes
                half = ''.join(np.random.choice(list('abcd'), np.random.randint(1, 4)))
                s = half + half[::-1]
                output = True
            else:
                length = np.random.randint(2, 8)
                s = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
                output = s == s[::-1]
            examples.append({"input": s, "output": output})

    elif template_name == "string_concat":
        for _ in range(num_examples - len(examples)):
            len1 = np.random.randint(1, 6)
            len2 = np.random.randint(1, 6)
            s1 = ''.join(np.random.choice(list('abcdefghij'), len1))
            s2 = ''.join(np.random.choice(list('klmnopqrst'), len2))
            output = s1 + s2
            examples.append({"input": [s1, s2], "output": output})

    # Conditional Logic Augmentation
    elif template_name == "max_of_three":
        for _ in range(num_examples - len(examples)):
            a, b, c = np.random.randint(-20, 20, 3)
            output = max(a, b, c)
            examples.append({"input": [a, b, c], "output": output})

    elif template_name == "min_of_three":
        for _ in range(num_examples - len(examples)):
            a, b, c = np.random.randint(-20, 20, 3)
            output = min(a, b, c)
            examples.append({"input": [a, b, c], "output": output})

    elif template_name == "grade_classifier":
        for _ in range(num_examples - len(examples)):
            score = np.random.randint(0, 101)
            if score >= 90:
                output = 'A'
            elif score >= 80:
                output = 'B'
            elif score >= 70:
                output = 'C'
            elif score >= 60:
                output = 'D'
            else:
                output = 'F'
            examples.append({"input": score, "output": output})

    elif template_name == "sign_function":
        for _ in range(num_examples - len(examples)):
            n = np.random.randint(-50, 51)
            if n > 0:
                output = 1
            elif n < 0:
                output = -1
            else:
                output = 0
            examples.append({"input": n, "output": output})

    elif template_name == "triangle_type":
        for _ in range(num_examples - len(examples)):
            rand_val = np.random.random()
            if rand_val < 0.2:  # 20% equilateral
                side = np.random.randint(1, 10)
                a, b, c = side, side, side
                output = 'e'
            elif rand_val < 0.6:  # 40% isosceles (0.2 to 0.6)
                side1 = np.random.randint(1, 10)
                side2 = np.random.randint(1, 10)
                while side2 == side1:  # Ensure it's not equilateral
                    side2 = np.random.randint(1, 10)
                a, b, c = side1, side1, side2
                output = 'i'
            else:  # 40% scalene (0.6 to 1.0)  
                a, b, c = np.random.randint(1, 10, 3)
                # Ensure all sides are different
                while len(set([a, b, c])) != 3:
                    a, b, c = np.random.randint(1, 10, 3)
                output = 's'
            examples.append({"input": [a, b, c], "output": output})

    elif template_name == "leap_year":
        for _ in range(num_examples - len(examples)):
            year = np.random.randint(1800, 2100)
            output = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            examples.append({"input": year, "output": output})

    # Default fallback for any missing templates
    else:
        # For templates without specific augmentation, just duplicate base examples
        while len(examples) < num_examples:
            if template["base_examples"]:
                base_example = np.random.choice(template["base_examples"])
                examples.append(base_example)
            else:
                break

    # Convert all numpy types to Python types before returning
    cleaned_examples = convert_numpy_to_python(examples[:num_examples])
    return cleaned_examples


def encode_types(type_specs: List[Dict]) -> List[int]:
    """Encode type information using vocabulary"""
    return [TYPE_VOCAB.get(spec.get('type', 'int'), 0) for spec in type_specs]


def simple_text_hash(text: str) -> List[int]:
    """Simple hash-based text encoding for descriptions"""
    hash_val = abs(hash(text))
    return [(hash_val >> i) & 0xFF for i in range(0, 32, 8)]  # 4 hash bytes


def encode_single_example(example: Dict) -> List[float]:
    """Encode one example as compact vector"""
    def flatten_input(inp):
        """Recursively flatten input to list of numbers"""
        if isinstance(inp, list):
            result = []
            for item in inp:
                if isinstance(item, list):
                    result.extend(flatten_input(item))
                elif isinstance(item, str):
                    # Encode string as hash for now (better than nothing)
                    result.append(float(abs(hash(item)) % 10000))
                elif isinstance(item, bool):
                    result.append(float(1 if item else 0))
                else:
                    result.append(float(item))
            return result
        elif isinstance(inp, str):
            # Encode string as hash
            return [float(abs(hash(inp)) % 10000)]
        elif isinstance(inp, bool):
            return [float(1 if inp else 0)]
        else:
            return [float(inp)]

    # Flatten and limit input size
    flat_input = flatten_input(example["input"])
    inputs = flat_input[:8] + [0] * max(0, 8 - len(flat_input))

    # Handle output (could be list or single value)
    if isinstance(example["output"], list):
        # For array outputs, take first few elements
        output_flat = flatten_input(example["output"])[:2]
        output = output_flat + [0] * (2 - len(output_flat))
    elif isinstance(example["output"], str):
        # Encode string output as hash
        output = [float(abs(hash(example["output"])) % 10000), 0]
    elif isinstance(example["output"], bool):
        output = [float(1 if example["output"] else 0), 0]
    else:
        output = [float(example["output"]), 0]

    return inputs + output  # 8 + 2 = 10 dims per example


def encode_program_specification_compressed(spec: Dict[str, Any]) -> np.ndarray:
    """Encode program specification using compressed format"""
    components = []

    # Basic metadata (3 dims)
    components.extend([
        len(spec["inputs"]),
        len(spec["outputs"]),
        len(spec["examples"])
    ])

    # Type information (2-8 dims) - Previously missing!
    input_types = encode_types(spec["inputs"])
    output_types = encode_types(spec["outputs"])
    components.extend(input_types + output_types)

    # Description hash (4 dims)
    desc_hash = simple_text_hash(spec["description"])
    components.extend(desc_hash)

    # Individual examples (10 dims each, variable count)
    for example in spec["examples"]:
        components.extend(encode_single_example(example))

    return np.array(components, dtype=np.float32)
    # Total: ~15 header + 10*num_examples = ~45-215 dims


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


def generate_program_files(program_id: int, template_name: str, template: Dict[str, Any],
                          examples: List[Dict[str, Any]], ast_processor: ASTSimplifier,
                          output_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate individual program files and return compressed arrays"""

    # Create program specification
    spec = {
        "description": template["description"],
        "inputs": template["inputs"],
        "outputs": template["outputs"],
        "examples": examples
    }

    # Generate Python implementation
    python_code = template["implementation"]

    # Generate simplified AST (for human inspection)
    simplified_ast = ast_processor.ast_to_simplified_json(python_code)

    # Generate compressed sparse AST representation
    sparse_ast = ast_processor.ast_to_sparse_representation(python_code)
    ast_tensor = ast_processor.flatten_sparse_ast(sparse_ast)

    # Encode specification using compressed format
    spec_encoding = encode_program_specification_compressed(spec)

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

    return spec_encoding, ast_tensor, sparse_ast


def validate_ast_graph(node_exists: np.ndarray, adjacency: np.ndarray, node_types: np.ndarray) -> bool:
    """Basic validation of AST graph structure (legacy)"""
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


def validate_sparse_ast(sparse_ast: Dict[str, Any]) -> bool:
    """Basic validation of sparse AST structure"""
    # Check that we have at least one node (function definition)
    num_nodes = sparse_ast['num_nodes']
    if num_nodes == 0:
        return False

    # Check that first node is FUNC_DEF
    if len(sparse_ast['node_types']) == 0 or sparse_ast['node_types'][0] != NodeType.FUNC_DEF:
        return False

    # Check edge list is valid
    edge_list = sparse_ast['edge_list']
    if len(edge_list) > 0:
        # All edge indices should be within node range
        if np.any(edge_list >= num_nodes) or np.any(edge_list < 0):
            return False

    return True


def pad_variable_length_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    """Pad variable-length arrays to same size for stacking"""
    if not arrays:
        return np.array([])

    # Find maximum length
    max_len = max(len(arr) for arr in arrays)

    # Pad all arrays to max length
    padded_arrays = []
    for arr in arrays:
        if len(arr) < max_len:
            padded = np.pad(arr, (0, max_len - len(arr)), constant_values=0)
        else:
            padded = arr
        padded_arrays.append(padded)

    return np.stack(padded_arrays, axis=0)


def build_dataset(config: DataProcessConfig):
    """Main dataset building function"""
    np.random.seed(config.seed)

    print(f"Building {config.dataset_name} dataset with {config.num_samples} samples")

    # Initialize AST processor with compressed format
    ast_processor = ASTSimplifier(max_nodes=config.max_nodes, max_edges=config.max_edges)

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
            spec_encoding, ast_tensor, sparse_ast = generate_program_files(
                program_id, template_name, template, examples_subset,
                ast_processor, output_dir
            )

            # Validate sparse AST
            if not validate_sparse_ast(sparse_ast):
                print(f"Warning: Invalid sparse AST for program {program_id}")

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
            # Handle variable-length arrays by padding
            results_np[key] = pad_variable_length_arrays(results[key])

        for key in ["program_identifiers", "program_indices", "group_indices"]:
            results_np[key] = np.array(results[key], dtype=np.int32)

        # Save numpy arrays
        for key, data in results_np.items():
            np.save(os.path.join(split_dir, f"all__{key}.npy"), data)

        # Create metadata for compressed format
        # Input: ~15 header + 10*examples (variable, ~45-215 dims)
        # Output: 2 + 3*max_nodes + 2*max_edges = 2 + 90 + 50 = 142 dims
        compressed_input_size = 15 + 10 * examples_per_template  # Conservative estimate
        compressed_output_size = 2 + 3 * config.max_nodes + 2 * config.max_edges

        metadata = PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,

            vocab_size=30,  # AST node types
            seq_len=compressed_input_size + compressed_output_size,  # Compressed total
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