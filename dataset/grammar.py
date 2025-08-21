from __future__ import annotations

import ast
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import random
from nltk import CFG, Nonterminal
from nltk.parse.generate import generate

from typing import Set


def get_token_patterns() -> Dict[str, List[str]]:
    """Get token patterns from the grammar for use by the tokenizer."""
    variables = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    digits = [str(i) for i in range(0, 21)]

    terminal_rules = {
        "VARIABLE": variables,
        "DIGIT": digits,

        # keywords
        "DEF": ["def"],
        "PROGRAM_NAME": ["program"],
        "RETURN": ["return"],
        "IF": ["if"],
        "ELSE": ["else"],

        # syntax
        "LPAREN": ["("],
        "RPAREN": [")"],
        "COMMA": [","],
        "COLON": [":"],
        "EQUALS": ["="],
        "NEWLINE": ["<NEWLINE>"],
        "INDENT": ["<INDENT>"],
        "DEDENT": ["<DEDENT>"],

        # operators
        "ADDOP": ["+", "-"],
        "MULOP": ["*", "/", "%"],
        "BINARY_CMP": ["<", ">", "<=", ">=", "==", "!="],
        "AND": ["and"],
        "OR": ["or"],
        "NOT": ["not"],

        # brackets and dot
        "LBRACKET": ["["],
        "RBRACKET": ["]"],
        "DOT": ["."],

        # augmented assignment operators
        "ADD_ASSIGN": ["+="],
        "SUB_ASSIGN": ["-="],
        "MUL_ASSIGN": ["*="],
        "DIV_ASSIGN": ["/="],
        "MOD_ASSIGN": ["%="],

        # power operator
        "POWER": ["**"],
        "FLOOR_DIV": ["//"],

        # built-in functions
        "SUM": ["sum"],
        "LEN": ["len"],
        "MIN": ["min"],
        "MAX": ["max"],
        "RANGE": ["range"],
        "ABS": ["abs"],
        "SORTED": ["sorted"],
        "SET": ["set"],
        "STR": ["str"],
        "INT": ["int"],

        # method names
        "APPEND": ["append"],
        "UPPER": ["upper"],
        "LOWER": ["lower"],

        # keywords for function arguments
        "REVERSE": ["reverse"],

        # boolean literals
        "TRUE": ["True"],
        "FALSE": ["False"],

        # string literals - support both empty and non-empty strings
        "STRING": ['""', "''", "STRING"],

        # loops
        "WHILE": ["while"],
        "FOR": ["for"],
        "IN": ["in"],
        "RANGE": ["range"],
        "BREAK": ["break"],
        "CONTINUE": ["continue"],
    }

    return terminal_rules


def get_cfg() -> CFG:
    # Use shared token patterns
    terminal_rules = get_token_patterns()

    non_terminal_rules = {
        # Start
        "S": ["FUNC_DEF"],

        # Function definition
        "FUNC_DEF": ["DEF PROGRAM_NAME LPAREN PARAMS RPAREN COLON NEWLINE INDENT BODY DEDENT"],

        # Parameters
        "PARAMS": ["VARIABLE", "VARIABLE COMMA PARAMS"],

        # Function body: multiple statements
        "BODY": ["STMT_LIST"],
        "STMT_LIST": ["STMT_OR_BLOCK", "STMT_OR_BLOCK STMT_LIST"],
        "STMT_OR_BLOCK": ["STMT", "IF_BLOCK", "WHILE_LOOP", "FOR_LOOP"],

        # Assignment and return
        "ASSIGNMENT": ["SIMPLE_ASSIGN", "AUGMENTED_ASSIGN"],
        "SIMPLE_ASSIGN": ["VARIABLE EQUALS EXPR NEWLINE"],
        "AUGMENTED_ASSIGN": ["VARIABLE ASSIGN_OP EXPR NEWLINE"],
        "ASSIGN_OP": ["ADD_ASSIGN", "SUB_ASSIGN", "MUL_ASSIGN", "DIV_ASSIGN", "MOD_ASSIGN"],
        "STMT": ["RETURN EXPR NEWLINE"],

        # If/else branching
        "IF_BLOCK": ["IF COND COLON NEWLINE INDENT STMT DEDENT ELSE_BLOCK"],
        "ELSE_BLOCK": ["ELSE COLON NEWLINE INDENT STMT DEDENT"],

        # Conditions (just expressions now, since precedence is layered)
        "COND": ["EXPR"],

        # ---- Expressions with operator precedence ----
        "EXPR": ["OR_EXPR"],

        "OR_EXPR": ["AND_EXPR", "OR_EXPR OR AND_EXPR"],

        "AND_EXPR": ["NOT_EXPR", "AND_EXPR AND NOT_EXPR"],

        "NOT_EXPR": ["NOT NOT_EXPR", "COMPARISON"],

        # Comparisons like a < b <= c
        "COMPARISON": ["ARITH_EXPR", "ARITH_EXPR COMP_CHAIN"],
        "COMP_CHAIN": ["BINARY_CMP ARITH_EXPR", "BINARY_CMP ARITH_EXPR COMP_CHAIN"],

        # Arithmetic expressions (with operator precedence)
        "ARITH_EXPR": ["TERM", "ARITH_EXPR ADDOP TERM"],
        "TERM": ["POWER_EXPR", "TERM MULOP POWER_EXPR", "TERM FLOOR_DIV POWER_EXPR"],
        "POWER_EXPR": ["FACTOR", "FACTOR POWER POWER_EXPR"],

        # Atoms
        "FACTOR": ["VARIABLE", "DIGIT", "STRING", "LPAREN EXPR RPAREN", "FUNCTION_CALL", "LIST_LITERAL", "LIST_INDEX", "METHOD_CALL"],

        # List literals
        "LIST_LITERAL": ["LBRACKET LIST_CONTENTS RBRACKET"],
        "LIST_CONTENTS": ["", "EXPR_LIST"],
        "EXPR_LIST": ["EXPR", "EXPR COMMA EXPR_LIST"],

        # List indexing
        "LIST_INDEX": ["VARIABLE LBRACKET EXPR RBRACKET"],

        # Method calls
        "METHOD_CALL": ["VARIABLE DOT METHOD_NAME LPAREN ARG_LIST RPAREN"],
        "METHOD_NAME": ["APPEND", "UPPER", "LOWER"],

        # Function calls (built-in and user-defined)
        "FUNCTION_CALL": ["BUILTIN_FUNC", "RANGE_CALL", "VARIABLE LPAREN ARG_LIST RPAREN"],
        "BUILTIN_FUNC": [
            "SUM LPAREN EXPR RPAREN",
            "LEN LPAREN EXPR RPAREN",
            "MIN LPAREN EXPR RPAREN",
            "MAX LPAREN EXPR RPAREN",
            "ABS LPAREN EXPR RPAREN",
            "SORTED LPAREN EXPR RPAREN",
            "SORTED LPAREN EXPR COMMA REVERSE_ARG RPAREN",
            "SET LPAREN EXPR RPAREN",
            "STR LPAREN EXPR RPAREN",
            "INT LPAREN EXPR RPAREN"
        ],
        "REVERSE_ARG": ["REVERSE EQUALS BOOL_VALUE"],
        "BOOL_VALUE": ["TRUE", "FALSE"],
        "ARG_LIST": ["", "EXPR", "EXPR COMMA ARG_LIST"],
        "RANGE_CALL": ["RANGE LPAREN RANGE_ARGS RPAREN"],
        "RANGE_ARGS": ["SIMPLE_EXPR", "SIMPLE_EXPR COMMA SIMPLE_EXPR", "SIMPLE_EXPR COMMA SIMPLE_EXPR COMMA SIMPLE_EXPR"],
        "SIMPLE_EXPR": ["VARIABLE", "DIGIT", "LPAREN SIMPLE_EXPR RPAREN"],

        # Loop constructs - separate from function statements
        "WHILE_LOOP": ["WHILE COND COLON NEWLINE INDENT LOOP_BODY DEDENT"],
        "FOR_LOOP": ["FOR VARIABLE IN ITERABLE COLON NEWLINE INDENT LOOP_BODY DEDENT"],
        "ITERABLE": ["RANGE_CALL", "VARIABLE"],
        "LOOP_BODY": ["LOOP_STMT_LIST"],
        "LOOP_STMT_LIST": ["LOOP_STMT", "LOOP_STMT LOOP_STMT_LIST"],
        "LOOP_STMT": ["ASSIGNMENT", "IF_BLOCK", "BREAK_STMT", "CONTINUE_STMT"],

        # Function statements (cannot have return in loops)
        "STMT": ["ASSIGNMENT", "RETURN_STMT"],
        "RETURN_STMT": ["RETURN EXPR NEWLINE"],
        "BREAK_STMT": ["BREAK NEWLINE"],
        "CONTINUE_STMT": ["CONTINUE NEWLINE"],
    }


    lines = []
    for lhs, rhs_list in terminal_rules.items():
        for rhs_item in rhs_list:
            lines.append(f"{lhs} -> '{rhs_item}'")

    for lhs, rhs in non_terminal_rules.items():
        rhs_str = " | ".join(rhs)
        lines.append(f"{lhs} -> {rhs_str}")

    grammar_text = "\n".join(lines)
    grammar = CFG.fromstring(grammar_text)
    grammar._start = Nonterminal("S")

    return grammar



def realize_program(tokens: Sequence[str]) -> str:
    code = []
    indent = 0
    pending_indent = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "<NEWLINE>":
            code.append("\n")
            pending_indent = True
        elif tok == "<INDENT>":
            indent += 1
            pending_indent = True
        elif tok == "<DEDENT>":
            indent -= 1
            pending_indent = True
        else:
            # Add pending indentation before the first real token
            if pending_indent:
                code.append("    " * indent)
                pending_indent = False
            # space before normal tokens except after newline/indent/dedent
            if code and not code[-1].endswith(("\n", " ", "(", "[")):
                code.append(" ")
            code.append(tok)
        i += 1
    return "".join(code)



def generate_random(
    grammar: CFG,
    symbol: Optional[Nonterminal] = None,
    max_depth: int = 6,
    defined_vars: Optional[Set[str]] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Randomly generate a sentence from a CFG with depth control and variable tracking."""
    if seed is not None:
        random.seed(seed)
        seed += 1

    if defined_vars is None:
        defined_vars = set()

    if symbol is None:
        symbol = grammar.start()

    # Terminal symbol
    if not isinstance(symbol, Nonterminal):
        return [str(symbol)]

    # Get productions for this non-terminal
    prods = grammar.productions(lhs=symbol)

    if max_depth <= 0:
        # Filter out self-recursive productions
        prods = [
            p for p in prods
            if all(not (isinstance(r, Nonterminal) and r == symbol) for r in p.rhs())
        ]
        if not prods:
            prods = grammar.productions(lhs=symbol)

    # Pick a random production
    prod = random.choice(list(prods))
    result = []

    # Track local variable definitions
    local_defined = defined_vars.copy()

    for r in prod.rhs():
        # Special handling: assignments define variables
        if isinstance(r, Nonterminal) and r.symbol() == "ASSIGNMENT":
            stmt_tokens = generate_random(grammar, r, max_depth-1, local_defined, seed)
            # first token is VARIABLE (assume VARIABLE EQUALS ...)
            local_defined.add(stmt_tokens[0])
            result.extend(stmt_tokens)

        elif isinstance(r, Nonterminal) and r.symbol() == "PARAMS":
            params_tokens = generate_random(grammar, r, max_depth-1, local_defined, seed)
            for tok in params_tokens:
                if tok.isalpha():  # simple check for variable name
                    local_defined.add(tok)
            result.extend(params_tokens)

        elif isinstance(r, Nonterminal) and r.symbol() == "VARIABLE":
            # Pick only from defined variables if any
            if local_defined:
                result.append(random.choice(list(local_defined)))
            else:
                # fallback: pick a random single-letter variable
                result.append(random.choice([chr(c) for c in range(ord('a'), ord('z')+1)]))

        elif isinstance(r, Nonterminal) and r.symbol() == "FUNCTION_CALL":
            # For function calls, prefer range over other functions
            if random.random() < 0.9:  # 90% chance for range
                func_tokens = generate_random(grammar, Nonterminal("RANGE_CALL"), max_depth-1, local_defined, seed)
            else:
                # Generate a simple function call with a random variable name
                func_name = random.choice([chr(c) for c in range(ord('a'), ord('z')+1)])
                # Ensure we have at least one argument
                if random.random() < 0.8:  # 80% chance of having arguments
                    arg_list = generate_random(grammar, Nonterminal("ARG_LIST"), max_depth-1, local_defined, seed)
                    func_tokens = [func_name, "(", *arg_list, ")"]
                else:
                    func_tokens = [func_name, "(", ")"]
            result.extend(func_tokens)

        elif isinstance(r, Nonterminal) and r.symbol() == "RANGE_ARGS":
            # Choose between 1, 2, or 3 arguments for range, using simple expressions
            arg_options = [
                [Nonterminal("SIMPLE_EXPR")],  # range(n)
                [Nonterminal("SIMPLE_EXPR"), ",", Nonterminal("SIMPLE_EXPR")],  # range(start, stop)
                [Nonterminal("SIMPLE_EXPR"), ",", Nonterminal("SIMPLE_EXPR"), ",", Nonterminal("SIMPLE_EXPR")]  # range(start, stop, step)
            ]
            chosen_args = random.choice(arg_options)
            for arg in chosen_args:
                if isinstance(arg, str):
                    result.append(arg)
                else:
                    result.extend(generate_random(grammar, arg, max_depth-1, local_defined, seed))

        elif isinstance(r, Nonterminal) and r.symbol() == "SIMPLE_EXPR":
            # Generate simple expressions for range arguments
            simple_options = [
                [Nonterminal("VARIABLE")],
                [Nonterminal("DIGIT")],
                [Nonterminal("DIGIT")],
                [Nonterminal("LPAREN"), Nonterminal("SIMPLE_EXPR"), Nonterminal("RPAREN")]
            ]
            chosen = random.choice(simple_options)
            for item in chosen:
                if isinstance(item, str):
                    result.append(item)
                else:
                    result.extend(generate_random(grammar, item, max_depth-1, local_defined, seed))

        elif isinstance(r, Nonterminal) and r.symbol() == "ARG_LIST":
            # Handle argument lists - prefer to have some arguments
            if random.random() < 0.1:  # 10% chance of empty args
                pass  # No arguments
            else:
                # Generate 1-2 arguments with simple expressions
                num_args = random.randint(1, 2)
                for i in range(num_args):
                    if i > 0:
                        result.append(",")
                    result.extend(generate_random(grammar, Nonterminal("SIMPLE_EXPR"), max_depth-1, local_defined, seed))

        elif isinstance(r, Nonterminal) and r.symbol() == "LOOP_STMT":
            # For loop statements, only allow valid loop constructs
            stmt_options = ["ASSIGNMENT", "BREAK_STMT", "CONTINUE_STMT"]
            if max_depth > 2 and random.random() < 0.4:  # Higher chance for nested constructs
                stmt_options.extend(["IF_BLOCK"])

            stmt_type = random.choice(stmt_options)
            result.extend(generate_random(grammar, Nonterminal(stmt_type), max_depth-1, local_defined, seed))

        elif isinstance(r, Nonterminal) and r.symbol() == "STMT_LIST":
            # Generate 1-4 statements for function body
            num_stmts = random.randint(1, 4)
            for i in range(num_stmts):
                if i == num_stmts - 1:  # Last statement should be return
                    result.extend(generate_random(grammar, Nonterminal("STMT"), max_depth-1, local_defined, seed))
                else:
                    result.extend(generate_random(grammar, Nonterminal("STMT_OR_BLOCK"), max_depth-1, local_defined, seed))

        elif isinstance(r, Nonterminal) and r.symbol() == "STMT_OR_BLOCK":
            # Choose statement type with preference for assignments and conditionals
            stmt_options = ["STMT", "STMT", "IF_BLOCK", "WHILE_LOOP", "FOR_LOOP", "ASSIGNMENT", "ASSIGNMENT", "ASSIGNMENT"]
            stmt_type = random.choice(stmt_options)
            result.extend(generate_random(grammar, Nonterminal(stmt_type), max_depth-1, local_defined, seed))

        elif isinstance(r, Nonterminal) and r.symbol() == "LOOP_STMT_LIST":
            # Generate 1-3 statements for loop body
            num_stmts = random.randint(1, 3)
            for i in range(num_stmts):
                result.extend(generate_random(grammar, Nonterminal("LOOP_STMT"), max_depth-1, local_defined, seed))

        elif isinstance(r, Nonterminal) and r.symbol() == "STRING":
            # Generate small string literals (1-3 characters)
            if random.random() < 0.3:  # 30% chance for empty string
                result.append(random.choice(['""', "''"]))
            else:
                # Generate small strings with 1-3 characters
                length = random.randint(1, 3)
                chars = []
                for _ in range(length):
                    if random.random() < 0.7:  # 70% chance for lowercase letters
                        chars.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
                    else:
                        chars.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

                # Randomly choose single or double quotes
                quote = random.choice(["'", '"'])
                string_content = ''.join(chars)
                result.append(f"{quote}{string_content}{quote}")

        else:
            result.extend(generate_random(grammar, r, max_depth-1, local_defined, seed))

    return result


def parse_program_with_ast(program: str) -> bool:
    try:
        ast.parse(program)
        return True
    except Exception as e:
        return False

def sample_programs(grammar: CFG, n: int = 100, **kwargs) -> List[str]:
    return [realize_program(generate_random(grammar, **kwargs)) for _ in range(n)]


# Function registry documenting supported built-in functions and their argument specifications
SUPPORTED_FUNCTIONS = {
    # Built-in functions with argument counts
    "sum": {"args": 1, "description": "Sum all elements in an iterable"},
    "len": {"args": 1, "description": "Get length of an iterable or string"},
    "min": {"args": 1, "description": "Find minimum element in an iterable"},
    "max": {"args": 1, "description": "Find maximum element in an iterable"},
    "abs": {"args": 1, "description": "Get absolute value of a number"},
    "str": {"args": 1, "description": "Convert value to string"},
    "int": {"args": 1, "description": "Convert value to integer"},
    "range": {"args": [1, 2, 3], "description": "Generate range of numbers"},
    "sorted": {"args": [1, 2], "description": "Sort an iterable"},
    "set": {"args": 1, "description": "Convert iterable to set"},
    "list": {"args": 1, "description": "Convert iterable to list"},
}

SUPPORTED_METHODS = {
    # String methods
    "upper": {"args": 0, "description": "Convert string to uppercase"},
    "lower": {"args": 0, "description": "Convert string to lowercase"},
    # List methods
    "append": {"args": 1, "description": "Append item to list"},
}

__all__ = [
    "get_cfg",
    "get_token_patterns",
    "realize_program",
    "sample_programs",
    "SUPPORTED_FUNCTIONS",
    "SUPPORTED_METHODS",
]


