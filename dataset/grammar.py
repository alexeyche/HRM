from __future__ import annotations

import ast
from typing import Iterable, List, Optional, Sequence, Tuple

import random
from nltk import CFG, Nonterminal
from nltk.parse.generate import generate

from typing import Set


def get_cfg() -> CFG:
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
        "MULOP": ["*", "/"],
        "BINARY_CMP": ["<", ">", "<=", ">=", "==", "!="],
        "AND": ["and"],
        "OR": ["or"],
        "NOT": ["not"],
    }

    non_terminal_rules = {
        # Start
        "S": ["FUNC_DEF"],

        # Function definition
        "FUNC_DEF": ["DEF PROGRAM_NAME LPAREN PARAMS RPAREN COLON NEWLINE INDENT BODY DEDENT"],

        # Parameters
        "PARAMS": ["VARIABLE", "VARIABLE COMMA PARAMS"],

        # Function body: either a statement or an if-block
        "BODY": ["STMT", "IF_BLOCK", "ASSIGNMENT"],

        # Assignment and return
        "ASSIGNMENT": ["VARIABLE EQUALS EXPR NEWLINE"],
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

        # Arithmetic expressions
        "ARITH_EXPR": ["TERM", "ARITH_EXPR ADDOP TERM"],
        "TERM": ["FACTOR", "TERM MULOP FACTOR"],

        # Atoms
        "FACTOR": ["VARIABLE", "DIGIT", "LPAREN EXPR RPAREN"],
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
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "<NEWLINE>":
            code.append("\n")
        elif tok == "<INDENT>":
            indent += 1
            code.append("    " * indent)
        elif tok == "<DEDENT>":
            indent -= 1
            code.append("    " * indent)
        else:
            # space before normal tokens except after newline/indent
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
        else:
            result.extend(generate_random(grammar, r, max_depth-1, local_defined, seed))

    return result


def parse_program(program: str) -> bool:
    try:
        ast.parse(program)
        return True
    except Exception as e:
        return False

def sample_programs(grammar: CFG, n: int = 100, **kwargs) -> List[str]:
    return [realize_program(generate_random(grammar, **kwargs)) for _ in range(n)]


__all__ = [
    "get_cfg",
    "realize_program",
    "sample_programs",
]


