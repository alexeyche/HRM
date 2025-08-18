from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import random
from nltk import CFG, Nonterminal
from nltk.parse.generate import generate


def _quote(token: str) -> str:
    """Quote a terminal token for NLTK grammar text."""
    # Escape single quotes inside tokens, then wrap in single quotes
    return "'" + token.replace("'", "\\'") + "'"


def _make_lexical_rule(lhs: str, terminals: Sequence[str]) -> str:
    """Create a single grammar line for lexical productions.

    Example: A -> 'a' | 'b' | 'c'
    """
    if not terminals:
        return f"{lhs} ->"  # empty alternative (epsilon)
    rhs = " | ".join(_quote(t) for t in terminals)
    return f"{lhs} -> {rhs}"


def get_cfg() -> CFG:
    """Build and return the NLTK CFG equivalent of `dataset/code_generator.py` grammar.

    Notes:
    - We preserve explicit formatting tokens as terminals: SPACE, NEW_LINE, TAB_INDENT.
    - Whitespace is realized later by `realize_program`.
    - Semantic constraints from the original generator (like consistent loop bounds) are
      intentionally omitted because plain CFGs cannot encode them.
    """

    variables = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    # Keep DIGIT domain modest for tractability, while allowing multi-digit terminals like original.
    digits = [str(i) for i in range(0, 21)]

    lines: List[str] = []

    # Formatting tokens
    lines.append(_make_lexical_rule("NEW_LINE", ["NEW_LINE"]))
    lines.append(_make_lexical_rule("TAB_INDENT", ["TAB_INDENT"]))
    lines.append(_make_lexical_rule("BRACKET_OPEN", ["("]))
    lines.append(_make_lexical_rule("BRACKET_CLOSE", [")"]))
    lines.append(_make_lexical_rule("EQUALS", ["="]))
    lines.append(_make_lexical_rule("COLON", [":"]))
    lines.append(_make_lexical_rule("COMMA", [","]))
    # SPACE is a literal token to be realized later
    lines.append(_make_lexical_rule("SPACE", ["SPACE"]))

    # Keywords
    lines.append(_make_lexical_rule("IF", ["if"]))
    lines.append(_make_lexical_rule("ELIF", ["elif"]))
    lines.append(_make_lexical_rule("ELSE", ["else"]))
    lines.append(_make_lexical_rule("FOR", ["for"]))
    lines.append(_make_lexical_rule("IN", ["in"]))
    lines.append(_make_lexical_rule("RANGE", ["range"]))
    lines.append(_make_lexical_rule("WHILE", ["while"]))
    lines.append(_make_lexical_rule("PRINT", ["print"]))

    # Lexical categories
    lines.append(_make_lexical_rule("VARIABLE", variables))
    lines.append(_make_lexical_rule("DIGIT", digits))
    lines.append(_make_lexical_rule("ARITHMETIC_OPERATOR", ["+", "-", "*", "/"]))
    lines.append(_make_lexical_rule("RELATIONAL_OPERATOR", ["<", ">", "<=", ">=", "!=", "=="]))
    lines.append(_make_lexical_rule("LOGICAL_OPERATOR_INFIX", ["and", "or"]))
    lines.append(_make_lexical_rule("LOGICAL_OPERATOR_PREFIX", ["not"]))

    # Non-lexical glue
    lines += [
        # Operators
        "OPERATOR -> ARITHMETIC_OPERATOR",

        # Terms and expressions
        "TERM -> EXPRESSION_IDENTIFIER | DIGIT",
        "EXPRESSION_IDENTIFIER -> VARIABLE | DIGIT",
        "EXPRESSION -> TERM SPACE OPERATOR SPACE TERM",
        "ENCLOSED_EXPRESSION -> BRACKET_OPEN SPACE_OPT EXPRESSION SPACE_OPT BRACKET_CLOSE",
        # Display expressions
        "DISPLAY_EXPRESSION -> EXPRESSION_IDENTIFIER SPACE OPERATOR SPACE EXPRESSION_IDENTIFIER"
        " | EXPRESSION_IDENTIFIER SPACE OPERATOR SPACE DIGIT",

        # Optional space nonterminal (helps with slightly nicer layouts)
        "SPACE_OPT -> SPACE |",  # epsilon

        # Initializations and assignments
        "IDENTIFIER_INITIALIZATION -> IDENTIFIER_INITIALIZATION INITIALIZATION | INITIALIZATION",
        "INITIALIZATION -> VARIABLE SPACE EQUALS SPACE DIGIT NEW_LINE",
        "SIMPLE_ASSIGNMENTS -> VARIABLE SPACE EQUALS SPACE EXPRESSION NEW_LINE |",  # epsilon
        "ADVANCED_ASSIGNMENTS -> VARIABLE SPACE EQUALS SPACE SIMPLE_ARITHMETIC_EVALUATION NEW_LINE"
        " | VARIABLE SPACE EQUALS SPACE EXPRESSION NEW_LINE |",  # epsilon
        "SIMPLE_ARITHMETIC_EVALUATION -> SIMPLE_ARITHMETIC_EVALUATION ARITHMETIC_OPERATOR ENCLOSED_EXPRESSION"
        " | ENCLOSED_EXPRESSION",

        # Conditions
        "SIMPLE_IF_STATEMENT -> IF SPACE CONDITION SPACE COLON NEW_LINE",
        "ADVANCED_IF_STATEMENT -> IF SPACE CHAIN_CONDITION SPACE COLON NEW_LINE",
        "SIMPLE_ELIF_STATEMENT -> ELIF SPACE CONDITION SPACE COLON NEW_LINE",
        "ADVANCED_ELIF_STATEMENT -> ELIF SPACE CHAIN_CONDITION SPACE COLON NEW_LINE",
        "ELSE_STATEMENT -> ELSE SPACE COLON NEW_LINE",
        "CHAIN_CONDITION -> CHAIN_CONDITION SPACE LOGICAL_OPERATOR_INFIX SPACE ENCLOSED_CONDITION"
        " | LOGICAL_OPERATOR_PREFIX SPACE ENCLOSED_CONDITION | ENCLOSED_CONDITION",
        "ENCLOSED_CONDITION -> BRACKET_OPEN CONDITION BRACKET_CLOSE",
        "CONDITION -> OPTIONAL_NOT CONDITION_EXPRESSION | CONDITION_EXPRESSION",
        "CONDITION_EXPRESSION -> EXPRESSION_IDENTIFIER SPACE RELATIONAL_OPERATOR SPACE EXPRESSION_IDENTIFIER"
        " | EXPRESSION_IDENTIFIER SPACE RELATIONAL_OPERATOR SPACE DIGIT",
        "OPTIONAL_NOT -> LOGICAL_OPERATOR_PREFIX SPACE | SPACE",

        # Loops (For)
        "FOR_HEADER -> FOR SPACE EXPRESSION_IDENTIFIER SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL COMMA SPACE STEP BRACKET_CLOSE SPACE COLON"
        " | FOR SPACE EXPRESSION_IDENTIFIER SPACE IN SPACE RANGE BRACKET_OPEN INITIAL COMMA SPACE FINAL BRACKET_CLOSE SPACE COLON",
        "INITIAL -> DIGIT",
        "FINAL -> DIGIT",
        "STEP -> DIGIT",
        "FOR_LOOP -> FOR_HEADER NEW_LINE TAB_INDENT DISPLAY",
        "ADVANCED_FOR_LOOP -> FOR_LOOP | FOR_HEADER NEW_LINE TAB_INDENT ADVANCED_DISPLAY",

        # While Loops (relaxed, without semantic coupling)
        "RELATIONAL_OPERATOR_LESS -> '<' | '<='",
        "RELATIONAL_OPERATOR_GREATER -> '>' | '>='",
        "EXPRESSION_IDENTIFIER_WHILE -> VARIABLE",
        "WHILE_IDENTIFIER -> VARIABLE",
        "FINAL_LESS -> DIGIT",
        "FINAL_GREATER -> DIGIT",
        "CONDITION_EXPRESSION_LESS -> EXPRESSION_IDENTIFIER_WHILE SPACE RELATIONAL_OPERATOR_LESS SPACE FINAL_LESS",
        "CONDITION_EXPRESSION_GREATER -> EXPRESSION_IDENTIFIER_WHILE SPACE RELATIONAL_OPERATOR_GREATER SPACE FINAL_GREATER",
        "WHILE_HEADER_LESS -> WHILE SPACE CONDITION_EXPRESSION_LESS SPACE COLON NEW_LINE",
        "WHILE_LOOP_LESS -> WHILE_HEADER_LESS TAB_INDENT DISPLAY NEW_LINE TAB_INDENT UPDATE_LESS",
        "UPDATE_LESS -> WHILE_IDENTIFIER SPACE EQUALS SPACE WHILE_IDENTIFIER SPACE '+' SPACE STEP",
        "WHILE_HEADER_GREATER -> WHILE SPACE CONDITION_EXPRESSION_GREATER SPACE COLON NEW_LINE",
        "WHILE_LOOP_GREATER -> WHILE_HEADER_GREATER TAB_INDENT DISPLAY NEW_LINE TAB_INDENT UPDATE_GREATER",
        "UPDATE_GREATER -> WHILE_IDENTIFIER SPACE EQUALS SPACE WHILE_IDENTIFIER SPACE '-' SPACE STEP",

        # Displaying
        "DISPLAY -> PRINT SPACE BRACKET_OPEN DISPLAY_IDENTIFIER BRACKET_CLOSE",
        "ADVANCED_DISPLAY -> DISPLAY | PRINT SPACE BRACKET_OPEN DISPLAY_EXPRESSION BRACKET_CLOSE",
        "DISPLAY_IDENTIFIER -> VARIABLE | DIGIT",

        # Top-level compositions (levels)
        "LEVEL1_1 -> IDENTIFIER_INITIALIZATION SIMPLE_ASSIGNMENTS ADVANCED_DISPLAY",
        "LEVEL1_2 -> IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_DISPLAY",
        "LEVEL2_1 -> IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY"
        " | IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY NEW_LINE SIMPLE_ELIF_STATEMENT TAB_INDENT DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT DISPLAY"
        " | IDENTIFIER_INITIALIZATION SIMPLE_IF_STATEMENT TAB_INDENT DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT DISPLAY",
        "LEVEL2_2 -> IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY"
        " | IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ADVANCED_ELIF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT ADVANCED_DISPLAY"
        " | IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_IF_STATEMENT TAB_INDENT ADVANCED_DISPLAY NEW_LINE ELSE_STATEMENT TAB_INDENT ADVANCED_DISPLAY",
        "LEVEL3_1 -> IDENTIFIER_INITIALIZATION FOR_LOOP",
        "LEVEL3_2 -> IDENTIFIER_INITIALIZATION ADVANCED_ASSIGNMENTS ADVANCED_FOR_LOOP",
        "LEVEL4_1 -> IDENTIFIER_INITIALIZATION WHILE_LOOP_LESS | IDENTIFIER_INITIALIZATION WHILE_LOOP_GREATER",
        "ALL -> LEVEL1_1 | LEVEL1_2 | LEVEL2_1 | LEVEL2_2 | LEVEL3_1 | LEVEL3_2 | LEVEL4_1",
    ]

    grammar_text = "\n".join(lines)
    return CFG.fromstring(grammar_text)


def realize_program(tokens: Sequence[str]) -> str:
    """Turn a list of terminal tokens into a program string.

    Replaces formatting placeholders with actual whitespace characters and removes
    token separators by concatenation.
    """
    text = "".join(tokens)
    text = text.replace("SPACE", " ")
    text = text.replace("NEW_LINE", "\n")
    text = text.replace("TAB_INDENT", "\t")
    return text


def sample_programs(
    n: int = 5,
    level: str = "ALL",
    max_depth: int = 30,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample programs from the CFG.

    - level: one of 'ALL', 'LEVEL1.1', 'LEVEL1.2', 'LEVEL2.1', 'LEVEL2.2', 'LEVEL3.1', 'LEVEL3.2', 'LEVEL4.1'
    - max_depth: controls derivation depth during generation
    """
    if seed is not None:
        random.seed(seed)

    cfg = get_cfg()
    start_symbol = level if level == "ALL" else level.replace(".", "_")
    start = Nonterminal(start_symbol)

    programs: List[str] = []
    # NLTK's generate enumerates breadth-first. We'll shuffle the results to add variety
    # and stop after collecting n items.
    all_candidates: List[str] = []
    for sent in generate(cfg, start=start, depth=max_depth):
        all_candidates.append(realize_program(sent))
        if len(all_candidates) >= max(100, n * 10):
            break

    random.shuffle(all_candidates)
    for s in all_candidates[:n]:
        programs.append(s)
    return programs


__all__ = [
    "get_cfg",
    "realize_program",
    "sample_programs",
]


