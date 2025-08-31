from __future__ import annotations

import ast
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Union

import random
import torch
from nltk import CFG, Nonterminal
from nltk.grammar import Production
from nltk.parse.generate import generate
from nltk.parse.earleychart import EarleyChartParser
from typing import Set
import torch_struct



def get_token_patterns() -> Dict[str, List[str]]:
    """Get token patterns from the grammar for use by the tokenizer."""
    variables = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    digits = [str(i) for i in range(0, 21)] + ["0.5"] + [
        # Additional numbers used in the programs
        "60", "70", "80", "90", "100", "400"
    ]

    terminal_rules = {
        "VARIABLE": variables,
        "DIGIT": digits,

        # keywords
        "DEF": ["def"],
        "PROGRAM_NAME": ["program"],
        "RETURN": ["return"],
        "IF": ["if"],
        "ELIF": ["elif"],
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
        "BINARY_CMP": ["<", ">", "<=", ">=", "==", "!=", "in"],
        "AND": ["and"],
        "OR": ["or"],
        "NOT": ["not"],

        # brackets and rest
        "LBRACKET": ["["],
        "RBRACKET": ["]"],
        "DOT": ["."],
        "UNDERSCORE": ["_"],

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
        "LIST": ["list"],

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
        "STRING": ['""', "''", "<STRING>"],

        # loops
        "WHILE": ["while"],
        "FOR": ["for"],
        "IN": ["in"],
        "RANGE": ["range"],
        "BREAK": ["break"],
        "CONTINUE": ["continue"],
    }

    return terminal_rules


def get_cfg(start: str = "S") -> CFG:
    # Use shared token patterns
    terminal_rules = get_token_patterns()

    non_terminal_rules = {
        # Start
        "S": ["FUNC_DEF"],

        # Function definition
        "FUNC_DEF": ["DEF PROGRAM_NAME LPAREN PARAMS RPAREN COLON NEWLINES INDENT BODY DEDENT"],
        "NEWLINES": ["NEWLINE", "NEWLINE NEWLINES"],

        # Parameters
        "PARAMS": ["VARIABLE", "VARIABLE COMMA PARAMS"],

        # Function body: multiple statements
        "BODY": ["STMT_LIST"],
        "STMT_LIST": ["STMT_OR_BLOCK", "STMT_OR_BLOCK STMT_LIST"],
        "STMT_OR_BLOCK": ["STMT", "IF_BLOCK", "WHILE_LOOP", "FOR_LOOP"],

        # Assignment and return
        "ASSIGNMENT": ["SIMPLE_ASSIGN", "AUGMENTED_ASSIGN", "TUPLE_ASSIGN"],
        "SIMPLE_ASSIGN": ["VARIABLE EQUALS EXPR NEWLINE"],
        "AUGMENTED_ASSIGN": ["VARIABLE ASSIGN_OP EXPR NEWLINE"],
        "TUPLE_ASSIGN": ["VARIABLE_LIST EQUALS EXPR_LIST NEWLINE"],
        "VARIABLE_LIST": ["VARIABLE", "VARIABLE COMMA VARIABLE_LIST"],
        "ASSIGN_OP": ["ADD_ASSIGN", "SUB_ASSIGN", "MUL_ASSIGN", "DIV_ASSIGN", "MOD_ASSIGN"],

        # If/elif/else branching
        "IF_BLOCK": ["IF COND COLON NEWLINE INDENT STMT DEDENT", "IF COND COLON NEWLINE INDENT STMT DEDENT ELIF_CHAIN"],
        "ELIF_BLOCK": ["ELIF COND COLON NEWLINE INDENT STMT DEDENT"],
        "ELIF_CHAIN": ["ELIF_BLOCK", "ELIF_BLOCK ELIF_CHAIN", "ELSE_BLOCK"],
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

        # Atoms and unary expressions
        "FACTOR": ["UNARY_EXPR", "VARIABLE", "DIGIT", "STRING", "TRUE", "FALSE", "LPAREN EXPR RPAREN", "FUNCTION_CALL", "LIST_LITERAL", "LIST_INDEX", "METHOD_CALL"],
        "UNARY_EXPR": ["ADDOP FACTOR"],

        # List literals
        "LIST_LITERAL": ["LBRACKET LIST_CONTENTS RBRACKET"],
        "LIST_CONTENTS": ["", "EXPR_LIST"],
        "EXPR_LIST": ["EXPR", "EXPR COMMA EXPR_LIST", "LPAREN EXPR_LIST RPAREN"],

        # List indexing and slicing
        "LIST_INDEX": ["VARIABLE LBRACKET EXPR RBRACKET", "FUNCTION_CALL LBRACKET SLICE RBRACKET"],
        "SLICE": ["COLON COLON UNARY_EXPR"],

        # Method calls
        "METHOD_CALL": ["VARIABLE DOT METHOD_NAME LPAREN ARG_LIST RPAREN"],
        "METHOD_NAME": ["APPEND", "UPPER", "LOWER"],

        # Function calls (built-in and user-defined)
        "FUNCTION_CALL": ["BUILTIN_FUNC", "RANGE_CALL", "VARIABLE LPAREN ARG_LIST RPAREN"],
        "BUILTIN_FUNC": [
            "SUM LPAREN EXPR RPAREN",
            "LEN LPAREN EXPR RPAREN",
            "MIN LPAREN EXPR RPAREN",
            "MIN LPAREN EXPR_LIST RPAREN",
            "MAX LPAREN EXPR RPAREN",
            "MAX LPAREN EXPR_LIST RPAREN",
            "ABS LPAREN EXPR RPAREN",
            "SORTED LPAREN EXPR RPAREN",
            "SORTED LPAREN EXPR COMMA REVERSE_ARG RPAREN",
            "SET LPAREN EXPR RPAREN",
            "STR LPAREN EXPR RPAREN",
            "INT LPAREN EXPR RPAREN",
            "LIST LPAREN EXPR RPAREN",
        ],
        "REVERSE_ARG": ["REVERSE EQUALS BOOL_VALUE"],
        "BOOL_VALUE": ["TRUE", "FALSE"],
        "ARG_LIST": ["", "EXPR", "EXPR COMMA ARG_LIST"],
        "RANGE_CALL": ["RANGE LPAREN RANGE_ARGS RPAREN"],
        "RANGE_ARGS": ["EXPR", "EXPR COMMA EXPR", "EXPR COMMA EXPR COMMA EXPR"],
        "SIMPLE_EXPR": ["VARIABLE", "DIGIT", "LPAREN SIMPLE_EXPR RPAREN"],

        # Loop constructs - separate from function statements
        "WHILE_LOOP": ["WHILE COND COLON NEWLINE INDENT LOOP_BODY DEDENT"],
        "FOR_VARIABLE": ["VARIABLE", "UNDERSCORE"],
        "FOR_LOOP": ["FOR FOR_VARIABLE IN ITERABLE COLON NEWLINE INDENT LOOP_BODY DEDENT"],
        "ITERABLE": ["RANGE_CALL", "VARIABLE", "FUNCTION_CALL"],
        "LOOP_BODY": ["LOOP_STMT_LIST"],
        "LOOP_STMT_LIST": ["LOOP_STMT", "LOOP_STMT LOOP_STMT_LIST"],
        "LOOP_STMT": ["ASSIGNMENT", "LOOP_IF_BLOCK", "WHILE_LOOP", "FOR_LOOP", "BREAK_STMT", "CONTINUE_STMT", "LOOP_EXPR_STMT", "RETURN_STMT"],
        "LOOP_EXPR_STMT": ["METHOD_CALL NEWLINE"],

        # If statements within loops (allow loop statements in body)
        "LOOP_IF_BLOCK": ["IF COND COLON NEWLINE INDENT LOOP_STMT_LIST DEDENT", "IF COND COLON NEWLINE INDENT LOOP_STMT_LIST DEDENT LOOP_ELIF_CHAIN"],
        "LOOP_ELIF_BLOCK": ["ELIF COND COLON NEWLINE INDENT LOOP_STMT_LIST DEDENT"],
        "LOOP_ELIF_CHAIN": ["LOOP_ELIF_BLOCK", "LOOP_ELIF_BLOCK LOOP_ELIF_CHAIN", "LOOP_ELSE_BLOCK"],
        "LOOP_ELSE_BLOCK": ["ELSE COLON NEWLINE INDENT LOOP_STMT_LIST DEDENT"],

        # Function statements
        "STMT": ["ASSIGNMENT", "RETURN_STMT", "EXPR_STMT"],
        "EXPR_STMT": ["METHOD_CALL NEWLINE"],
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
    # assert start in grammar._lexical_index, f"Start symbol {start} not found in grammar"
    grammar._start = Nonterminal(start)

    return grammar


def get_parser(grammar: CFG | None= None, start: str = "S") -> EarleyChartParser:
    if grammar is None:
        grammar = get_cfg(start)
    return EarleyChartParser(grammar)

def parse_tokens_to_productions(tokens: List[str], grammar: CFG) -> Tuple[List[Tuple[Nonterminal, int]], List[Tuple[str, str, Optional[List[str]]]]]:
    """
    Parse a token sequence and map it to complete grammar derivation sequence with context tracking.

    Uses NLTK's EarleyChartParser to get the exact parse tree, then extracts
    the complete sequence of productions used in the derivation. Additionally tracks
    identifier context for proper copy mechanism training.

    Args:
        tokens: List of program tokens
        grammar: CFG grammar

    Returns:
        Tuple of (production_sequence, terminal_requirements)
        - production_sequence: Complete list of (nonterminal, production_idx) pairs in derivation order
        - terminal_requirements: List of (terminal_type, target_value, context) tuples
          where context is the list of identifiers available at that point (for VARIABLE terminals)
    """
    from nltk.parse.earleychart import EarleyChartParser
    from nltk.tree import Tree

    if not tokens:
        return [], []

    # Build production mappings for efficient lookup
    production_to_idx = {}
    for i, prod in enumerate(grammar.productions()):
        production_to_idx[prod] = i

    # Build token patterns for terminal classification
    token_patterns = get_token_patterns()

    def classify_token(token: str) -> str:
        """Classify a token into its terminal type based on grammar patterns."""
        # Check direct matches in token patterns
        for terminal_type, patterns in token_patterns.items():
            if token in patterns:
                return terminal_type

        # Pattern-based classification
        if token.isalpha() and len(token) == 1 and token.islower():
            return "VARIABLE"
        elif token.isdigit():
            return "DIGIT"
        elif token == "True":
            return "TRUE"
        elif token == "False":
            return "FALSE"
        elif (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
            return "STRING"

        return "UNKNOWN"

    def extract_productions_from_tree(tree: Tree) -> List[Tuple[Nonterminal, int]]:
        """Extract production sequence from parse tree in derivation order."""
        productions = []

        def traverse_tree(node):
            if isinstance(node, Tree):
                # This is a non-terminal node
                label = node.label()
                if isinstance(label, str):
                    label = Nonterminal(label)

                # Find the production used for this expansion
                rhs_symbols = []
                for child in node:
                    if isinstance(child, Tree):
                        rhs_symbols.append(Nonterminal(child.label()))
                    else:
                        # Terminal - we need to find which terminal symbol this maps to
                        child_str = str(child)
                        # Look for exact match in grammar terminals
                        found_terminal = None
                        for prod in grammar.productions():
                            for rhs_symbol in prod.rhs():
                                if str(rhs_symbol) == f"'{child_str}'" or str(rhs_symbol) == child_str:
                                    found_terminal = rhs_symbol
                                    break
                            if found_terminal:
                                break

                        if found_terminal:
                            rhs_symbols.append(found_terminal)
                        else:
                            # Fallback: classify and find matching terminal
                            terminal_type = classify_token(child_str)
                            for prod in grammar.productions():
                                for rhs_symbol in prod.rhs():
                                    if str(rhs_symbol) == terminal_type:
                                        rhs_symbols.append(rhs_symbol)
                                        break
                                if rhs_symbols and str(rhs_symbols[-1]) == terminal_type:
                                    break

                # Find matching production
                for prod in grammar.productions(lhs=label):
                    if len(prod.rhs()) == len(rhs_symbols):
                        # Check if RHS matches (allowing for some flexibility)
                        match = True
                        for i, (expected, actual) in enumerate(zip(prod.rhs(), rhs_symbols)):
                            if str(expected) != str(actual):
                                # Allow terminal flexibility
                                if not isinstance(expected, Nonterminal) and not isinstance(actual, Nonterminal):
                                    continue  # Both terminals, allow mismatch for now
                                elif isinstance(expected, Nonterminal) and isinstance(actual, Nonterminal):
                                    if expected.symbol() != actual.symbol():
                                        match = False
                                        break
                                else:
                                    match = False
                                    break

                        if match:
                            prod_idx = production_to_idx.get(prod)
                            if prod_idx is not None:
                                productions.append((label, prod_idx))
                            break

                # Recursively process children
                for child in node:
                    traverse_tree(child)

        traverse_tree(tree)
        return productions

    # Parse using EarleyChartParser - no fallbacks, must work precisely
    parser = EarleyChartParser(grammar)
    parse_trees = list(parser.parse(tokens))

    if not parse_trees:
        raise ValueError(f"Failed to parse tokens {tokens} with grammar. This indicates either "
                        f"the tokens are not valid according to the grammar, or there's an issue "
                        f"with token classification. Grammar start: {grammar.start()}")

    # Use the first parse tree (handle ambiguity by taking first)
    tree = parse_trees[0]
    production_sequence = extract_productions_from_tree(tree)

    if not production_sequence:
        raise ValueError(f"Failed to extract production sequence from parse tree for tokens {tokens}")

    # Build terminal requirements with simplified context tracking
    def build_terminal_requirements_with_context(tokens: List[str]) -> List[Tuple[str, str, Optional[List[str]]]]:
        """Build terminal requirements with simplified, deterministic identifier context tracking."""
        terminal_requirements = []
        identifier_context = []  # Track identifiers as they are defined

        # Simplified approach: scan tokens sequentially and detect definitions
        # This is more deterministic than trying to track complex parsing state
        for i, token in enumerate(tokens):
            terminal_type = classify_token(token)

            if terminal_type == "UNKNOWN":
                raise ValueError(f"Unknown terminal type for token '{token}'. This token is not "
                               f"recognized by the grammar's token patterns.")

            current_context = None
            if terminal_type == "VARIABLE":
                # Always provide current context for variables
                current_context = list(identifier_context)

                # Simple definition detection: check common patterns
                is_definition = False

                # Pattern 1: Assignment - var = value
                if i + 1 < len(tokens) and tokens[i + 1] == "=":
                    is_definition = True

                # Pattern 2: Function parameter - def func(var) or func(var1, var2)
                elif (i > 0 and tokens[i - 1] in ["(", ","]):
                    # Check if we're in a function definition context
                    # Look backwards for 'def' keyword within reasonable distance
                    for j in range(max(0, i - 10), i):
                        if tokens[j] == "def":
                            is_definition = True
                            break

                # Pattern 3: for loop variable - for var in ...
                elif i > 0 and tokens[i - 1] == "for":
                    is_definition = True
                elif i > 1 and tokens[i - 2] == "for" and tokens[i - 1] in [" ", ""]:
                    is_definition = True

                # Add to context if this is a definition and not already present
                if is_definition and token not in identifier_context:
                    identifier_context.append(token)

            # Add terminal requirement with context
            terminal_requirements.append((terminal_type, token, current_context))

        return terminal_requirements

    terminal_requirements = build_terminal_requirements_with_context(tokens)
    return production_sequence, terminal_requirements




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

class TorchStructCFGConverter:
    """
    Convert NLTK CFG to torch-struct format for Neural PCFG operations.

    Torch-struct expects:
    - terms: (N x T) terminal emission scores
    - rules: (NT x (NT+T) x (NT+T)) binary rule scores
    - root: (NT) root nonterminal scores
    """

    def __init__(self, nltk_cfg: CFG):
        self.nltk_cfg = nltk_cfg

        # Build symbol mappings
        self.all_terminals = set()
        self.all_nonterminals = set()
        self.binary_rules = []  # Rules with exactly 2 RHS symbols
        self.terminal_rules = []  # Rules that produce terminals

        self._extract_symbols_and_rules()

        # Create ordered lists for indexing
        self.terminals = sorted(list(self.all_terminals))
        self.nonterminals = sorted(list(self.all_nonterminals))

        # Create mappings
        self.terminal_to_idx = {term: i for i, term in enumerate(self.terminals)}
        self.nonterminal_to_idx = {nt: i for i, nt in enumerate(self.nonterminals)}

        # Combined symbol space: [nonterminals, terminals]
        self.symbol_to_idx = {}
        for i, nt in enumerate(self.nonterminals):
            self.symbol_to_idx[nt] = i
        for i, term in enumerate(self.terminals):
            self.symbol_to_idx[term] = len(self.nonterminals) + i

    def _extract_symbols_and_rules(self):
        """Extract terminals, nonterminals, and categorize rules from NLTK CFG."""
        for production in self.nltk_cfg.productions():
            lhs = production.lhs()
            rhs = production.rhs()

            self.all_nonterminals.add(lhs)

            # Categorize by RHS length and content
            if len(rhs) == 2:
                # Binary rule: A -> B C
                self.binary_rules.append(production)
                for symbol in rhs:
                    if isinstance(symbol, Nonterminal):
                        self.all_nonterminals.add(symbol)
                    else:
                        self.all_terminals.add(str(symbol))
            elif len(rhs) == 1:
                symbol = rhs[0]
                if isinstance(symbol, Nonterminal):
                    # Unary rule: A -> B (will need special handling)
                    self.all_nonterminals.add(symbol)
                else:
                    # Terminal rule: A -> 'token'
                    self.terminal_rules.append(production)
                    self.all_terminals.add(str(symbol))

    def convert_to_cnf(self) -> CFG:
        """
        Convert grammar to Chomsky Normal Form for torch-struct compatibility.

        CNF requires:
        1. A -> BC (two nonterminals)
        2. A -> a (one terminal)
        3. S -> ε (only start symbol can produce empty)
        """
        # For now, we'll implement a simplified CNF conversion
        # that handles the most common cases in our grammar

        cnf_productions = []

        # Add binary rules as-is (already in CNF form A -> BC)
        for prod in self.binary_rules:
            cnf_productions.append(prod)

        # Add terminal rules as-is (already in CNF form A -> a)
        for prod in self.terminal_rules:
            cnf_productions.append(prod)

        # Handle other rules by introducing intermediate nonterminals
        for production in self.nltk_cfg.productions():
            if production not in self.binary_rules and production not in self.terminal_rules:
                rhs = production.rhs()
                if len(rhs) > 2:
                    # Convert A -> B C D to A -> B X1, X1 -> C D
                    # This is a simplified conversion - a full implementation
                    # would handle all cases recursively
                    lhs = production.lhs()

                    # Create intermediate nonterminal
                    intermediate = Nonterminal(f"{lhs}_INT")

                    # A -> B X_INT
                    cnf_productions.append(Production(lhs, [rhs[0], intermediate]))

                    # X_INT -> C D (assuming length 3 for simplicity)
                    if len(rhs) == 3:
                        cnf_productions.append(Production(intermediate, rhs[1:]))
                elif len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
                    # Unary rule A -> B: convert to A -> B epsilon_B, epsilon_B -> ε
                    # For torch-struct, we'll handle this by treating it as a binary rule
                    # with a special epsilon nonterminal
                    epsilon_nt = Nonterminal("EPSILON")
                    cnf_productions.append(Production(production.lhs(), [rhs[0], epsilon_nt]))
                    cnf_productions.append(Production(epsilon_nt, ['<EMPTY>']))

        return CFG(self.nltk_cfg.start(), cnf_productions)

    def to_torch_struct_tensors(self, sequence_length: int,
                              device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert NLTK CFG to torch-struct tensor format.

        Args:
            sequence_length: Length of sequences to parse
            device: Device to place tensors on

        Returns:
            Tuple of (terms, rules, root) tensors for torch-struct.SentCFG
        """
        if device is None:
            device = torch.device('cpu')

        N = sequence_length
        T = len(self.terminals)
        NT = len(self.nonterminals)

        # Initialize with small negative values (logits)
        terms = torch.full((N, T), -10.0, device=device)
        rules = torch.full((NT, NT + T, NT + T), -10.0, device=device)
        root = torch.full((NT,), -10.0, device=device)

        # Set root nonterminal (start symbol)
        start_symbol = self.nltk_cfg.start()
        if start_symbol in self.nonterminal_to_idx:
            root[self.nonterminal_to_idx[start_symbol]] = 0.0

        # Process binary rules: A -> B C
        for production in self.binary_rules:
            lhs = production.lhs()
            rhs = production.rhs()

            if len(rhs) == 2 and lhs in self.nonterminal_to_idx:
                lhs_idx = self.nonterminal_to_idx[lhs]

                # Get indices for RHS symbols
                rhs_indices = []
                for symbol in rhs:
                    if isinstance(symbol, Nonterminal) and symbol in self.nonterminal_to_idx:
                        rhs_indices.append(self.nonterminal_to_idx[symbol])
                    elif str(symbol) in self.terminal_to_idx:
                        rhs_indices.append(len(self.nonterminals) + self.terminal_to_idx[str(symbol)])

                if len(rhs_indices) == 2:
                    rules[lhs_idx, rhs_indices[0], rhs_indices[1]] = 0.0

        # Process terminal rules: A -> 'token'
        # These get encoded in the terms tensor during parsing
        # For now, we'll set uniform probabilities
        for i in range(N):
            for production in self.terminal_rules:
                lhs = production.lhs()
                rhs_token = str(production.rhs()[0])

                if lhs in self.nonterminal_to_idx and rhs_token in self.terminal_to_idx:
                    # This is a simplification - in practice, terms would be set
                    # based on neural network predictions for each position
                    terms[i, self.terminal_to_idx[rhs_token]] = 0.0

        return terms, rules, root

    def create_torch_struct_cfg(self, sequence_length: int,
                              device: Optional[torch.device] = None) -> torch_struct.SentCFG:
        """
        Create a torch-struct SentCFG distribution from NLTK CFG.

        Args:
            sequence_length: Length of sequences to parse
            device: Device to place tensors on

        Returns:
            torch_struct.SentCFG distribution
        """
        terms, rules, root = self.to_torch_struct_tensors(sequence_length, device)
        # Add batch dimension for torch-struct compatibility
        terms = terms.unsqueeze(0)  # (1, N, T)
        rules = rules.unsqueeze(0)  # (1, NT, NT+T, NT+T)
        root = root.unsqueeze(0)    # (1, NT)
        return torch_struct.SentCFG((terms, rules, root))

    def get_symbol_mappings(self) -> Dict[str, Union[Dict, List]]:
        """Get symbol to index mappings for external use."""
        return {
            'terminals': self.terminal_to_idx,
            'nonterminals': self.nonterminal_to_idx,
            'combined': self.symbol_to_idx,
            'terminal_list': self.terminals,
            'nonterminal_list': self.nonterminals
        }


def get_torch_struct_converter(grammar: Optional[CFG] = None) -> TorchStructCFGConverter:
    """Get a converter for NLTK CFG to torch-struct format."""
    if grammar is None:
        grammar = get_cfg()
    return TorchStructCFGConverter(grammar)


class BatchedStructuredParser:
    """
    Batched parsing interface using torch-struct for efficient GPU parsing.
    """

    def __init__(self, converter: TorchStructCFGConverter):
        self.converter = converter
        self.symbol_mappings = converter.get_symbol_mappings()

    def parse_batch(self,
                   batch_sequences: List[List[str]],
                   neural_params: Optional[Dict[str, torch.Tensor]] = None,
                   device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Parse a batch of token sequences using torch-struct.

        Args:
            batch_sequences: List of token sequences to parse
            neural_params: Optional neural parameters for rule/terminal scores
            device: Device to run computation on

        Returns:
            Dictionary with parsing results including marginals, partition function, etc.
        """
        if device is None:
            device = torch.device('cpu')

        batch_size = len(batch_sequences)
        max_length = max(len(seq) for seq in batch_sequences) if batch_sequences else 1

        # Pad sequences to same length
        padded_sequences = []
        lengths = []

        for seq in batch_sequences:
            lengths.append(len(seq))
            padded_seq = seq + ['<PAD>'] * (max_length - len(seq))
            padded_sequences.append(padded_seq)

        lengths_tensor = torch.tensor(lengths, device=device)

        # Create batch of CFG distributions
        batch_results = []

        for i in range(batch_size):
            seq_length = lengths[i]

            if neural_params is not None:
                # Use neural parameters if provided
                terms = neural_params.get('terms', None)
                rules = neural_params.get('rules', None)
                root = neural_params.get('root', None)

                if terms is not None and i < terms.size(0):
                    batch_terms = terms[i:i+1, :seq_length]
                else:
                    batch_terms, _, _ = self.converter.to_torch_struct_tensors(seq_length, device)
                    batch_terms = batch_terms.unsqueeze(0)

                if rules is not None and i < rules.size(0):
                    batch_rules = rules[i:i+1]
                else:
                    _, batch_rules, _ = self.converter.to_torch_struct_tensors(seq_length, device)
                    batch_rules = batch_rules.unsqueeze(0)

                if root is not None and i < root.size(0):
                    batch_root = root[i:i+1]
                else:
                    _, _, batch_root = self.converter.to_torch_struct_tensors(seq_length, device)
                    batch_root = batch_root.unsqueeze(0)
            else:
                # Use default grammar parameters
                batch_terms, batch_rules, batch_root = self.converter.to_torch_struct_tensors(seq_length, device)
                batch_terms = batch_terms.unsqueeze(0)
                batch_rules = batch_rules.unsqueeze(0)
                batch_root = batch_root.unsqueeze(0)

            # Create torch-struct CFG distribution
            # SentCFG expects batched inputs: terms (B, N, T), rules (B, NT, NT+T, NT+T), root (B, NT)
            # But since we're processing one at a time, we need to add batch dimension
            batch_terms_3d = batch_terms.squeeze(0).unsqueeze(0)  # (1, N, T)
            batch_rules_3d = batch_rules.squeeze(0).unsqueeze(0)  # (1, NT, NT+T, NT+T)
            batch_root_2d = batch_root.squeeze(0).unsqueeze(0)    # (1, NT)

            cfg_dist = torch_struct.SentCFG((batch_terms_3d.squeeze(0), batch_rules_3d.squeeze(0), batch_root_2d.squeeze(0)))

            # Compute properties safely
            try:
                marginals = cfg_dist.marginals
                partition = cfg_dist.partition
                entropy = cfg_dist.entropy
                argmax = cfg_dist.argmax
            except Exception as e:
                print(f"Warning: Failed to compute CFG properties: {e}")
                # Provide fallback values
                marginals = torch.zeros_like(batch_terms_3d.squeeze(0))
                partition = torch.tensor(0.0, device=device)
                entropy = torch.tensor(0.0, device=device)
                argmax = torch.zeros(seq_length, seq_length, len(self.converter.nonterminals), device=device)

            batch_results.append({
                'distribution': cfg_dist,
                'marginals': marginals,
                'partition': partition,
                'entropy': entropy,
                'argmax': argmax
            })

        # Combine batch results
        if batch_results:
            combined_results = {
                'batch_marginals': [r['marginals'] for r in batch_results],
                'batch_partitions': torch.stack([r['partition'] for r in batch_results]),
                'batch_entropies': torch.stack([r['entropy'] for r in batch_results]),
                'batch_argmax': [r['argmax'] for r in batch_results],
                'lengths': lengths_tensor
            }
        else:
            # Handle empty batch case
            combined_results = {
                'batch_marginals': [],
                'batch_partitions': torch.empty(0, device=device),
                'batch_entropies': torch.empty(0, device=device),
                'batch_argmax': [],
                'lengths': lengths_tensor
            }

        return combined_results

    def structured_attention(self,
                           hidden_states: torch.Tensor,
                           sequence_lengths: List[int],
                           device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute structured attention weights using torch-struct marginals.

        Args:
            hidden_states: Hidden states from transformer (batch_size, max_length, hidden_dim)
            sequence_lengths: Actual lengths for each sequence
            device: Device to run computation on

        Returns:
            Structured attention weights (batch_size, max_length, max_length)
        """
        if device is None:
            device = hidden_states.device

        batch_size, max_length, hidden_dim = hidden_states.shape

        # Placeholder for structured attention - in practice this would use
        # neural parameterization of the CFG rules based on hidden states
        attention_weights = torch.zeros(batch_size, max_length, max_length, device=device)

        for i, length in enumerate(sequence_lengths):
            # Create neural parameters from hidden states
            # This is a simplified version - full implementation would use
            # learned projections from hidden_states to rule/terminal scores

            # For now, use uniform attention within actual sequence length
            attention_weights[i, :length, :length] = 1.0 / length

        return attention_weights


def get_batched_parser(grammar: Optional[CFG] = None) -> BatchedStructuredParser:
    """Get a batched structured parser using torch-struct."""
    converter = get_torch_struct_converter(grammar)
    return BatchedStructuredParser(converter)


__all__ = [
    "get_cfg",
    "get_token_patterns",
    "realize_program",
    "sample_programs",
    "SUPPORTED_FUNCTIONS",
    "SUPPORTED_METHODS",
    "TorchStructCFGConverter",
    "get_torch_struct_converter",
    "BatchedStructuredParser",
    "get_batched_parser",
]


