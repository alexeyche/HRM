from __future__ import annotations

import re
from typing import List, Dict, Set
from dataset.grammar import get_cfg, get_token_patterns
from nltk import CFG


def extract_tokens_from_grammar(grammar: CFG) -> Dict[str, Set[str]]:
    """Extract all terminal tokens from the grammar."""
    tokens = {}

    # Get all productions
    for production in grammar.productions():
        lhs = str(production.lhs())
        rhs = [str(symbol) for symbol in production.rhs()]

        # Only process terminals (non-nonterminals)
        if all(not symbol.startswith('\'') for symbol in rhs):
            continue

        # Extract quoted terminals
        terminals = []
        for symbol in rhs:
            if symbol.startswith('\'') and symbol.endswith('\''):
                terminals.append(symbol[1:-1])

        if terminals and lhs not in tokens:
            tokens[lhs] = set()
        for terminal in terminals:
            tokens[lhs].add(terminal)

    return tokens


def create_token_patterns() -> Dict[str, str]:
    """Create regex patterns for token matching based on grammar token definitions."""

    # Get token definitions from grammar
    token_patterns = get_token_patterns()

    # Create regex patterns from token definitions
    patterns = {}

    # Handle multi-character operators separately
    binary_cmp_tokens = token_patterns.get('BINARY_CMP', [])
    if '<=' in binary_cmp_tokens:
        patterns['LTE'] = re.escape('<=')
    if '>=' in binary_cmp_tokens:
        patterns['GTE'] = re.escape('>=')
    if '==' in binary_cmp_tokens:
        patterns['EQ'] = re.escape('==')
    if '!=' in binary_cmp_tokens:
        patterns['NEQ'] = re.escape('!=')

    # Handle other multi-character tokens
    for token_type, tokens in token_patterns.items():
        if token_type == 'BINARY_CMP':
            # Already handled above for multi-char operators
            single_char_cmps = [t for t in tokens if len(t) == 1]
            if '<' in single_char_cmps:
                patterns['LT'] = re.escape('<')
            if '>' in single_char_cmps:
                patterns['GT'] = re.escape('>')
        elif token_type in ['VARIABLE', 'DIGIT']:
            # Handle these separately below
            continue
        else:
            # For tokens that should be single patterns
            if tokens:
                # Create regex pattern that matches any of the tokens
                escaped_tokens = [re.escape(token) for token in tokens]
                if len(escaped_tokens) == 1:
                    patterns[token_type] = escaped_tokens[0]
                else:
                    patterns[token_type] = '|'.join(escaped_tokens)

    # Handle variables (single letters a-z)
    variables = token_patterns.get('VARIABLE', [])
    if variables:
        patterns['VARIABLE'] = r'[a-z]'

    # Handle digits (0-21)
    digits = token_patterns.get('DIGIT', [])
    if digits:
        # Create a pattern that matches any of the allowed digits
        escaped_digits = [re.escape(digit) for digit in digits]
        patterns['DIGIT'] = '|'.join(escaped_digits)

    return patterns


def tokenize_code(code: str) -> List[str]:
    """Tokenize Python code according to the grammar rules."""

    # Remove leading/trailing whitespace and normalize newlines
    code = code.strip()
    code = code.replace('\r\n', '\n').replace('\r', '\n')

    # Add newline at end if not present
    if not code.endswith('\n'):
        code += '\n'

    tokens = []
    patterns = create_token_patterns()

    # Define priority order - keywords and multi-char operators first
    priority_order = [
        # Multi-character operators first
        'LTE', 'GTE', 'EQ', 'NEQ',
        # Keywords (longest first)
        'PROGRAM_NAME', 'CONTINUE', 'RETURN',
        'WHILE', 'BREAK', 'RANGE',
        'DEF', 'FOR', 'AND', 'NOT', 'ELSE', 'OR', 'IF', 'IN',
        # Special tokens
        'NEWLINE', 'INDENT', 'DEDENT',
        # Single character operators and punctuation
        'LPAREN', 'RPAREN', 'COMMA', 'COLON', 'EQUALS',
        'ADDOP', 'MULOP', 'LT', 'GT',
        # Variables and digits last (most general)
        'VARIABLE', 'DIGIT'
    ]

    # Sort patterns by priority order
    pattern_list = []
    for token_type in priority_order:
        if token_type in patterns:
            pattern_list.append((patterns[token_type], token_type))

    # Compile all patterns
    compiled_patterns = [(re.compile(pattern), token_type) for pattern, token_type in pattern_list]

    i = 0
    while i < len(code):
        # Skip whitespace
        if code[i].isspace():
            i += 1
            continue

        # Try to match a token with the highest priority patterns first
        matched = False
        for regex, token_type in compiled_patterns:
            match = regex.match(code, i)
            if match:
                token = match.group(0)
                tokens.append(token)
                i = match.end()
                matched = True
                break

        if not matched:
            # No token matched - raise error
            raise ValueError(f"Unexpected character '{code[i]}' at position {i}")

    return tokens


def get_token_mapping() -> Dict[str, str]:
    """Get mapping from token names to their string representations."""
    grammar = get_cfg()
    tokens = extract_tokens_from_grammar(grammar)

    mapping = {}
    for token_type, token_set in tokens.items():
        for token in token_set:
            mapping[token] = token_type

    return mapping


# For backward compatibility, also provide a simple tokenize function
def tokenize(text: str) -> List[str]:
    """Simple wrapper function for tokenizing text."""
    return tokenize_code(text)


if __name__ == "__main__":
    # Test the tokenizer
    test_code = """
    def program(a, b):
        if a < b:
            return a
        else:
            return b
    """

    tokens = tokenize_code(test_code)
    print("Tokens:", tokens)
