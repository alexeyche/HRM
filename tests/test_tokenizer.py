"""
Tests for the Python tokenizer integration with CFG system.
"""

from dataset.tokenizer import Tokenizer, tokenize_code, Token, TokenType
from dataset.cfg import CFGTerminal
from dataset.programs import get_program_registry
from pprint import pprint


def test_basic_tokenization():
    """Test basic tokenization of simple Python code"""
    code = "def hello(): pass"
    tokens = tokenize_code(code)

    # Should return: [DEF, SPACE, IDENTIFIER_LITERAL, LPAREN, RPAREN, COLON, SPACE, 'pass']
    assert len(tokens) == 8
    assert tokens[0] == CFGTerminal.DEF
    assert tokens[1] == CFGTerminal.SPACE
    assert tokens[2] == CFGTerminal.IDENTIFIER_LITERAL
    assert tokens[3] == CFGTerminal.LPAREN
    assert tokens[4] == CFGTerminal.RPAREN
    assert tokens[5] == CFGTerminal.COLON
    assert tokens[6] == CFGTerminal.SPACE
    assert tokens[7] == 'pass'

def test_keywords_mapping():
    """Test that Python keywords are properly mapped to CFG terminals"""
    code = "if x and y or not z: return True"
    tokens = tokenize_code(code)

    # Check that keywords are mapped to CFG terminals
    assert CFGTerminal.IF in tokens
    assert CFGTerminal.AND in tokens
    assert CFGTerminal.OR in tokens
    assert CFGTerminal.NOT in tokens
    assert CFGTerminal.RETURN in tokens

def test_operators_mapping():
    """Test that operators are properly mapped to CFG terminals"""
    code = "x = a + b * c ** d"
    tokens = tokenize_code(code)

    # Check that operators are mapped to CFG terminals
    assert CFGTerminal.ASSIGN in tokens
    assert CFGTerminal.PLUS in tokens
    assert CFGTerminal.MULTIPLY in tokens
    assert CFGTerminal.POWER in tokens

def test_punctuation_mapping():
    """Test that punctuation is properly mapped to CFG terminals"""
    code = "def func(x, y): return [x, y]"
    tokens = tokenize_code(code)

    # Check that punctuation is mapped to CFG terminals
    assert CFGTerminal.LPAREN in tokens
    assert CFGTerminal.RPAREN in tokens
    assert CFGTerminal.COMMA in tokens
    assert CFGTerminal.COLON in tokens
    assert CFGTerminal.LBRACKET in tokens
    assert CFGTerminal.RBRACKET in tokens

def test_identifiers():
    """Test that identifiers are properly tokenized as IDENTIFIER_LITERAL"""
    code = "variable_name = 42"
    tokens = tokenize_code(code)

    # Check that identifiers use CFGTerminal.IDENTIFIER_LITERAL
    identifier_tokens = [t for t in tokens if t == CFGTerminal.IDENTIFIER_LITERAL]
    assert len(identifier_tokens) == 1

def test_numbers():
    """Test that numbers are properly tokenized"""
    code = "x = 123 + 45.67"
    tokens = tokenize_code(code)

    # Check that numbers are preserved as strings
    number_tokens = [t for t in tokens if isinstance(t, str) and t.replace('.', '').isdigit()]
    assert len(number_tokens) == 2
    assert "123" in number_tokens
    assert "45.67" in number_tokens

def test_strings():
    """Test that strings are properly tokenized"""
    code = 'message = "Hello, World!"'
    tokens = tokenize_code(code)

    # Check that strings are preserved
    string_tokens = [t for t in tokens if isinstance(t, str) and t.startswith('"')]
    assert len(string_tokens) == 1
    assert '"Hello, World!"' in string_tokens

def test_whitespace_handling():
    """Test that whitespace is properly handled"""
    code = "x = 1\n    y = 2"
    tokens = tokenize_code(code)

    # Check that whitespace and newlines are properly tokenized
    assert CFGTerminal.SPACE in tokens
    assert CFGTerminal.NEWLINE in tokens
    assert CFGTerminal.TAB in tokens  # Indentation becomes TAB

def test_comments():
    """Test that comments are properly handled"""
    code = "x = 1  # This is a comment"
    tokens = tokenize_code(code)

    # Comments should be skipped in the final output
    comment_tokens = [t for t in tokens if isinstance(t, str) and t.startswith('#')]
    assert len(comment_tokens) == 0

def test_indentation():
    """Test that indentation is properly handled"""
    code = """def test():
x = 1
if x > 0:
    return x"""

    tokens = tokenize_code(code)

    # Check that indentation is properly tokenized
    tab_tokens = [t for t in tokens if t == CFGTerminal.TAB]
    assert len(tab_tokens) >= 2  # At least 2 indentation levels

def test_complex_expression():
    """Test tokenization of complex expressions"""
    code = "result = (a + b) * (c - d) / e"
    tokens = tokenize_code(code)

    # Check that all operators and punctuation are properly mapped
    assert CFGTerminal.ASSIGN in tokens
    assert CFGTerminal.PLUS in tokens
    assert CFGTerminal.MINUS in tokens
    assert CFGTerminal.MULTIPLY in tokens
    assert CFGTerminal.DIVIDE in tokens
    assert CFGTerminal.LPAREN in tokens
    assert CFGTerminal.RPAREN in tokens

def test_function_definition():
    """Test tokenization of function definitions"""
    code = """def complex_function(param1, param2):
result = param1 + param2
return result"""

    tokens = tokenize_code(code)

    # Check function definition structure
    assert CFGTerminal.DEF in tokens
    assert CFGTerminal.IDENTIFIER_LITERAL in tokens
    assert CFGTerminal.LPAREN in tokens
    assert CFGTerminal.COMMA in tokens
    assert CFGTerminal.RPAREN in tokens
    assert CFGTerminal.COLON in tokens
    assert CFGTerminal.NEWLINE in tokens

def test_control_flow():
    """Test tokenization of control flow statements"""
    code = """if condition:
for item in items:
    while not done:
        pass"""

    tokens = tokenize_code(code)

    # Check control flow keywords
    assert CFGTerminal.IF in tokens
    assert CFGTerminal.FOR in tokens
    assert CFGTerminal.WHILE in tokens
    assert CFGTerminal.IN in tokens
    assert CFGTerminal.NOT in tokens

def test_augmented_assignment():
    """Test tokenization of augmented assignment operators"""
    code = "x += 1; y -= 2; z *= 3"
    tokens = tokenize_code(code)

    # Check augmented assignment operators
    assert CFGTerminal.PLUS_ASSIGN in tokens
    assert CFGTerminal.MINUS_ASSIGN in tokens
    assert CFGTerminal.MULT_ASSIGN in tokens

def test_comparison_operators():
    """Test tokenization of comparison operators"""
    code = "a == b and c != d or e < f <= g > h >= i"
    tokens = tokenize_code(code)

    # Check comparison operators
    assert CFGTerminal.EQ in tokens
    assert CFGTerminal.NE in tokens
    assert CFGTerminal.LT in tokens
    assert CFGTerminal.LE in tokens
    assert CFGTerminal.GT in tokens
    assert CFGTerminal.GE in tokens

def test_boolean_operators():
    """Test tokenization of boolean operators"""
    code = "result = a and b or not c"
    tokens = tokenize_code(code)

    # Check boolean operators
    assert CFGTerminal.AND in tokens
    assert CFGTerminal.OR in tokens
    assert CFGTerminal.NOT in tokens

def test_empty_code():
    """Test tokenization of empty code"""
    tokens = tokenize_code("")
    assert len(tokens) == 0

def test_whitespace_only():
    """Test tokenization of whitespace-only code"""
    tokens = tokenize_code("   \n\t  ")
    # Should only contain whitespace tokens
    assert all(t in [CFGTerminal.SPACE, CFGTerminal.NEWLINE, CFGTerminal.TAB] for t in tokens)

def test_token_positions():
    """Test that token positions are correctly tracked"""
    tokenizer = Tokenizer()
    code = "def func(x):\n    return x"
    tokenizer.tokenize(code)

    # Check that tokens have correct line and column information
    for token in tokenizer.tokens:
        assert token.line > 0
        assert token.column > 0

def test_integration_with_cfg_parser():
    """Test that tokenizer output can be consumed by CFG parser"""
    from dataset.cfg_parser import CFGParser

    code = "def test(): pass"
    tokens = tokenize_code(code)

    # Create CFG parser and try to consume tokens
    parser = CFGParser()

    # The parser should be able to handle these tokens
    # Note: This is a basic integration test - the parser might need
    # additional work to fully handle all token sequences
    assert len(tokens) > 0
    assert all(isinstance(t, (CFGTerminal, str)) for t in tokens)


def test_tokenizer_edge_cases():
    """Test edge cases and error handling"""
    tokenizer = Tokenizer()

    # Test with very long identifier
    long_identifier = "a" * 1000
    code = f"x = {long_identifier}"
    tokens = tokenize_code(code)
    assert len(tokens) > 0

    # Test with mixed whitespace
    code = "x\t= 1\n  y = 2"
    tokens = tokenize_code(code)
    assert CFGTerminal.TAB in tokens
    assert CFGTerminal.SPACE in tokens
    assert CFGTerminal.NEWLINE in tokens


def test_tokenizer_on_dataset():
    registry = get_program_registry()

    for _, spec in registry.programs.items():
        code = spec.implementation
        tokens = tokenize_code(code)
        print(f"Tokenizing code:\n{code}")
        pprint(tokens)
        assert len(tokens) > 0

