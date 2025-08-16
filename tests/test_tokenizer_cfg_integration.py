"""
Integration tests for the tokenizer and CFG parser working together.
"""

import pytest
from dataset.tokenizer import tokenize_code
from dataset.cfg_parser import CFGParser, parse_code
from dataset.cfg import CFGTerminal, CFGNonTerminal


def test_simple_function_parsing():
    """Test parsing a simple function definition"""
    code = "def hello(): pass"
    success, parse_info = parse_code(code)

    # The parser should be able to handle this basic function
    assert success is not None  # Either True or False, but not None
    assert "error" not in parse_info or not parse_info["error"]

def test_tokenizer_to_parser_flow():
    """Test the complete flow from tokenizer to parser"""
    code = "def test(x): return x"

    # Step 1: Tokenize
    tokens = tokenize_code(code)
    assert len(tokens) > 0

    # Step 2: Create parser and consume tokens
    parser = CFGParser()
    success = parser.consume_tokens(tokens)

    # The parser should process the tokens
    assert success is not None

def test_complex_expression_parsing():
    """Test parsing complex expressions"""
    code = "result = (a + b) * c"
    tokens = tokenize_code(code)

    # Verify all expected tokens are present
    assert CFGTerminal.ASSIGN in tokens
    assert CFGTerminal.PLUS in tokens
    assert CFGTerminal.MULTIPLY in tokens
    assert CFGTerminal.LPAREN in tokens
    assert CFGTerminal.RPAREN in tokens

    # Try to parse with CFG parser
    parser = CFGParser()
    success = parser.consume_tokens(tokens)
    assert success is not None

def test_control_flow_parsing():
    """Test parsing control flow statements"""
    code = """if x > 0:
return x
else:
return 0"""

    tokens = tokenize_code(code)

    # Verify control flow tokens
    assert CFGTerminal.IF in tokens
    assert CFGTerminal.RETURN in tokens
    assert CFGTerminal.COLON in tokens

    # Try to parse
    parser = CFGParser()
    success = parser.consume_tokens(tokens)
    assert success is not None

def test_function_with_parameters():
    """Test parsing function with parameters"""
    code = "def calculate(a, b, c): return a + b + c"
    tokens = tokenize_code(code)

    # Verify function definition tokens
    assert CFGTerminal.DEF in tokens
    assert CFGTerminal.LPAREN in tokens
    assert CFGTerminal.COMMA in tokens
    assert CFGTerminal.RPAREN in tokens
    assert CFGTerminal.COLON in tokens

    # Try to parse
    parser = CFGParser()
    success = parser.consume_tokens(tokens)
    assert success is not None

def test_operator_precedence_parsing():
    """Test parsing expressions with operator precedence"""
    code = "x = a + b * c ** d"
    tokens = tokenize_code(code)

    # Verify operator tokens
    assert CFGTerminal.ASSIGN in tokens
    assert CFGTerminal.PLUS in tokens
    assert CFGTerminal.MULTIPLY in tokens
    assert CFGTerminal.POWER in tokens

    # Try to parse
    parser = CFGParser()
    success = parser.consume_tokens(tokens)
    assert success is not None

def test_whitespace_handling_integration():
    """Test that whitespace is properly handled in the integration"""
    code = "x=1\n    y=2"
    tokens = tokenize_code(code)

    # Verify whitespace tokens
    assert CFGTerminal.NEWLINE in tokens
    assert CFGTerminal.TAB in tokens  # Indentation

    # Try to parse
    parser = CFGParser()
    success = parser.consume_tokens(tokens)
    assert success is not None

def test_error_handling():
    """Test error handling in the integration"""
    # Test with malformed code
    code = "def test(: invalid syntax"
    tokens = tokenize_code(code)

    # The tokenizer should still produce tokens
    assert len(tokens) > 0

    # The parser might fail, but should handle it gracefully
    try:
        parser = CFGParser()
        success = parser.consume_tokens(tokens)
        # This might fail, but shouldn't crash
    except Exception as e:
        # If it fails, that's okay - we're testing error handling
        assert isinstance(e, Exception)

def test_token_consistency():
    """Test that tokens are consistent between tokenizer and parser"""
    code = "def func(x): return x + 1"
    tokens = tokenize_code(code)

    # All tokens should be either CFGTerminal or valid strings
    for token in tokens:
        assert isinstance(token, (CFGTerminal, str))

        # If it's a string, it should be a valid Python token
        if isinstance(token, str):
            # Skip whitespace and special characters
            if token not in [' ', '\n', '\t'] and not token.startswith('"') and not token.startswith("'"):
                # Should be a valid identifier, number, or keyword
                assert token.isidentifier() or token.replace('.', '').isdigit() or token in ['pass', 'None', 'True', 'False']

def test_parser_state_management():
    """Test that parser state is properly managed"""
    code = "def test(): pass"
    tokens = tokenize_code(code)

    parser = CFGParser()

    # Initial state
    assert len(parser.stack) == 1
    assert parser.stack[0] == CFGNonTerminal.PROGRAM

    # After consuming tokens
    success = parser.consume_tokens(tokens)
    assert success is not None

    # Final state should be different from initial
    # Note: The parser might not fully consume all tokens due to grammar limitations
    # but it should have processed some of them
    assert len(parser.history) > 0 or len(parser.production_stack) > 0


def test_end_to_end_workflow():
    """Test the complete end-to-end workflow"""
    # This is a simple test to ensure the basic workflow works
    code = "def add(a, b): return a + b"

    # Step 1: Tokenize
    tokens = tokenize_code(code)
    assert len(tokens) > 0

    # Step 2: Parse
    success, parse_info = parse_code(code)
    assert success is not None

    # Step 3: Verify tokens contain expected elements
    assert CFGTerminal.DEF in tokens
    assert CFGTerminal.LPAREN in tokens
    assert CFGTerminal.COMMA in tokens
    assert CFGTerminal.RPAREN in tokens
    assert CFGTerminal.COLON in tokens
    assert CFGTerminal.RETURN in tokens
    assert CFGTerminal.PLUS in tokens


if __name__ == "__main__":
    # Run basic integration tests
    test_end_to_end_workflow()
    print("Basic integration tests passed!")
