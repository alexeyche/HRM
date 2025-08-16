"""
Python code tokenizer for CFG parser.

This module provides functionality to tokenize Python code into a sequence
of tokens that can be consumed by the CFG parser.
"""

import re
from typing import List, Union, Tuple, Any
from enum import Enum
from dataset.cfg import CFGTerminal


class TokenType(Enum):
    """Types of tokens that can be produced by the tokenizer"""
    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    STRING = "string"
    NUMBER = "number"
    OPERATOR = "operator"
    PUNCTUATION = "punctuation"
    WHITESPACE = "whitespace"
    NEWLINE = "newline"
    INDENT = "indent"
    DEDENT = "dedent"
    COMMENT = "comment"
    EOF = "eof"


class Token:
    """Represents a single token in the source code"""

    def __init__(self, type: TokenType, value: Union[CFGTerminal, str], line: int = 0, column: int = 0):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type.value}, {repr(self.value)}, line={self.line}, col={self.column})"

    def __eq__(self, other):
        if isinstance(other, Token):
            return (self.type == other.type and
                   self.value == other.value and
                   self.line == other.line and
                   self.column == other.column)
        return False


class PythonTokenizer:
    """Tokenizes Python source code into a sequence of tokens"""

    # Python keywords that map to CFG terminals
    KEYWORDS = {
        'def': CFGTerminal.DEF,
        'if': CFGTerminal.IF,
        'elif': CFGTerminal.ELIF,
        'else': CFGTerminal.ELSE,
        'for': CFGTerminal.FOR,
        'while': CFGTerminal.WHILE,
        'in': CFGTerminal.IN,
        'return': CFGTerminal.RETURN,
        'and': CFGTerminal.AND,
        'or': CFGTerminal.OR,
        'not': CFGTerminal.NOT,
        'pass': 'pass',
        'None': 'None',
        'True': 'True',
        'False': 'False',
    }

    # Operators that map to CFG terminals
    OPERATORS = {
        '+': CFGTerminal.PLUS,
        '-': CFGTerminal.MINUS,
        '*': CFGTerminal.MULTIPLY,
        '/': CFGTerminal.DIVIDE,
        '%': CFGTerminal.MODULO,
        '**': CFGTerminal.POWER,
        '=': CFGTerminal.ASSIGN,
        '+=': CFGTerminal.PLUS_ASSIGN,
        '-=': CFGTerminal.MINUS_ASSIGN,
        '*=': CFGTerminal.MULT_ASSIGN,
        '/=': CFGTerminal.DIV_ASSIGN,
        '==': CFGTerminal.EQ,
        '!=': CFGTerminal.NE,
        '<': CFGTerminal.LT,
        '<=': CFGTerminal.LE,
        '>': CFGTerminal.GT,
        '>=': CFGTerminal.GE,
    }

    # Punctuation that maps to CFG terminals
    PUNCTUATION = {
        '(': CFGTerminal.LPAREN,
        ')': CFGTerminal.RPAREN,
        '[': CFGTerminal.LBRACKET,
        ']': CFGTerminal.RBRACKET,
        ':': CFGTerminal.COLON,
        ',': CFGTerminal.COMMA,
    }

    def __init__(self):
        self.source = ""
        self.tokens = []
        self.current_pos = 0
        self.line = 1
        self.column = 1
        self.indent_stack = [0]

    def tokenize(self, source: str) -> List[Union[CFGTerminal, str]]:
        """
        Tokenize Python source code into a list of tokens.

        Args:
            source: Python source code as string

        Returns:
            List of tokens (CFGTerminal values or strings) suitable for CFG parser
        """
        self.source = source
        self.tokens = []
        self.current_pos = 0
        self.line = 1
        self.column = 1
        self.indent_stack = [0]

        while self.current_pos < len(self.source):
            char = self.source[self.current_pos]

            if char.isspace():
                self._handle_whitespace()
            elif char == '#':
                self._handle_comment()
            elif char.isalpha() or char == '_':
                self._handle_identifier_or_keyword()
            elif char.isdigit():
                self._handle_number()
            elif char in '"\'':
                self._handle_string()
            elif char in '+-*/%=<>!&|':
                self._handle_operator()
            elif char in '()[]{}:,':
                self._handle_punctuation()
            else:
                # Unknown character, skip
                self.current_pos += 1
                self.column += 1

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))

        # Convert tokens to CFG parser format
        return self._convert_to_cfg_tokens()

    def _handle_whitespace(self):
        """Handle whitespace characters"""
        start_pos = self.current_pos
        start_column = self.column

        # Collect all consecutive whitespace
        while (self.current_pos < len(self.source) and
               self.source[self.current_pos].isspace()):
            char = self.source[self.current_pos]
            if char == '\n':
                # Handle newline
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.line += 1
                self.column = 1
                self.current_pos += 1

                # Handle indentation after newline
                if self.current_pos < len(self.source):
                    self._handle_indentation()
                break
            else:
                # Regular whitespace
                self.current_pos += 1
                self.column += 1

        # Add whitespace token if we collected any
        if self.current_pos > start_pos and self.source[self.current_pos - 1] != '\n':
            whitespace = self.source[start_pos:self.current_pos]
            if whitespace:
                self.tokens.append(Token(TokenType.WHITESPACE, whitespace, self.line, start_column))

    def _handle_indentation(self):
        """Handle indentation after a newline"""
        start_pos = self.current_pos
        start_column = self.column

        # Count spaces/tabs for indentation
        indent_level = 0
        while (self.current_pos < len(self.source) and
               self.source[self.current_pos] in ' \t'):
            if self.source[self.current_pos] == '\t':
                indent_level += 8  # Treat tab as 8 spaces
            else:
                indent_level += 1
            self.current_pos += 1
            self.column += 1

        if indent_level > 0:
            current_indent = self.indent_stack[-1]

            if indent_level > current_indent:
                # Indent
                self.tokens.append(Token(TokenType.INDENT, '\t', self.line, start_column))
                self.indent_stack.append(indent_level)
            elif indent_level < current_indent:
                # Dedent
                while self.indent_stack and self.indent_stack[-1] > indent_level:
                    self.indent_stack.pop()
                    self.tokens.append(Token(TokenType.DEDENT, '', self.line, start_column))

                if indent_level != self.indent_stack[-1]:
                    raise ValueError(f"Invalid indentation at line {self.line}")

    def _handle_comment(self):
        """Handle Python comments"""
        start_column = self.column

        # Skip the # character
        self.current_pos += 1
        self.column += 1

        # Collect comment text until newline or EOF
        while (self.current_pos < len(self.source) and
               self.source[self.current_pos] != '\n'):
            self.current_pos += 1
            self.column += 1

        # Add comment token
        comment_text = self.source[start_column-1:self.current_pos]
        self.tokens.append(Token(TokenType.COMMENT, comment_text, self.line, start_column))

    def _handle_identifier_or_keyword(self):
        """Handle identifiers and keywords"""
        start_pos = self.current_pos
        start_column = self.column

        # Collect alphanumeric characters and underscores
        while (self.current_pos < len(self.source) and
               (self.source[self.current_pos].isalnum() or
                self.source[self.current_pos] == '_')):
            self.current_pos += 1
            self.column += 1

        identifier = self.source[start_pos:self.current_pos]

        # Check if it's a keyword
        if identifier in self.KEYWORDS:
            keyword_value = self.KEYWORDS[identifier]
            if isinstance(keyword_value, CFGTerminal):
                self.tokens.append(Token(TokenType.KEYWORD, keyword_value, self.line, start_column))
            else:
                # String keyword (like 'pass')
                self.tokens.append(Token(TokenType.KEYWORD, keyword_value, self.line, start_column))
        else:
            # Regular identifier
            self.tokens.append(Token(TokenType.IDENTIFIER, identifier, self.line, start_column))

    def _handle_number(self):
        """Handle numeric literals"""
        start_pos = self.current_pos
        start_column = self.column

        # Collect digits
        while (self.current_pos < len(self.source) and
               self.source[self.current_pos].isdigit()):
            self.current_pos += 1
            self.column += 1

        # Handle decimal point
        if (self.current_pos < len(self.source) and
            self.source[self.current_pos] == '.'):
            self.current_pos += 1
            self.column += 1

            # Collect digits after decimal point
            while (self.current_pos < len(self.source) and
                   self.source[self.current_pos].isdigit()):
                self.current_pos += 1
                self.column += 1

        # Handle scientific notation
        if (self.current_pos < len(self.source) and
            self.source[self.current_pos] in 'eE'):
            self.current_pos += 1
            self.column += 1

            # Handle optional sign
            if (self.current_pos < len(self.source) and
                self.source[self.current_pos] in '+-'):
                self.current_pos += 1
                self.column += 1

            # Collect digits in exponent
            while (self.current_pos < len(self.source) and
                   self.source[self.current_pos].isdigit()):
                self.current_pos += 1
                self.column += 1

        number = self.source[start_pos:self.current_pos]
        self.tokens.append(Token(TokenType.NUMBER, number, self.line, start_column))

    def _handle_string(self):
        """Handle string literals"""
        quote_char = self.source[self.current_pos]
        start_column = self.column

        # Skip the opening quote
        self.current_pos += 1
        self.column += 1

        # Collect string content
        while (self.current_pos < len(self.source) and
               self.source[self.current_pos] != quote_char):
            if self.source[self.current_pos] == '\\':
                # Handle escape sequences
                self.current_pos += 2
                self.column += 2
            else:
                self.current_pos += 1
                self.column += 1

        # Skip the closing quote
        if self.current_pos < len(self.source):
            self.current_pos += 1
            self.column += 1

        string_content = self.source[start_column:self.current_pos]
        self.tokens.append(Token(TokenType.STRING, string_content, self.line, start_column))

    def _handle_operator(self):
        """Handle operators"""
        start_pos = self.current_pos
        start_column = self.column

        # Try to match multi-character operators first
        if (self.current_pos + 1 < len(self.source)):
            two_char_op = self.source[self.current_pos:self.current_pos + 2]
            if two_char_op in self.OPERATORS:
                self.current_pos += 2
                self.column += 2
                operator = two_char_op
            else:
                # Single character operator
                operator = self.source[self.current_pos]
                self.current_pos += 1
                self.column += 1
        else:
            # Single character operator at end of source
            operator = self.source[self.current_pos]
            self.current_pos += 1
            self.column += 1

        if operator in self.OPERATORS:
            self.tokens.append(Token(TokenType.OPERATOR, self.OPERATORS[operator], self.line, start_column))
        else:
            # Unknown operator, treat as string
            self.tokens.append(Token(TokenType.OPERATOR, operator, self.line, start_column))

    def _handle_punctuation(self):
        """Handle punctuation characters"""
        char = self.source[self.current_pos]
        start_column = self.column

        if char in self.PUNCTUATION:
            self.tokens.append(Token(TokenType.PUNCTUATION, self.PUNCTUATION[char], self.line, start_column))
        else:
            # Unknown punctuation, treat as string
            self.tokens.append(Token(TokenType.PUNCTUATION, char, self.line, start_column))

        self.current_pos += 1
        self.column += 1

    def _convert_to_cfg_tokens(self) -> List[Union[CFGTerminal, str]]:
        """Convert internal tokens to CFG parser format"""
        cfg_tokens = []

        for token in self.tokens:
            if token.type == TokenType.EOF:
                break
            elif token.type == TokenType.COMMENT:
                # Skip comments
                continue
            elif token.type == TokenType.WHITESPACE:
                # Add whitespace as string
                cfg_tokens.append(token.value)
            elif token.type == TokenType.NEWLINE:
                # Add newline as string
                cfg_tokens.append('\n')
            elif token.type == TokenType.INDENT:
                # Add indent as tab character
                cfg_tokens.append('\t')
            elif token.type == TokenType.DEDENT:
                # Skip dedent tokens for now
                continue
            elif token.type == TokenType.KEYWORD:
                # Add keyword value (CFGTerminal or string)
                cfg_tokens.append(token.value)
            elif token.type == TokenType.IDENTIFIER:
                # Add identifier as string
                cfg_tokens.append(token.value)
            elif token.type == TokenType.STRING:
                # Add string as string
                cfg_tokens.append(token.value)
            elif token.type == TokenType.NUMBER:
                # Add number as string
                cfg_tokens.append(token.value)
            elif token.type == TokenType.OPERATOR:
                # Add operator value (CFGTerminal or string)
                cfg_tokens.append(token.value)
            elif token.type == TokenType.PUNCTUATION:
                # Add punctuation value (CFGTerminal or string)
                cfg_tokens.append(token.value)

        return cfg_tokens


def tokenize_python_code(code: str) -> List[Union[CFGTerminal, str]]:
    """
    Convenience function to tokenize Python code.

    Args:
        code: Python source code as string

    Returns:
        List of tokens suitable for CFG parser
    """
    tokenizer = PythonTokenizer()
    return tokenizer.tokenize(code)


if __name__ == "__main__":
    # Test the tokenizer
    test_code = '''
def function_name():
    pass
'''

    tokens = tokenize_python_code(test_code)
    print("Tokens:")
    for i, token in enumerate(tokens):
        print(f"{i}: {token}")
