from typing import Any
import torch
from dataset.ast import ASTNodeType, ASTNode
from enum import Enum
from collections import deque
from dataset.cfg import CFGNonTerminal, CFGTerminal, CFGGrammar

class CFGParserState(Enum):
    EXPECTING_FUNCTION_DEF = "expecting_function_def"
    EXPECTING_FUNCTION_NAME = "expecting_function_name"
    EXPECTING_PARAMS = "expecting_params"
    EXPECTING_COLON = "expecting_colon"
    EXPECTING_BODY = "expecting_body"
    EXPECTING_STATEMENT = "expecting_statement"

class CFGParser:
    history: list[tuple[CFGNonTerminal, list[str]]] = []
    stack: deque[CFGTerminal | CFGNonTerminal] = deque()
    grammar: CFGGrammar = CFGGrammar()

    def __init__(self):
        self.reset()


    def consume_token(self, token: ASTNode):
        pass

    def reset(self):
        """Reset parser to initial state with start symbol"""
        self.history.clear()
        self.stack.clear()
        # Start with the start symbol of the grammar
        self.stack.append(CFGNonTerminal.PROGRAM)

    def valid_next_tokens(self) -> set[CFGTerminal]:
        """
        Return the set of valid terminals that can come next,
        given the current stack and grammar.
        """
        if not self.stack:
            return set()

        sym = self.stack[-1]

        # Case 1: terminal → it must be matched next
        if isinstance(sym, CFGTerminal):
            return {sym}

        # Case 2: nonterminal → union of FIRST sets of all expansions
        elif isinstance(sym, CFGNonTerminal):
            tokens = set()
            for rhs in self.grammar.get_productions(sym):
                tokens |= set(rhs)
            return tokens

        else:
            raise TypeError(f"Unexpected symbol on stack: {sym}")


    def advance(self, token: CFGTerminal):
        """
        Consume the given token, updating stack state.
        Raises ValueError if token is not valid here.
        """
        if not self.stack:
            raise ValueError("Advance called on empty stack")

        top = self.stack[-1]

        # Case 1: terminal
        if isinstance(top, CFGTerminal):
            if top == token:
                self.stack.pop()
                return
            else:
                raise ValueError(f"Expected {top}, got {token}")

        # Case 2: nonterminal
        elif isinstance(top, CFGNonTerminal):
            # Find a production whose FIRST set contains the token
            productions = self.grammar.get_productions(top)

            for rhs in productions:
                # Check if this production can start with the token
                if self._can_start_with(rhs, token):
                    # Record rule choice (for AST reconstruction)
                    self.history.append((top, rhs))
                    # Push RHS in reverse order (leftmost derivation)
                    for sym in reversed(rhs):
                        if non_terminal := CFGNonTerminal.find(sym):
                            # Non-terminal
                            self.stack.append(non_terminal)
                        elif terminal := CFGTerminal.find(sym):
                            # Terminal (string literal)
                            self.stack.append(terminal)
                        else:
                            raise ValueError(f"Unexpected symbol: {sym}")
                    # Now retry with updated stack
                    return self.advance(token)

            raise ValueError(f"Token {token} not valid for nonterminal {top}")

    def _can_start_with(self, production: list[str], token: CFGTerminal) -> bool:
        """Check if a production can start with the given token"""
        if not production:
            return False

        first_symbol = production[0]

        # If first symbol is a terminal, check if it matches
        if not first_symbol.isupper():
            return first_symbol == token.value

        # If first symbol is a non-terminal, check if it can derive the token
        # This is a simplified check - you might need more sophisticated FIRST set computation
        return self._non_terminal_can_start_with(CFGNonTerminal(first_symbol), token)

    def _non_terminal_can_start_with(self, non_terminal: CFGNonTerminal, token: CFGTerminal) -> bool:
        """Check if a non-terminal can start with the given token"""
        productions = self.grammar.get_productions(non_terminal)

        for rhs in productions:
            if self._can_start_with(rhs, token):
                return True

        return False

def validate_ast_syntax(ast_graph: dict[str, Any]) -> bool:
    """Validate the syntax of an AST graph"""
    return True