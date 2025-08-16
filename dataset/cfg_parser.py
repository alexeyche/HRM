from typing import Any, Optional
import torch
from dataset.ast import ASTNodeType, ASTNode
from enum import Enum
from collections import deque
from dataset.cfg import CFGNonTerminal, CFGTerminal, CFGGrammar
import logging

log = logging.getLogger(__name__)

class CFGParserState(Enum):
    EXPECTING_FUNCTION_DEF = "expecting_function_def"
    EXPECTING_FUNCTION_NAME = "expecting_function_name"
    EXPECTING_PARAMS = "expecting_params"
    EXPECTING_COLON = "expecting_colon"
    EXPECTING_BODY = "expecting_body"
    EXPECTING_STATEMENT = "expecting_statement"


def resolve_symbol(symbol: str) -> CFGTerminal | CFGNonTerminal:
    if terminal := CFGTerminal.find(symbol):
        return terminal
    elif non_terminal := CFGNonTerminal.find(symbol):
        return non_terminal
    else:
        if symbol.isalnum():
            return CFGTerminal.IDENTIFIER_LITERAL
        else:
            raise ValueError(f"Unknown symbol: {symbol}")


class CFGParser:
    history: list[tuple[CFGNonTerminal, list[str]]] = []
    stack: deque[CFGTerminal | CFGNonTerminal | str] = deque()
    grammar: CFGGrammar = CFGGrammar()
    # Track the current production being parsed
    current_production: Optional[tuple[CFGNonTerminal, list[str], int]] = None

    def __init__(self):
        self.reset()

    def consume_token(self, token: ASTNode):
        pass

    def reset(self):
        """Reset parser to initial state with start symbol"""
        self.history.clear()
        self.stack.clear()
        self.current_production = None
        # Start with the start symbol of the grammar
        self.stack.append(CFGNonTerminal.PROGRAM)

    def _compute_first_set(self, non_terminal: CFGNonTerminal, visited: Optional[set] = None) -> set[CFGTerminal | str]:
        """Compute FIRST set for a non-terminal (terminals that can start strings derived from it)"""
        if visited is None:
            visited = set()

        if non_terminal in visited:
            return set()  # Avoid infinite recursion

        visited.add(non_terminal)
        first_set = set()

        productions = self.grammar.get_productions(non_terminal)

        for production in productions:
            if not production:  # Empty production
                continue

            # Check if first symbol can start with epsilon (empty string)
            first_symbol = production[0]
            first_symbol_resolved = resolve_symbol(first_symbol)

            if isinstance(first_symbol_resolved, CFGTerminal):
                first_set.add(first_symbol_resolved)
            elif isinstance(first_symbol_resolved, CFGNonTerminal):
                # Recursively compute FIRST set for non-terminal
                first_set.update(self._compute_first_set(first_symbol_resolved, visited.copy()))
            elif isinstance(first_symbol_resolved, str):
                first_set.add(first_symbol_resolved)

        return first_set

    def valid_next_tokens(self) -> set[CFGTerminal | str]:
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

        # Case 2: nonterminal → compute FIRST set
        elif isinstance(sym, CFGNonTerminal):
            return self._compute_first_set(sym)

        # Case 3: string literal
        elif isinstance(sym, str):
            return {sym}

        else:
            raise TypeError(f"Unexpected symbol on stack: {sym}")

    def advance(self, token: CFGTerminal | str):
        """
        Consume the given token, updating stack state.
        Raises ValueError if token is not valid here.
        """
        if not self.stack:
            raise ValueError("Advance called on empty stack")

        top = self.stack[-1]

        # Case 1: terminal
        if isinstance(top, CFGTerminal):
            if isinstance(token, str):
                # Special case: if the terminal is IDENTIFIER_LITERAL, any valid Python identifier should match
                if top == CFGTerminal.IDENTIFIER_LITERAL:
                    if token.isidentifier():
                        self.stack.pop()
                        # Continue with the current production if we have one
                        if self.current_production:
                            self._continue_production()
                        return
                    else:
                        raise ValueError(f"Expected valid identifier, got {token}")
                # Compare terminal value with string token
                elif top.value == token:
                    self.stack.pop()
                    # Continue with the current production if we have one
                    if self.current_production:
                        self._continue_production()
                    return
                else:
                    raise ValueError(f"Expected {top.value}, got {token}")
            else:
                # Compare terminal with terminal
                if top == token:
                    self.stack.pop()
                    # Continue with the current production if we have one
                    if self.current_production:
                        self._continue_production()
                    return
                else:
                    raise ValueError(f"Expected {top}, got {token}")

        # Case 2: nonterminal
        elif isinstance(top, CFGNonTerminal):
            # Find a production whose FIRST set contains the token
            productions = self.grammar.get_productions(top)

            log.debug(f"Productions for {top}: {productions}")

            for rhs in productions:
                log.debug(f"Checking production {rhs} for {top}")
                # Check if this production can start with the token
                if self._can_start_with(rhs, token):
                    log.debug(f"Production {rhs} matches token {token}")
                    # Record rule choice (for AST reconstruction)
                    self.history.append((top, rhs))
                    # Set current production for tracking
                    self.current_production = (top, rhs, 0)
                    # Push RHS in reverse order (leftmost derivation)
                    for sym in reversed(rhs):
                        resolved_sym = resolve_symbol(sym)
                        self.stack.append(resolved_sym)

                    # Now try to advance again with the same token to handle immediate expansion
                    # This handles the case where we expand a non-terminal and immediately need to match
                    return self.advance(token)
                else:
                    log.debug(f"Production {rhs} does not match token {token}")

            log.debug(f"No production found for {top} that can start with {token}")
            raise ValueError(f"Token {token} not valid for nonterminal {top}")

        # Case 3: string literal
        elif isinstance(top, str):
            if isinstance(token, str):
                if top == token:
                    self.stack.pop()
                    # Continue with the current production if we have one
                    if self.current_production:
                        self._continue_production()
                    return
                else:
                    raise ValueError(f"Expected {top}, got {token}")
            else:
                # Token is CFGTerminal, compare with string value
                if top == token.value:
                    self.stack.pop()
                    # Continue with the current production if we have one
                    if self.current_production:
                        self._continue_production()
                    return
                else:
                    raise ValueError(f"Expected {top}, got {token.value}")

    def _continue_production(self):
        """Continue with the current production after consuming a terminal"""
        if not self.current_production:
            return

        non_terminal, production, position = self.current_production
        position += 1

        # If we've consumed all symbols in the production, we're done
        if position >= len(production):
            self.current_production = None
            return

        # Update position
        self.current_production = (non_terminal, production, position)

        # Get the next symbol in the production
        next_symbol = production[position]
        next_symbol_resolved = resolve_symbol(next_symbol)

        # Push the next symbol onto the stack
        self.stack.append(next_symbol_resolved)

    def _can_start_with(self, production: list[str], token: CFGTerminal | str, visited: Optional[set] = None) -> bool:
        """Check if a production can start with the given token"""
        if visited is None:
            visited = set()

        if not production:
            return False

        first_symbol = production[0]
        first_symbol_resolved = resolve_symbol(first_symbol)

        log.debug(f"Checking if {first_symbol} can start with {token}, resolved into {first_symbol_resolved}")

        # If first symbol is a terminal, check if it matches
        if isinstance(first_symbol_resolved, CFGTerminal):
            log.debug(f"First symbol is CFGTerminal: {first_symbol_resolved}")

            if isinstance(token, str):
                # Special case: if the terminal is IDENTIFIER_LITERAL, any valid Python identifier should match
                if first_symbol_resolved == CFGTerminal.IDENTIFIER_LITERAL:
                    log.debug(f"Checking IDENTIFIER_LITERAL against string token: {token}")
                    log.debug(f"Token is valid identifier: {token.isidentifier()}")
                    return token.isidentifier()
                log.debug(f"Comparing {first_symbol_resolved.value} == {token} = {first_symbol_resolved.value == token}")
                return first_symbol_resolved.value == token

            else:
                log.debug(f"Comparing {first_symbol_resolved} == {token} = {first_symbol_resolved == token}")
                return first_symbol_resolved == token

        elif isinstance(first_symbol_resolved, CFGNonTerminal):
            log.debug(f"First symbol is CFGNonTerminal: {first_symbol_resolved}")
            # Check if this non-terminal can start with the token
            return self._non_terminal_can_start_with(first_symbol_resolved, token, visited)

        elif isinstance(first_symbol_resolved, str):
            log.debug(f"First symbol is str: {first_symbol_resolved}")
            if isinstance(token, str):
                log.debug(f"Comparing {first_symbol_resolved} == {token} = {first_symbol_resolved == token}")
                return first_symbol_resolved == token
            else:
                log.debug(f"Comparing {first_symbol_resolved} == {token.value} = {first_symbol_resolved == token.value}")
                return first_symbol_resolved == token.value
        else:
            raise ValueError(f"Unexpected symbol: {first_symbol_resolved}")

    def _non_terminal_can_start_with(self, non_terminal: CFGNonTerminal, token: CFGTerminal | str, visited: Optional[set] = None) -> bool:
        """Check if a non-terminal can start with the given token"""
        if visited is None:
            visited = set()

        if non_terminal in visited:
            return False  # Avoid infinite recursion

        visited.add(non_terminal)
        productions = self.grammar.get_productions(non_terminal)

        for rhs in productions:
            if self._can_start_with(rhs, token, visited):
                return True

        return False

def validate_ast_syntax(ast_graph: dict[str, Any]) -> bool:
    """Validate the syntax of an AST graph"""
    return True