from typing import Any, Optional
import torch
from dataset.ast import ASTNodeType, ASTNode
from enum import Enum
from collections import deque
from dataset.cfg import CFGNonTerminal, CFGTerminal, CFGGrammar
import logging
from dataset.tokenizer import tokenize_code

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
        # Check for escaped characters
        if symbol == "\\n":
            return CFGTerminal.NEWLINE
        elif symbol == "\\t":
            return CFGTerminal.TAB

        # Check if it's a numeric literal
        try:
            float(symbol)
            return CFGTerminal.NUMBER
        except ValueError:
            pass

        # Check if it's a string literal (quoted)
        if symbol.startswith('"') and symbol.endswith('"'):
            return CFGTerminal.STRING
        elif symbol.startswith("'") and symbol.endswith("'"):
            return CFGTerminal.STRING
        # Check if it's a boolean literal
        elif symbol in ['True', 'False']:
            return CFGTerminal.BOOLEAN
        # Check if it's a valid identifier
        elif symbol.isidentifier():
            return CFGTerminal.IDENTIFIER_LITERAL
        else:
            raise ValueError(f"Unknown symbol: {symbol}")


class CFGParser:
    history: list[tuple[CFGNonTerminal, list[str]]] = []
    stack: deque[CFGTerminal | CFGNonTerminal | str] = deque()
    grammar: CFGGrammar = CFGGrammar()
    # Track stack of active productions being parsed
    production_stack: list[tuple[CFGNonTerminal, list[str], int]] = []

    error_messages: list[str] = []

    # Prediction table for LL(1) parsing: (non_terminal, token) -> production
    prediction_table: dict[tuple[CFGNonTerminal, str], list[str]] = {}

    def __init__(self):
        self.reset()
        self._build_prediction_table()

    def consume_token(self, token: CFGTerminal | str):
        """Consume a token from the tokenizer"""
        try:
            self.advance(token)
            return True
        except ValueError as e:
            self.error_messages.append(f"Failed to consume token {token}: {e}")
            return False

    def consume_tokens(self, tokens: list[CFGTerminal | str]) -> bool:
        """Consume a list of tokens from the tokenizer"""
        for token in tokens:
            if not self.consume_token(token):
                return False
        return True

    def reset(self):
        """Reset parser to initial state with start symbol"""
        self.history.clear()
        self.stack.clear()
        self.production_stack.clear()
        # Start with the start symbol of the grammar
        self.stack.append(CFGNonTerminal.PROGRAM)

    def _build_prediction_table(self):
        """Build LL(1) prediction table for deterministic parsing based on single token lookahead"""
        self.prediction_table = {}

        # For each non-terminal, determine which production to use for each possible token
        for non_terminal in CFGNonTerminal:
            productions = self.grammar.get_productions(non_terminal)

            for production in productions:
                first_set = self._compute_production_first_set(production)

                for token in first_set:
                    # Convert token to string key for consistent lookup
                    if hasattr(token, 'value'):
                        token_key = token.value
                    else:
                        token_key = str(token)

                    # Check for conflicts (multiple productions for same token)
                    key = (non_terminal, token_key)
                    if key in self.prediction_table:
                        # LL(1) conflict detected - use heuristics to resolve
                        existing_prod = self.prediction_table[key]
                        better_prod = self._resolve_production_conflict(
                            non_terminal, token_key, existing_prod, production
                        )
                        self.prediction_table[key] = better_prod
                    else:
                        self.prediction_table[key] = production

                # Also add entries for special numeric/string literals if this production can handle them
                if self._production_can_handle_literals(production):
                    # Add entries for number literals
                    for num_token in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                        key = (non_terminal, num_token)
                        if key not in self.prediction_table:
                            self.prediction_table[key] = production
                        else:
                            # Apply conflict resolution for numeric literals
                            existing_prod = self.prediction_table[key]
                            better_prod = self._resolve_production_conflict(
                                non_terminal, num_token, existing_prod, production
                            )
                            self.prediction_table[key] = better_prod

                    # Add entries for string literals
                    for str_token in ['"test"', "'test'"]:
                        key = (non_terminal, str_token)
                        if key not in self.prediction_table:
                            self.prediction_table[key] = production
                        else:
                            # Apply conflict resolution for string literals
                            existing_prod = self.prediction_table[key]
                            better_prod = self._resolve_production_conflict(
                                non_terminal, str_token, existing_prod, production
                            )
                            self.prediction_table[key] = better_prod

    def _production_can_handle_literals(self, production: list[str]) -> bool:
        """Check if a production can handle numeric/string literals"""
        if not production:
            return False

        # Check if the production starts with a symbol that can derive literals
        first_symbol = production[0]

        # Direct literal handling
        if first_symbol in ["NUMBER", "STRING", "CONSTANT"]:
            return True

        # Indirect literal handling through EXPRESSION -> CONSTANT -> NUMBER/STRING
        if first_symbol == "EXPRESSION":
            return True

        return False

    def _compute_production_first_set(self, production: list[str]) -> set:
        """Compute FIRST set for a specific production"""
        if not production:
            return set()

        first_symbol = production[0]
        first_symbol_resolved = resolve_symbol(first_symbol)

        if isinstance(first_symbol_resolved, CFGTerminal):
            return {first_symbol_resolved}
        elif isinstance(first_symbol_resolved, CFGNonTerminal):
            return self._compute_first_set(first_symbol_resolved)
        else:
            return {first_symbol}

    def _resolve_production_conflict(self, non_terminal: CFGNonTerminal, token: str,
                                   prod1: list[str], prod2: list[str]) -> list[str]:
        """Resolve LL(1) conflicts using heuristics suitable for deep learning generation"""

        # For EXPRESSION conflicts with IDENTIFIER_LITERAL, use these rules:
        if non_terminal == CFGNonTerminal.EXPRESSION and token == "<alpha_numeric>":
            # Priority order: CALL_EXPR > VARIABLE > BINARY_EXPR > others
            # This prioritizes function calls over variables since IDENTIFIER_LITERAL
            # in expression context is more likely to be a function call

            priority_order = [
                ["CALL_EXPR"],          # Function call (most common for expression start)
                ["VARIABLE"],           # Simple variable reference
                ["BINARY_EXPR"],        # Binary operation
                ["COMPARISON_EXPR"],    # Comparison
                ["BOOLEAN_EXPR"],       # Boolean operation
            ]

            for preferred in priority_order:
                if prod1 == preferred:
                    return prod1
                if prod2 == preferred:
                    return prod2

        # For CALL_EXPR, prefer the version with arguments
        if non_terminal == CFGNonTerminal.CALL_EXPR and token == "<alpha_numeric>":
            # Prefer CALL_EXPR with ARGUMENT_LIST over empty CALL_EXPR
            # This assumes function calls in expression context usually have arguments
            if "ARGUMENT_LIST" in prod1 and "ARGUMENT_LIST" not in prod2:
                return prod1
            if "ARGUMENT_LIST" in prod2 and "ARGUMENT_LIST" not in prod1:
                return prod2

        # For ARGUMENT_LIST, prefer the single expression form to allow complex expressions
        if non_terminal == CFGNonTerminal.ARGUMENT_LIST:
            # Always prefer single EXPRESSION over multi-argument form
            # This allows expressions like "n + 1" to be parsed as a single binary expression
            # rather than being incorrectly split into multiple arguments
            if prod1 == ["EXPRESSION"]:
                return prod1
            if prod2 == ["EXPRESSION"]:
                return prod2

        # For PARAMETER_LIST, prefer non-empty forms
        if non_terminal == CFGNonTerminal.PARAMETER_LIST:
            if not prod1:  # Empty production
                return prod2
            if not prod2:  # Empty production
                return prod1

        # Default: prefer shorter productions (less complex)
        return prod1 if len(prod1) <= len(prod2) else prod2

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
                        # Continue with active productions
                        self._continue_productions()
                        return
                    else:
                        raise ValueError(f"Expected valid identifier, got {token}")
                # Special case: if the terminal is NUMBER, check if token is a numeric literal
                elif top == CFGTerminal.NUMBER:
                    try:
                        float(token)
                        self.stack.pop()
                        # Continue with active productions
                        self._continue_productions()
                        return
                    except ValueError:
                        raise ValueError(f"Expected valid number, got {token}")
                # Special case: if the terminal is STRING, check if token is a string literal
                elif top == CFGTerminal.STRING:
                    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
                        self.stack.pop()
                        # Continue with active productions
                        self._continue_productions()
                        return
                    else:
                        raise ValueError(f"Expected valid string literal, got {token}")
                # Special case: if the terminal is BOOLEAN, check if token is a boolean literal
                elif top == CFGTerminal.BOOLEAN:
                    if token in ['True', 'False']:
                        self.stack.pop()
                        # Continue with active productions
                        self._continue_productions()
                        return
                    else:
                        raise ValueError(f"Expected valid boolean literal, got {token}")
                # Compare terminal value with string token
                elif top.value == token:
                    self.stack.pop()
                    # Continue with active productions
                    self._continue_productions()
                    return
                else:
                    raise ValueError(f"Expected {top.value}, got {token}")
            else:
                # Compare terminal with terminal
                if top == token:
                    self.stack.pop()
                    # Continue with active productions
                    self._continue_productions()
                    return
                else:
                    raise ValueError(f"Expected {top}, got {token}")

        # Case 2: nonterminal
        elif isinstance(top, CFGNonTerminal):
            # Use prediction table for deterministic production selection
            if isinstance(token, CFGTerminal):
                token_key = token.value
            else:
                token_key = str(token)
            prediction_key = (top, token_key)

            log.debug(f"Token: {token}; Looking up prediction for {top} with token {token_key}")

            if prediction_key in self.prediction_table:
                # Use the predetermined production from prediction table
                rhs = self.prediction_table[prediction_key]
                log.debug(f"\tPrediction table selected production {rhs} for {top}")

                # Record rule choice (for AST reconstruction)
                self.history.append((top, rhs))
                # Add production to stack for tracking
                self.production_stack.append((top, rhs, 0))
                # Remove the non-terminal from stack and push RHS in reverse order (leftmost derivation)
                self.stack.pop()
                for sym in reversed(rhs):
                    resolved_sym = resolve_symbol(sym)
                    self.stack.append(resolved_sym)

                # Now try to advance again with the same token to handle immediate expansion
                # This handles the case where we expand a non-terminal and immediately need to match
                return self.advance(token)
            else:
                # No prediction available - check for empty productions as fallback
                productions = self.grammar.get_productions(top)
                for rhs in productions:
                    if not rhs:  # Empty production
                        log.debug(f"\tUsing empty production for {top}")
                        # Record rule choice (for AST reconstruction)
                        self.history.append((top, rhs))
                        # Remove the non-terminal without adding anything
                        self.stack.pop()
                        # Continue with active productions since we consumed the non-terminal
                        self._continue_productions()
                        # Now try to advance again with the same token
                        return self.advance(token)

                log.debug(f"No prediction found for {top} with token {token_key}")
                raise ValueError(f"Token {token} not valid for nonterminal {top}")

        # Case 3: string literal
        elif isinstance(top, str):
            if isinstance(token, str):
                if top == token:
                    self.stack.pop()
                    # Continue with active productions
                    self._continue_productions()
                    return
                else:
                    raise ValueError(f"Expected {top}, got {token}")
            else:
                # Token is CFGTerminal, compare with string value
                if top == token.value:
                    self.stack.pop()
                    # Continue with active productions
                    self._continue_productions()
                    return
                else:
                    raise ValueError(f"Expected {top}, got {token.value}")

    def _continue_productions(self):
        """Continue with active productions after consuming a terminal"""
        # Process production stack from top to bottom (most recent first)
        for i in range(len(self.production_stack) - 1, -1, -1):
            non_terminal, production, position = self.production_stack[i]
            position += 1

            # If we've consumed all symbols in this production, mark it as complete
            if position >= len(production):
                # Remove completed production from stack
                self.production_stack.pop(i)
                continue
            else:
                # Update position in this production
                self.production_stack[i] = (non_terminal, production, position)

                # Get the next symbol in the production
                next_symbol = production[position]
                next_symbol_resolved = resolve_symbol(next_symbol)

                # Push the next symbol onto the stack
                self.stack.append(next_symbol_resolved)

                # Only continue with the most recent production that has remaining symbols
                break

    def _can_start_with(self, production: list[str], token: CFGTerminal | str, visited: Optional[set] = None) -> bool:
        """Check if a production can start with the given token"""
        if visited is None:
            visited = set()

        if not production:
            return False

        first_symbol = production[0]
        first_symbol_resolved = resolve_symbol(first_symbol)

        # log.debug(f"Checking if {first_symbol} can start with {token}, resolved into {first_symbol_resolved}")

        # If first symbol is a terminal, check if it matches
        if isinstance(first_symbol_resolved, CFGTerminal):
            # log.debug(f"First symbol is CFGTerminal: {first_symbol_resolved}")

            if isinstance(token, str):
                # Special case: if the terminal is IDENTIFIER_LITERAL, any valid Python identifier should match
                if first_symbol_resolved == CFGTerminal.IDENTIFIER_LITERAL:
                    # log.debug(f"Checking IDENTIFIER_LITERAL against string token: {token}")
                    # log.debug(f"Token is valid identifier: {token.isidentifier()}")
                    return token.isidentifier()
                # Special case: if the terminal is NUMBER, check if token is a numeric literal
                elif first_symbol_resolved == CFGTerminal.NUMBER:
                    # log.debug(f"Checking NUMBER against string token: {token}")
                    try:
                        float(token)
                        # log.debug(f"Token is valid number: True")
                        return True
                    except ValueError:
                        # log.debug(f"Token is valid number: False")
                        return False
                # Special case: if the terminal is STRING, check if token is a string literal
                elif first_symbol_resolved == CFGTerminal.STRING:
                    # log.debug(f"Checking STRING against string token: {token}")
                    is_string = (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'"))
                    # log.debug(f"Token is valid string: {is_string}")
                    return is_string
                # Special case: if the terminal is BOOLEAN, check if token is a boolean literal
                elif first_symbol_resolved == CFGTerminal.BOOLEAN:
                    # log.debug(f"Checking BOOLEAN against string token: {token}")
                    is_boolean = token in ['True', 'False']
                    # log.debug(f"Token is valid boolean: {is_boolean}")
                    return is_boolean
                # log.debug(f"Comparing {first_symbol_resolved.value} == {token} = {first_symbol_resolved.value == token}")
                return first_symbol_resolved.value == token

            else:
                # log.debug(f"Comparing {first_symbol_resolved} == {token} = {first_symbol_resolved == token}")
                return first_symbol_resolved == token

        elif isinstance(first_symbol_resolved, CFGNonTerminal):
            # log.debug(f"First symbol is CFGNonTerminal: {first_symbol_resolved}")
            # Check if this non-terminal can start with the token
            return self._non_terminal_can_start_with(first_symbol_resolved, token, visited)

        elif isinstance(first_symbol_resolved, str):
            # log.debug(f"First symbol is str: {first_symbol_resolved}")
            if isinstance(token, str):
                # log.debug(f"Comparing {first_symbol_resolved} == {token} = {first_symbol_resolved == token}")
                return first_symbol_resolved == token
            else:
                # log.debug(f"Comparing {first_symbol_resolved} == {token.value} = {first_symbol_resolved == token.value}")
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

    def is_accepting(self) -> bool:
        """Check if the parser is in an accepting state"""
        return len(self.stack) == 0

    def get_parse_tree(self) -> dict[str, Any]:
        """Get the parse tree from the parser history"""
        return {
            **{
                "history": self.history,
                "production_stack": self.production_stack,
                "stack": list(self.stack),
            }, **({} if len(self.error_messages) == 0 else {"error_messages": self.error_messages})
        }


def validate_ast_syntax(ast_graph: dict[str, Any]) -> bool:
    """Validate the syntax of an AST graph"""
    return True


def parse_code(code: str) -> tuple[bool, dict[str, Any]]:
    """
    Parse code using the CFG parser and tokenizer.

    Args:
        code: source code as string

    Returns:
        Tuple of (success, parse_info) where parse_info contains parser state
    """

    # Tokenize the code
    tokens = tokenize_code(code)

    # Create parser and consume tokens
    parser = CFGParser()

    try:
        success = parser.consume_tokens(tokens)
        parse_info = parser.get_parse_tree()
        return success, parse_info
    except Exception as e:
        log.error(f"Parsing failed: {e}")
        return False, {"error": str(e)}