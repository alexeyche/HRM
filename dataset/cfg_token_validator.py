"""
CFG Token Validator for Constrained Generation

This module provides a lightweight token validation engine for neural program synthesis.
Instead of complex parsing, it focuses on real-time token validation during generation.
"""

from typing import Any, Optional, Set, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from dataset.cfg import CFGNonTerminal, CFGTerminal, CFGGrammar
from dataset.cfg_parser import resolve_symbol

log = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """Lightweight state for generation - much simpler than full parser state"""
    syntax_stack: List[CFGNonTerminal | CFGTerminal | str]  # What we're expecting
    context_stack: List[str]  # Context for disambiguation (parent productions)
    token_history: List[str]  # Recently generated tokens for context
    completion_depth: int  # How deep in nested structures we are

    @classmethod
    def initial(cls) -> "GenerationState":
        """Create initial state with PROGRAM on stack"""
        return cls(
            syntax_stack=[CFGNonTerminal.PROGRAM],
            context_stack=[],
            token_history=[],
            completion_depth=1
        )

    def is_complete(self) -> bool:
        """Check if generation is complete (empty stack)"""
        return len(self.syntax_stack) == 0


class CFGTokenValidator:
    """Core token validation engine for constrained generation"""

    def __init__(self, grammar: Optional[CFGGrammar] = None):
        self.grammar = grammar or CFGGrammar()
        self.prediction_table: Dict[Tuple[CFGNonTerminal, str], List[str]] = {}
        self.first_sets: Dict[CFGNonTerminal, Set[str]] = {}
        self.follow_sets: Dict[CFGNonTerminal, Set[str]] = {}

        self._build_prediction_table()
        self._compute_all_first_sets()

    def get_valid_tokens(self, state: GenerationState) -> Set[str]:
        """Return set of tokens that are syntactically valid at current position"""
        if not state.syntax_stack:
            return {"<END>"}  # Program complete

        top = state.syntax_stack[-1]

        if isinstance(top, CFGTerminal):
            return self._get_terminal_tokens(top)
        elif isinstance(top, CFGNonTerminal):
            first_tokens = self._get_nonterminal_first_tokens(top, state.context_stack)

            # If the non-terminal can derive epsilon, also include what can follow it
            if "<EPSILON>" in first_tokens:
                first_tokens.remove("<EPSILON>")  # Remove epsilon marker
                # Look at what's next on the stack
                if len(state.syntax_stack) > 1:
                    # Get tokens for the next symbol on the stack
                    next_symbol = state.syntax_stack[-2]
                    if isinstance(next_symbol, CFGTerminal):
                        first_tokens.update(self._get_terminal_tokens(next_symbol))
                    elif isinstance(next_symbol, CFGNonTerminal):
                        first_tokens.update(self._get_nonterminal_first_tokens(next_symbol, state.context_stack))
                    else:
                        first_tokens.add(next_symbol)  # String literal
                else:
                    # Nothing follows, so we can end
                    first_tokens.add("<END>")

            return first_tokens
        else:
            return {top}  # String literal

    def is_valid_token(self, token: str, state: GenerationState) -> bool:
        """Check if a specific token is valid at current position"""
        valid_tokens = self.get_valid_tokens(state)

        # Check for direct match
        if token in valid_tokens:
            return True

        # Check for placeholder matches
        if "<IDENTIFIER>" in valid_tokens and self._is_valid_identifier(token):
            return True
        if "<NUMBER>" in valid_tokens and self._is_valid_number(token):
            return True
        if "<STRING>" in valid_tokens and self._is_valid_string(token):
            return True
        if "<BOOLEAN>" in valid_tokens and self._is_valid_boolean(token):
            return True

        return False

    def advance_with_token(self, token: str, state: GenerationState) -> GenerationState:
        """Update generation state after consuming a valid token"""
        if not self.is_valid_token(token, state):
            raise ValueError(f"Invalid token '{token}' at current position")

        new_stack = state.syntax_stack.copy()
        new_context = state.context_stack.copy()
        new_history = state.token_history + [token]

        # Keep consuming from the stack until we match the token
        token_consumed = False

        while new_stack and not token_consumed:
            top = new_stack.pop()

            if isinstance(top, CFGTerminal):
                # Terminal must match the token
                if self._terminal_matches_token(top, token):
                    token_consumed = True
                else:
                    raise ValueError(f"Expected terminal {top}, got {token}")

            elif isinstance(top, CFGNonTerminal):
                # Expand non-terminal using prediction table
                production = self._select_production(top, token, new_context)
                if production is not None:  # production can be empty list
                    if production:  # Non-empty production
                        new_context.append(f"{top.value}→{' '.join(production)}")
                        # Push production symbols in reverse order for leftmost derivation
                        for symbol in reversed(production):
                            resolved = resolve_symbol(symbol)
                            new_stack.append(resolved)
                    else:  # Empty production (epsilon)
                        new_context.append(f"{top.value}→ε")
                        # Don't push anything for empty production
                else:
                    raise ValueError(f"No valid production for {top} with token {token}")

            else:
                # String literal must match the token
                if top == token:
                    token_consumed = True
                else:
                    raise ValueError(f"Expected '{top}', got '{token}'")

        if not token_consumed:
            raise ValueError(f"Could not consume token {token}")

        return GenerationState(
            syntax_stack=new_stack,
            context_stack=new_context[-10:],  # Keep recent context
            token_history=new_history[-20:],  # Keep recent tokens
            completion_depth=len(new_stack)
        )

    def _terminal_matches_token(self, terminal: CFGTerminal, token: str) -> bool:
        """Check if a terminal matches the given token"""
        if terminal == CFGTerminal.IDENTIFIER_LITERAL:
            return self._is_valid_identifier(token)
        elif terminal == CFGTerminal.NUMBER:
            return self._is_valid_number(token)
        elif terminal == CFGTerminal.STRING:
            return self._is_valid_string(token)
        elif terminal == CFGTerminal.BOOLEAN:
            return self._is_valid_boolean(token)
        else:
            return terminal.value == token

    def get_completion_probability(self, state: GenerationState) -> float:
        """Return probability that current state can lead to valid completion"""
        # Simple heuristic: shorter stack = closer to completion
        if state.is_complete():
            return 1.0

        # Penalize very deep nesting
        max_depth = 50
        if state.completion_depth > max_depth:
            return 0.1

        # Higher probability for simpler completions
        return max(0.1, 1.0 - (state.completion_depth / max_depth))

    def _get_terminal_tokens(self, terminal: CFGTerminal) -> Set[str]:
        """Get valid tokens for a terminal symbol"""
        if terminal == CFGTerminal.IDENTIFIER_LITERAL:
            return {"<IDENTIFIER>"}  # Placeholder - model generates actual identifier
        elif terminal == CFGTerminal.NUMBER:
            return {"<NUMBER>"}  # Placeholder - model generates actual number
        elif terminal == CFGTerminal.STRING:
            return {"<STRING>"}  # Placeholder - model generates actual string
        elif terminal == CFGTerminal.BOOLEAN:
            return {"<BOOLEAN>"}  # Placeholder - model generates actual boolean
        else:
            return {terminal.value}  # Exact token required

    def _get_nonterminal_first_tokens(self, nt: CFGNonTerminal, context: List[str]) -> Set[str]:
        """Get valid first tokens for a non-terminal with context"""
        valid_tokens = set()

        for production in self.grammar.get_productions(nt):
            # Apply context-aware production selection
            if self._production_valid_in_context(production, context):
                first_tokens = self._get_production_first_tokens(production)
                valid_tokens.update(first_tokens)

        return valid_tokens

    def _get_production_first_tokens(self, production: List[str]) -> Set[str]:
        """Get first tokens that can start this production"""
        if not production:
            return set()

        first_symbol = production[0]
        resolved = resolve_symbol(first_symbol)

        if isinstance(resolved, CFGTerminal):
            return self._get_terminal_tokens(resolved)
        elif isinstance(resolved, CFGNonTerminal):
            return self.first_sets.get(resolved, set())
        else:
            return {first_symbol}  # String literal

    def _production_valid_in_context(self, production: List[str], context: List[str]) -> bool:
        """Use context to disambiguate productions - replaces complex lookahead"""

        # Convert production to string for context checking
        prod_str = "→".join(production) if production else "ε"

        # Example: ARGUMENT_LIST disambiguation - but be more permissive
        if production == ["EXPRESSION"] and any("range" in ctx for ctx in context[-3:]):
            # In range() context, still allow single expressions (common case)
            return True
        elif (production == ["EXPRESSION", ",", " ", "ARGUMENT_LIST"] and
              any("range" in ctx for ctx in context[-3:])):
            return True

        # EXPRESSION disambiguation - prefer simpler forms in simple contexts
        if production == ["CALL_EXPR"] and any("return" in ctx for ctx in context[-2:]):
            # In return context, function calls are common
            return True
        elif production == ["VARIABLE"] and any("=" in ctx for ctx in context[-2:]):
            # In assignment context, variables are common
            return True

        # For loops prefer range calls
        if production == ["CALL_EXPR"] and any("for" in ctx for ctx in context[-3:]):
            return True

        # Default: allow all productions
        return True

    def _select_production(self, non_terminal: CFGNonTerminal, token: str, context: List[str]) -> Optional[List[str]]:
        """Select appropriate production for non-terminal given token and context"""

        # First check prediction table
        prediction_key = (non_terminal, token)
        if prediction_key in self.prediction_table:
            production = self.prediction_table[prediction_key]
            if self._production_valid_in_context(production, context):
                return production

        # Check if any non-empty production can start with this token
        for production in self.grammar.get_productions(non_terminal):
            if (production and  # Non-empty production
                self._production_can_start_with_token(production, token) and
                self._production_valid_in_context(production, context)):
                return production

        # Check if there's an empty production and the token can follow this non-terminal
        for production in self.grammar.get_productions(non_terminal):
            if not production:  # Empty production
                # For now, allow empty production as fallback
                # TODO: Should check if token is in FOLLOW set of non_terminal
                return production

        return None

    def _production_can_start_with_token(self, production: List[str], token: str) -> bool:
        """Check if production can start with given token"""
        if not production:
            return False

        first_symbol = production[0]
        resolved = resolve_symbol(first_symbol)

        if isinstance(resolved, CFGTerminal):
            if resolved == CFGTerminal.IDENTIFIER_LITERAL:
                return self._is_valid_identifier(token)
            elif resolved == CFGTerminal.NUMBER:
                return self._is_valid_number(token)
            elif resolved == CFGTerminal.STRING:
                return self._is_valid_string(token)
            elif resolved == CFGTerminal.BOOLEAN:
                return self._is_valid_boolean(token)
            else:
                return resolved.value == token
        elif isinstance(resolved, CFGNonTerminal):
            # Check if non-terminal's first set contains this token
            first_set = self.first_sets.get(resolved, set())
            return (token in first_set or
                    ("<IDENTIFIER>" in first_set and self._is_valid_identifier(token)) or
                    ("<NUMBER>" in first_set and self._is_valid_number(token)) or
                    ("<STRING>" in first_set and self._is_valid_string(token)) or
                    ("<BOOLEAN>" in first_set and self._is_valid_boolean(token)))
        else:
            return first_symbol == token

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
                        # LL(1) conflict detected - use simpler production
                        existing_prod = self.prediction_table[key]
                        better_prod = self._resolve_production_conflict(
                            non_terminal, token_key, existing_prod, production
                        )
                        self.prediction_table[key] = better_prod
                    else:
                        self.prediction_table[key] = production

    def _compute_production_first_set(self, production: List[str]) -> Set:
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
                                   prod1: List[str], prod2: List[str]) -> List[str]:
        """Resolve LL(1) conflicts using heuristics suitable for generation"""

        # Prefer shorter productions (simpler)
        if len(prod1) < len(prod2):
            return prod1
        elif len(prod2) < len(prod1):
            return prod2

        # For EXPRESSION, prefer specific forms over general
        if non_terminal == CFGNonTerminal.EXPRESSION:
            priority_order = [
                ["CALL_EXPR"],          # Function call
                ["VARIABLE"],           # Simple variable
                ["CONSTANT"],           # Literal values
                ["BINARY_EXPR"],        # Binary operation
                ["COMPARISON_EXPR"],    # Comparison
                ["BOOLEAN_EXPR"],       # Boolean operation
            ]

            for preferred in priority_order:
                if prod1 == preferred:
                    return prod1
                if prod2 == preferred:
                    return prod2

        # Default: prefer first production
        return prod1

    def _compute_all_first_sets(self):
        """Compute FIRST sets for all non-terminals"""
        self.first_sets = {}

        # Initialize with empty sets
        for nt in CFGNonTerminal:
            self.first_sets[nt] = set()

        # Iterate until no changes (fixed point)
        changed = True
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for nt in CFGNonTerminal:
                old_size = len(self.first_sets[nt])
                new_first = self._compute_first_set(nt)
                self.first_sets[nt].update(new_first)
                if len(self.first_sets[nt]) > old_size:
                    changed = True

    def _compute_first_set(self, non_terminal: CFGNonTerminal, visited: Optional[Set] = None) -> Set[str]:
        """Compute FIRST set for a non-terminal"""
        if visited is None:
            visited = set()

        if non_terminal in visited:
            return set()  # Avoid infinite recursion

        visited.add(non_terminal)
        first_set = set()

        productions = self.grammar.get_productions(non_terminal)

        # Handle non-terminals with no productions (like OPERATOR)
        if not productions:
            log.warning(f"Non-terminal {non_terminal} has no productions defined")
            # For OPERATOR, provide some reasonable defaults
            if non_terminal == CFGNonTerminal.OPERATOR:
                first_set.update(["+", "-", "*", "/", "%", "**", "==", "!=", "<", ">", "<=", ">=", "and", "or", "not"])
            return first_set

        has_empty_production = False
        for production in productions:
            if not production:  # Empty production
                has_empty_production = True
                continue

            first_symbol = production[0]
            first_symbol_resolved = resolve_symbol(first_symbol)

            if isinstance(first_symbol_resolved, CFGTerminal):
                if first_symbol_resolved == CFGTerminal.IDENTIFIER_LITERAL:
                    first_set.add("<IDENTIFIER>")
                elif first_symbol_resolved == CFGTerminal.NUMBER:
                    first_set.add("<NUMBER>")
                elif first_symbol_resolved == CFGTerminal.STRING:
                    first_set.add("<STRING>")
                elif first_symbol_resolved == CFGTerminal.BOOLEAN:
                    first_set.add("<BOOLEAN>")
                else:
                    first_set.add(first_symbol_resolved.value)
            elif isinstance(first_symbol_resolved, CFGNonTerminal):
                # Recursively compute FIRST set for non-terminal
                first_set.update(self._compute_first_set(first_symbol_resolved, visited.copy()))
            elif isinstance(first_symbol_resolved, str):
                first_set.add(first_symbol_resolved)

        # If there's an empty production, we need to include FOLLOW set
        if has_empty_production:
            first_set.add("<EPSILON>")  # Mark that epsilon is in FIRST set

        return first_set

    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid Python identifier"""
        import keyword
        return token.isidentifier() and not keyword.iskeyword(token)

    def _is_valid_number(self, token: str) -> bool:
        """Check if token is a valid numeric literal"""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def _is_valid_string(self, token: str) -> bool:
        """Check if token is a valid string literal"""
        return ((token.startswith('"') and token.endswith('"')) or
                (token.startswith("'") and token.endswith("'")))

    def _is_valid_boolean(self, token: str) -> bool:
        """Check if token is a valid boolean literal"""
        return token in ['True', 'False']


# Convenience functions for neural generation integration

def constrained_beam_search(model, validator: CFGTokenValidator, initial_state: GenerationState,
                          beam_size: int = 5, max_length: int = 100):
    """Generate code with syntax constraints using beam search"""
    beams = [(initial_state, [], 0.0)]  # (state, tokens, score)
    completed = []

    for step in range(max_length):
        if not beams:
            break

        new_beams = []

        for state, tokens, score in beams:
            valid_tokens = validator.get_valid_tokens(state)

            if "<END>" in valid_tokens:
                completed.append((tokens, score))  # Complete program
                continue

            # Get model probabilities for valid tokens only
            # Note: This would interface with actual model
            # token_probs = model.get_token_probabilities(tokens, valid_tokens)

            # For now, simulate with uniform probabilities
            for token in list(valid_tokens)[:beam_size]:
                if token not in ["<END>", "<IDENTIFIER>", "<NUMBER>", "<STRING>", "<BOOLEAN>"]:
                    try:
                        new_state = validator.advance_with_token(token, state)
                        new_beams.append((new_state, tokens + [token], score + 1.0))
                    except ValueError:
                        continue  # Skip invalid tokens

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]

    return completed + [(tokens, score) for state, tokens, score in beams]


def generate_with_constraints(validator: CFGTokenValidator, token_sequence: List[str]) -> bool:
    """Validate a sequence of tokens against the grammar constraints"""
    state = GenerationState.initial()

    try:
        for token in token_sequence:
            if not validator.is_valid_token(token, state):
                return False
            state = validator.advance_with_token(token, state)

        # Check if we reached a valid completion state
        return state.is_complete() or "<END>" in validator.get_valid_tokens(state)
    except ValueError:
        return False
