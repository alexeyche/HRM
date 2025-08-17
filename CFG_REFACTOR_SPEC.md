# Proposed Spec: CFG Token Validator for Constrained Generation

## Core Concept: From Parser to Token Constraint Engine

Instead of parsing complete programs, we focus on **real-time token validation** during generation:

```
Neural Model → Generates Token → CFG Validator → Valid/Invalid + Next Valid Tokens
```

## Simplified Architecture

### 1. **Core Interface**
```python
class CFGTokenValidator:
    def get_valid_tokens(self, generation_state: GenerationState) -> set[str]:
        """Return set of tokens that are syntactically valid at current position"""

    def is_valid_token(self, token: str, generation_state: GenerationState) -> bool:
        """Check if a specific token is valid at current position"""

    def advance_with_token(self, token: str, generation_state: GenerationState) -> GenerationState:
        """Update generation state after consuming a valid token"""

    def get_completion_probability(self, generation_state: GenerationState) -> float:
        """Return probability that current state can lead to valid completion"""
```

### 2. **Simplified State Management**
```python
@dataclass
class GenerationState:
    """Lightweight state for generation - much simpler than full parser state"""
    syntax_stack: list[CFGNonTerminal | CFGTerminal]  # What we're expecting
    context_stack: list[str]  # Context for disambiguation (parent productions)
    token_history: list[str]  # Recently generated tokens for context
    completion_depth: int  # How deep in nested structures we are

    # These replace the complex parser state
    # No need for: production_stack, history, error_messages
```

## Key Simplifications vs Current CFGParser

### ❌ Remove Complex Parser Logic
- **No `advance()` method** - no need to fully parse and maintain parse state
- **No `_continue_productions()`** - no need to track production continuations
- **No parse history/AST reconstruction** - we only care about next valid tokens
- **No error recovery** - generation just backtracks or resamples

### ✅ Keep Essential Components
- **Prediction table** - this is perfect for token validation
- **Grammar rules** - needed to compute valid tokens
- **Stack-based state** - simplified version for syntax context

## Detailed Implementation Plan

### 1. **Token Validation Engine**
```python
class CFGTokenValidator:
    def __init__(self, grammar: CFGGrammar):
        self.grammar = grammar
        self.prediction_table = self._build_prediction_table()
        self.first_sets = self._compute_all_first_sets()
        self.follow_sets = self._compute_all_follow_sets()

    def get_valid_tokens(self, state: GenerationState) -> set[str]:
        """Core method: return all tokens valid at current position"""
        if not state.syntax_stack:
            return {"<END>"}  # Program complete

        top = state.syntax_stack[-1]

        if isinstance(top, CFGTerminal):
            return self._get_terminal_tokens(top)
        elif isinstance(top, CFGNonTerminal):
            return self._get_nonterminal_first_tokens(top, state.context_stack)
        else:
            return {top}  # String literal

    def _get_terminal_tokens(self, terminal: CFGTerminal) -> set[str]:
        """Get valid tokens for a terminal symbol"""
        if terminal == CFGTerminal.IDENTIFIER_LITERAL:
            return {"<IDENTIFIER>"}  # Placeholder - model generates actual identifier
        elif terminal == CFGTerminal.NUMBER:
            return {"<NUMBER>"}  # Placeholder - model generates actual number
        else:
            return {terminal.value}  # Exact token required

    def _get_nonterminal_first_tokens(self, nt: CFGNonTerminal, context: list[str]) -> set[str]:
        """Get valid first tokens for a non-terminal with context"""
        valid_tokens = set()

        for production in self.grammar.get_productions(nt):
            # Apply context-aware production selection
            if self._production_valid_in_context(production, context):
                first_tokens = self._get_production_first_tokens(production)
                valid_tokens.update(first_tokens)

        return valid_tokens
```

### 2. **Context-Aware Production Selection**
```python
def _production_valid_in_context(self, production: list[str], context: list[str]) -> bool:
    """Use context to disambiguate productions - replaces complex lookahead"""

    # Example: ARGUMENT_LIST disambiguation
    if production == ["EXPRESSION"] and "range" in context[-3:]:
        # In range() context, prefer multi-argument form for first argument
        return False
    elif production == ["EXPRESSION", "COMMA", "SPACE", "ARGUMENT_LIST"] and "range" in context[-3:]:
        return True

    # More context rules...
    return True  # Default: allow all productions
```

### 3. **Lightweight State Updates**
```python
def advance_with_token(self, token: str, state: GenerationState) -> GenerationState:
    """Update state after token - much simpler than full parsing"""
    new_stack = state.syntax_stack.copy()
    new_context = state.context_stack.copy()
    new_history = state.token_history + [token]

    if not new_stack:
        return state  # Already complete

    top = new_stack.pop()

    if isinstance(top, CFGTerminal):
        # Terminal matched - just continue
        pass
    elif isinstance(top, CFGNonTerminal):
        # Expand non-terminal using prediction table
        production = self._select_production(top, token, new_context)
        if production:
            new_context.append(f"{top.value}→{production}")
            # Push production symbols in reverse order
            for symbol in reversed(production):
                resolved = resolve_symbol(symbol)
                new_stack.append(resolved)

    return GenerationState(
        syntax_stack=new_stack,
        context_stack=new_context[-10:],  # Keep recent context
        token_history=new_history[-20:],  # Keep recent tokens
        completion_depth=len(new_stack)
    )
```

## Integration with Neural Generation

### 1. **Constrained Beam Search**
```python
def constrained_beam_search(model, validator, initial_state, beam_size=5):
    """Generate code with syntax constraints"""
    beams = [(initial_state, [], 0.0)]  # (state, tokens, score)

    while beams:
        new_beams = []

        for state, tokens, score in beams:
            valid_tokens = validator.get_valid_tokens(state)

            if "<END>" in valid_tokens:
                yield tokens, score  # Complete program
                continue

            # Get model probabilities for valid tokens only
            token_probs = model.get_token_probabilities(tokens, valid_tokens)

            # Expand beam with top-k valid tokens
            for token, prob in token_probs.top_k(beam_size):
                new_state = validator.advance_with_token(token, state)
                new_beams.append((new_state, tokens + [token], score + log(prob)))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]
```

### 2. **Real-time Validation During Generation**
```python
def generate_with_constraints(model, validator, max_length=100):
    """Generate tokens one by one with real-time validation"""
    state = GenerationState.initial()  # Start with PROGRAM on stack
    tokens = []

    for _ in range(max_length):
        valid_tokens = validator.get_valid_tokens(state)

        if "<END>" in valid_tokens:
            break  # Program complete

        # Constrain model output to valid tokens
        token = model.generate_next_token(tokens, constraints=valid_tokens)

        if not validator.is_valid_token(token, state):
            raise ValueError(f"Model generated invalid token: {token}")

        tokens.append(token)
        state = validator.advance_with_token(token, state)

    return tokens
```

## Benefits of This Approach

### ✅ **Perfectly Aligned with Neural Generation**
- **Token-level constraints** - exactly what beam search needs
- **No complex parsing state** - just simple stack + context
- **Real-time validation** - check each token as it's generated
- **Backtrack-friendly** - easy to undo invalid generations

### ✅ **Much Simpler Implementation**
- **~200 lines vs ~600 lines** of current parser
- **No production tracking** - just prediction + validation
- **No error recovery** - generation handles invalid paths
- **Clear separation** - syntax validation vs parsing

### ✅ **Context-Aware Disambiguation**
- **Replaces lookahead** with learned context patterns
- **Handles ambiguous grammars** through context rules
- **Extensible** - easy to add new disambiguation rules

## Migration Path from Current CFGParser

1. **Phase 1**: Extract `valid_next_tokens()` method and prediction table
2. **Phase 2**: Simplify state to just syntax_stack + context
3. **Phase 3**: Remove parsing logic, keep only token validation
4. **Phase 4**: Add context-aware production selection
5. **Phase 5**: Integrate with neural generation pipeline

This approach transforms your CFGParser from a **complex recursive descent parser** into a **lightweight syntax constraint engine** perfectly suited for neural program synthesis!