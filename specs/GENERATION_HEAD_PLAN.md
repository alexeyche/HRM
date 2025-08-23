# Grammar-Aware Generation Head Implementation Plan

Following the agile development guidelines from CLAUDE.md, I'll implement the generation head incrementally:

## Phase 1: Core Infrastructure (Skeleton) ✅
1. **Create `model/generation_head.py`** with basic class structure:
   - `GrammarAwareGenerationHead` main class
   - `ProductionHead` for rule selection
   - Basic scaffolding for specialized value heads
   - Integration with NLTK grammar from `dataset/grammar.py`

2. **Create `tests/test_generation_head.py`** with initial tests:
   - Test grammar integration and rule masking
   - Test basic forward pass structure
   - Test production stack management

## Phase 2: Production Head Implementation
3. **Implement core Production Head logic**:
   - Rule masking based on current non-terminal
   - Softmax over valid productions
   - Integration with grammar productions from `grammar.py`

4. **Add production head tests**:
   - Test valid rule selection
   - Test masking correctness
   - Test grammar compliance

## Phase 3: Specialized Value Heads
5. **Implement basic value heads**:
   - `IdentifierHead` (generation + copy mechanism)
   - `LiteralHead` (type + value prediction)
   - Simple routing logic

6. **Add value head tests**:
   - Test identifier generation and copying
   - Test literal type/value prediction
   - Test routing between heads

## Phase 4: Integration & Stack Management
7. **Implement expansion workflow**:
   - Production expansion logic
   - Stack management for non-terminals
   - Terminal emission and concatenation

8. **Add integration tests**:
   - End-to-end generation tests
   - Test complete program generation
   - Test syntactic validity

## Key Design Decisions Made:
- Use NLTK CFG as single source of truth for rule validation
- Implement expansion stack internally within the generation head
- Route to specialized heads based on grammar productions deterministically
- Target next production rule prediction for training efficiency