    Priority 1: Fix Program Completion (CRITICAL - 80% of effort)

    1. Implement proper termination logic in generate_program()
      - Add completion detection for empty expansion stack
      - Fix end-of-program recognition
      - Ensure all non-terminals properly expanded
    2. Fix production-to-head routing logic (Line 459)
      - Analyze selected production to determine required terminals
      - Route to appropriate specialized heads based on production RHS
      - Implement deterministic routing instead of calling all heads
    3. Improve stack management
      - Better handling of complex nested structures (function bodies, if-blocks)
      - Proper order of expansion for correct program flow

    Priority 2: Enhance Terminal Handling (15% of effort)

    1. Standardize terminal symbol processing
      - Consistent mapping for all terminal types
      - Better integration with grammar token patterns
      - Handle edge cases in terminal-to-token conversion
    2. Fix specialized head integration
      - Proper context passing for identifier copy mechanism
      - Better literal type selection based on context

    Priority 3: Add Training Support (5% of effort)

    1. Implement next-production target generation
      - For training the production head properly
      - Loss computation for grammar compliance
      - Better convergence on valid programs

    The core issue is that 100% of generated programs are incomplete due to broken termination logic and missing production routing. Fixing these two
    issues should immediately improve validity from 0% to a reasonable percentage (target: 60-80% valid programs).