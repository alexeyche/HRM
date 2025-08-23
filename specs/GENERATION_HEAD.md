# 📐 Grammar-Aware Generation Head Design

## 1. Context

We want a **GPT-like code generator** constrained by a Python-subset grammar (in `grammar.py`).

* Backbone: Transformer with attention (modern GPT style).
* Decoder Head: Grammar-aware, drives the **expansion process**.
* Output: Always a sequence of **terminals only** (compilable Python code).
* Internals: Uses **non-terminals** as scaffolding during expansion.

---

## 2. Inputs to the Head

The head takes:

1. **Context embeddings** → hidden state from Transformer (conditioned on all previous expansions).
2. **Current production state** → the non-terminal at the top of the expansion stack (empty at sequence start).

---

## 3. Head Components

### A. **Production Head**

* Core function: **choose a grammar production rule** for the current non-terminal.
* Operates at the **rule level** (expanding both into terminals and non-terminals).
* Implementation:

  * Mask softmax over *only valid rules* for that non-terminal (use `grammar.py` + NLTK parsing utilities).
  * Produces either:

    * Non-terminal(s) → push onto stack for further expansion.
    * Terminal(s) → emit directly or pass to specialized head.

---

### B. **Specialized Value Heads**

When a production expands into terminals that require modeling, we route prediction to the right head:

1. **Identifier Head**

   * Handles `<identifier>` expansions.
   * Supports:

     * Generation of new identifiers.
     * Copy/pointer mechanism to reuse existing identifiers in scope.

2. **Literal Head**

   * Handles `<literal>` expansions (numbers, strings, booleans, etc).
   * Factorized prediction: first literal type → then literal value.

3. **Function Call Head**

   * Handles productions like `Call → <func_name> "(" ArgList ")"`.
   * Predicts:

     * Which function (from known set or identifier head).
     * How many arguments.
     * Delegates each argument expansion back into Production Head.

4. **Control Flow Head**

   * Handles constructs like `if`, `while`, `for`.
   * Predicts specific control keyword (masked choice).
   * Expands required sub-blocks (condition, body) by calling back into Production Head.

---

## 4. Workflow (One Step)

1. Transformer provides contextual embedding of current state.
2. Production Head selects a valid expansion rule for the active non-terminal.
3. If expansion contains:

   * Pure terminals (keywords, operators): emit directly.
   * Typed terminals (`<identifier>`, `<literal>`, `<func_call>`, `<ctrl_flow>`): route to appropriate **value head**.
4. Push any new non-terminals from the expansion onto the stack.
5. Repeat until no non-terminals remain.
6. Output = concatenated terminals → valid Python code.

---

## 5. Integration with `grammar.py`

* Use the NLTK CFG in `grammar.py` as the **single source of truth** for valid expansions.
* At each step:

  * Query grammar object for valid productions of the current non-terminal.
  * Use NLTK’s parsing utilities to manage expansion stacks.
* This avoids reinventing rule masking, parse stacks, or grammar traversal.

---

## 6. Key Advantages

* **Grammar guarantees syntactic validity** (no junk code).
* **Specialized heads provide inductive bias**, improving training efficiency.
* **Separation of concerns**:

  * Production Head = structure.
  * Value Heads = content.
* **Composable**: easy to extend grammar or add new value heads.

---

✅ **Summary in one line**:
A **grammar-aware GPT-like generation head** that, given embeddings + current production state, uses a **Production Head** (rule-level expansion) plus **specialized value heads** (identifiers, literals, function calls, control flow) to generate valid Python code by expanding the NLTK grammar in `grammar.py`.

