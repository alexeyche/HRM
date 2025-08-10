# Plan: Grammar-/production-level decoding with graph embeddings as input

- Goal
  - Encode an input program graph and decode a grammar-constrained AST (initially as an auto-encoder to validate decoder + masking). Later, swap encoder input to problem-text or spec.

- Data/targets
  - Use `ASTSimplifier.ast_to_graph` to parse `ProgramSpecification.implementation` into target ASTs.
  - Build an oracle converter: AST → action sequence, and its inverse: actions → AST.
  - Start with a compact action space:
    - Node-introduction actions for each `ASTNodeType` with arity/child roles.
    - Attribute-selection actions for fields: `op` (from operator vocab), `ctx`, `dtype`, `function`, `attr`, `var_id`, `const_exact_int_id`.
    - List/variadic constructs (e.g., function args, list elts) via begin_list, emit_item, end_list or fixed-arity if you choose.
  - Use existing vocabs from `GraphEncoder.vocab_sizes()` for masking/logit heads.

- Grammar spec and masking
  - Define a CFG/state machine keyed by nonterminals corresponding to `ASTNodeType` plus role labels (e.g., Expr, Stmt, ArgList).
  - For each expansion, provide:
    - Valid next productions given top-of-stack nonterminal.
    - Attribute heads and masks (e.g., `op` only for BINARY_OPERATION/UNARY_OPERATION/COMPARISON/BOOLEAN_OPERATION; `dtype` only for CONSTANT).
  - Child pushing: after a production, push expected child roles onto a stack; pop when satisfied.
  - Sibling order is implicit by expansion order; add `NEXT_SIBLING` edges post-build as in `ast_to_graph`.

- Inputs (graph embeddings)
  - Encode the input graph with `GraphEncoder.encode` and `to_pyg_data`.
  - Build node embeddings via `NodeFeatureBuilder` (sum/concat mode).
  - Apply a light GNN (e.g., 2–3 GCN/GAT layers) and a graph readout (mean/attention) to get:
    - Global context embedding.
    - Optional node-memory tokens for cross-attention (improves conditioning).

- Decoder
  - Transformer decoder over action tokens with:
    - Token embeddings for actions (separate embeddings per action type or a unified vocab with type IDs).
    - Cross-attention to graph context (global vector or node tokens).
    - Masking function that zeroes logits of invalid productions/attributes at each step based on parser stack and role.
  - Attribute prediction:
    - Separate projection heads over shared decoder hidden for `op`, `dtype`, etc., with masks activated only when needed.
    - For ints: first support exact-id via `small_int_vocab`; fall back to UNK now; later add regression head.

- Supervision and loss
  - Teacher-forcing with cross-entropy over:
    - Production choice.
    - Attribute choices (only when applicable).
  - Optionally, auxiliary losses:
    - Predict number of children (for lists/variadics).
    - Predict stop/end_list.
  - Label smoothing and dropout; gradient clipping.

- Decoding/inference
  - Grammar-masked greedy or beam search (small beam, e.g., 4–8).
  - Reconstruct AST, add `NEXT_SIBLING`, then compare to target AST for exact-tree accuracy.
  - Later add execution accuracy by turning the AST back to Python and running on `ProgramSpecification` examples.

- Datasets and loaders
  - Build a dataset that yields:
    - input_graph: `EncodedGraph` (numpy) → PyG `Data`.
    - target_actions: int sequence with per-step action-type tags and attribute labels.
  - Collate with padding; generate per-step masks for attribute heads.

- Minimal milestones
  1. Define action vocabulary and parser stack roles for a subset: `FUNCTION_DEF`, `RETURN`, `VARIABLE`, `BINARY_OPERATION`, `CONSTANT`, `LIST`, `FUNCTION_CALL`, `ATTRIBUTE`. Map `GraphEncoder` vocabs to action spaces.
  2. Implement AST ↔ actions converters and unit tests.
  3. Implement masking module given parser state; unit tests to guarantee zero invalid choices at each step.
  4. Build graph encoder (NodeFeatureBuilder + GNN + readout) and a tiny decoder (1–2 layers) with masked softmax; smoke-train on a tiny subset to overfit.
  5. Add beam search; evaluate exact reconstruction rate; then add execution tests using examples in `ProgramSpecification`.
  6. Extend coverage to remaining node types and attributes; add small-int and function/attr name vocab heads.

- Evaluation metrics
  - Exact AST reconstruction.
  - Validity (should be 100% with masking).
  - Sequence accuracy/per-step accuracy.
  - Execution accuracy on base and generated examples.

- Future extensions
  - Condition on text/problem statements by swapping encoder to a text encoder and keeping this decoder/masking unchanged.
  - Add denoising objectives (mask subtrees and reconstruct).
  - Add a critic or reranker (GNN over decoded AST) for beam reranking.

Status: Read `dataset/ast.py` and `dataset/programs.py` to align grammar, vocabs, and feasible supervision. Proposed a concrete production-level decoding plan using your existing `GraphEncoder` and `NodeFeatureBuilder`, focusing first on auto-encoding to validate grammar-masked decoding.

- Defined a concise, staged plan for grammar-constrained decoding using `GraphEncoder`/`NodeFeatureBuilder` as input encoder and a Transformer decoder over productions with strict masks.
- Outlined action vocab, parser stack, masking, AST↔sequence converters, supervision, and evaluation; targeting an auto-encoder first for fast validation, then execution checks via `ProgramSpecification` examples.


### Step 1 — Action vocabulary and parser stack roles

- Nonterminal roles (parser stack symbols)
  - Program, Stmt, Expr, Name, Const, ListElts, ArgList, AttrValue

- Productions (Apply one when the top-of-stack role requests it)
  - Program:
    - Program → FunctionDef
  - Stmt:
    - Stmt → Return(Expr)
  - Expr:
    - Expr → Variable
    - Expr → ConstantInt
    - Expr → BinaryOp(Expr, Expr)
    - Expr → List(ListElts)
    - Expr → Call(Expr, ArgList)
    - Expr → Attribute(AttrValue, attr_name)
  - Name:
    - Name → Variable
  - Const:
    - Const → ConstantInt
  - ListElts:
    - ListElts → SetListLen(K) then K× Expr
  - ArgList:
    - ArgList → SetArgLen(K) then K× Expr
  - AttrValue:
    - AttrValue → Expr

- Per-node attribute schema (emitted immediately after the node is introduced, in this order)
  - Variable:
    - SetVarId(var_id) from 0..max_var_vocab_size−1 (use `GraphEncoder.max_var_vocab_size`)
    - Optional: SetCtx(ctx_id) from {Load, Store, Del} if needed later (default Load for Expr)
  - ConstantInt:
    - SetConstExactInt(const_exact_id) from `small_int_vocab` (UNK allowed for OOV)
  - BinaryOp:
    - SetOp(op_id) from `op_vocab`
  - Call:
    - Optional SetFunctionName(fn_id) from `function_vocab` when the callee is a simple name; still expand the callee as an Expr child for structural fidelity
  - Attribute:
    - SetAttributeName(attr_id) from `attribute_vocab`
  - FunctionDef (root):
    - For now: SetFunctionName to constant 'program' and SetParamLen(K) with K capped; params themselves won’t be generated in Step 1 training target (we will focus on auto-encoding the function body’s return expression first; full params later)

- Structural/repetition actions
  - SetListLen(K): K in [0, Kmax_list] (use 8 to match `GraphEncoder.list_first_k`)
  - SetArgLen(K): K in [0, Kmax_args] (cap at 8 initially)

- Special tokens
  - BOS, EOS, PAD
  - NOOP not needed; masking prevents invalid actions

- Sequence format and traversal
  - Depth-first, pre-order expansion with a stack:
    - When you Apply a production, immediately emit its attributes (if any), then push children in reverse order so they are expanded left-to-right.
    - For lists/args: emit SetLen first, then push K child Expr roles (right-to-left).
  - Masking:
    - Valid productions depend on the top-of-stack role (e.g., Expr allows Variable, ConstantInt, BinaryOp, List, Call, Attribute).
    - Attribute heads only active for the current node type; logits for others are masked out.
    - Value spaces for attrs come from `GraphEncoder.vocab_sizes()`:
      - op: `len(op_vocab)`
      - fn: `len(function_vocab)`
      - attr: `len(attribute_vocab)`
      - var: `max_var_vocab_size`
      - const_exact: `len(small_int_vocab)`

- Initial coverage (subset)
  - Cover these `ASTNodeType`: FUNCTION_DEF (minimal), RETURN, VARIABLE, BINARY_OPERATION, CONSTANT (int only), LIST, FUNCTION_CALL, ATTRIBUTE.
  - Comparisons/boolean ops/float/str can be added later as new productions with their own attribute heads.

- Notes on fidelity to `ASTSimplifier.ast_to_graph`
  - Maintain child order to match node creation in `ast_to_graph` (e.g., BinaryOp: left then right; Call: func then args; Attribute: value then attr).
  - `NEXT_SIBLING` edges will be reconstructed from the order of AST edges per parent after decoding.

