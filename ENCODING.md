
### What to encode
- **Node type (you already have this)**: keep a learned embedding per `ASTNodeType`.
- **Operator kind**: for `BINARY_OPERATION`, `UNARY_OPERATION`, `COMPARISON`, `BOOLEAN_OPERATION`, embed the operator token (e.g., "+", "-", "==", "and") via a small embedding table.
- **Variables (categorical, unbounded names)**:
  - Canonicalize per-program: map each distinct identifier to a small integer `var_id` in order of first appearance; use `var_emb[var_id]` with `var_id=0` as UNK.
  - Add a unique “symbol” node for each distinct variable; connect all occurrences to that symbol node with a dedicated edge type (captures aliasing/renaming invariance).
  - Optionally add read/write flags and scope depth as small scalar features.
- **Constants (typed values)**:
  - Int: mix categorical + continuous:
    - Bucket by magnitude: `bucket = clamp(floor(log10(|x|)), [-B, B])` → bucket embedding.
    - Small scalars: maintain a top-K frequent constants vocabulary (e.g., -1, 0, 1, 2, 10); hits use a learned constant embedding; misses fall back to the bucketed scheme.
    - Continuous head: sign bit, zero bit, parity bit, scaled residual `x / 10^bucket` clamped to [-1,1], optionally `x % {2,3,5}/k` in [0,1].
  - Float: decompose into sign, exponent bucket, mantissa in [-1,1].
  - Bool: 2-way embedding or one-hot.
  - Str: character n-gram hashing or averaged char-embedding; include length bucket.
  - List[int]: length bucket + summary stats (min, max, mean, sum clipped), optionally first-K elements with same int-encoder and masked.
- **Context features**: depth, preorder index, sibling index; optional data-flow edges (assignment uses) and next-sibling edges.
- **Attention**: graph attention often helps when variables/constants repeat (see Graph Attention Auto-Encoders). Linking same-variable occurrences to a symbol node is especially effective (as in “mapping variables between programs”).

References you can follow:
- Learned embeddings and mixing categorical/continuous features: see the GNN representation notes in the ENCCS GNN+Transformers tutorial (handling categorical vs continuous).
- Variable symbol node trick: “Graph Neural Networks For Mapping Variables Between Programs.”
- Node value embeddings: see libraries like `ast-node-encoding` for inspiration on AST node value encoders.


### Turning nodes into vectors (sketch)
- Define embeddings:
  - `type_emb[len(ASTNodeType)]`
  - `op_emb[num_ops]`
  - `var_emb[max_vars+1]`
  - `const_bucket_emb[num_buckets]`
- Define small MLP heads for continuous features per dtype.
- Build node vectors by concatenating:
  - `type_emb[type]`
  - optional `op_emb[op]` (if present)
  - variable head if `type=="variable"`: `var_emb[var_id]` (+ read/write bit)
  - constant head if `type=="constant"`: dtype-specific mix of bucket embedding + scaled numeric features (or char n-gram for strings)
  - positional scalars (depth, sibling index), optionally via sinusoidal or small MLP
- Mask components not applicable to a node type (or use type-specific MLPs with a gating mask).

Very concise PyG collation idea:

```python
# Pseudocode
x = []
for n in nodes:
    h = type_emb[idx_of(n['type'])]
    if 'op' in n:
        h = h + op_emb[idx_of(n['op'])]
    if n['type'] == 'variable':
        h = torch.cat([h, var_emb[n['var_id']], one_hot_rw(n.get('ctx'))])
    elif n['type'] == 'constant':
        h = torch.cat([h, encode_constant(n['dtype'], n['value'])])
    x.append(h)
edge_index, edge_type = build_edge_tensors(edges)  # include 'ast', 'child', 'symbol', 'next_sibling'
```

### Why this works
- Variables: learned embeddings over small per-program IDs + “symbol edges” make the model robust to renaming and let it aggregate all mentions of the same identifier.
- Constants: mixed categorical+continuous encoding preserves arithmetic properties (magnitude, sign, parity) while still learning frequent-token semantics (e.g., 0, 1, -1).
- Operator embeddings and ordering edges help compositional reasoning over expressions.
- Attention heads can prioritize the relevant symbol nodes and reuse across occurrences.
