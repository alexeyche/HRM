from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import ast as pyast
import re
import numpy as np

from .ast import ASTNodeType, EdgeType

# -----------------------------
# Vocabularies
# -----------------------------


class Vocab:
    """Simple string-to-id vocabulary with PAD and UNK.

    id mapping:
    - 0: PAD
    - 1: UNK
    - 2..: tokens
    """

    def __init__(self, tokens: Optional[Iterable[str]] = None) -> None:
        self._tok_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self._id_to_tok: List[str] = ["<PAD>", "<UNK>"]
        if tokens:
            self.add_many(tokens)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self._id_to_tok)

    def add(self, token: str) -> int:
        if token in self._tok_to_id:
            return self._tok_to_id[token]
        idx = len(self._id_to_tok)
        self._tok_to_id[token] = idx
        self._id_to_tok.append(token)
        return idx

    def add_many(self, tokens: Iterable[str]) -> None:
        for t in tokens:
            self.add(t)

    def to_id(self, token: Optional[str]) -> int:
        if token is None:
            return self.unk_id
        return self._tok_to_id.get(token, self.unk_id)

    def to_token(self, idx: int) -> str:
        if 0 <= idx < len(self._id_to_tok):
            return self._id_to_tok[idx]
        return "<UNK>"


def default_operator_vocab() -> Vocab:
    # Unified operator tokens across node types
    ops = [
        # binary
        "+",
        "-",
        "*",
        "/",
        "%",
        "**",
        # unary
        "not",
        # comparisons
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        # boolean
        "and",
        "or",
        # augmented assignment (AST stores op separately but we reuse same tokens)
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "**=",
    ]
    return Vocab(ops)


def scan_program_implementations_for_names() -> Tuple[Set[str], Set[str], Set[str], Set[int]]:
    """Scan dataset.programs for function names, attribute names, string literals, and frequent ints.

    Returns: (functions, attributes, string_literals, int_literals)
    """
    functions: Set[str] = set()
    attributes: Set[str] = set()
    strings: Set[str] = set()
    ints: Set[int] = set()

    try:
        from .programs import DEFAULT_REGISTRY

        for spec in DEFAULT_REGISTRY.programs.values():  # type: ignore[attr-defined]
            impl = spec.implementation
            try:
                mod = pyast.parse(impl)
            except Exception:
                continue

            class Visitor(pyast.NodeVisitor):
                def visit_Call(self, node: pyast.Call) -> Any:  # type: ignore[override]
                    # function names (builtins or Name nodes)
                    if isinstance(node.func, pyast.Name):
                        functions.add(node.func.id)
                    self.generic_visit(node)

                def visit_Attribute(self, node: pyast.Attribute) -> Any:  # type: ignore[override]
                    attributes.add(node.attr)
                    self.generic_visit(node)

                def visit_Constant(self, node: pyast.Constant) -> Any:  # type: ignore[override]
                    val = node.value
                    if isinstance(val, str):
                        strings.add(val)
                    elif isinstance(val, int):
                        ints.add(val)

            Visitor().visit(mod)
    except Exception:
        # Fallback minimal seeds
        functions.update(["range", "sum", "max", "min", "len", "int", "str", "sorted", "set", "abs"])
        attributes.update(["upper", "lower"])
        strings.update(["", "aeiouAEIOU"])
        ints.update([-1, 0, 1, 2, 3, 4, 5, 7, 10, 42, 100, 1000])

    # Ensure some common tokens present
    if not strings:
        strings.update(["", "aeiouAEIOU"])
    if not ints:
        ints.update([-1, 0, 1, 2, 3, 4, 5, 7, 10, 42])

    return functions, attributes, strings, ints


def build_default_vocabs_from_programs() -> Tuple[Vocab, Vocab, Vocab, Vocab]:
    """Return (operator_vocab, function_name_vocab, attribute_name_vocab, small_int_vocab)."""
    op_vocab = default_operator_vocab()
    fn_names, attr_names, str_lits, int_lits = scan_program_implementations_for_names()
    fn_vocab = Vocab(sorted(fn_names))
    attr_vocab = Vocab(sorted(attr_names))
    # Represent small exact ints as strings in a Vocab for a compact id space
    small_int_vocab = Vocab([str(v) for v in sorted(int_lits)])
    # Also create a string literal vocab users can optionally use
    _ = Vocab(sorted(str_lits))  # not returned for now; we keep numeric/string features instead of IDs
    return op_vocab, fn_vocab, attr_vocab, small_int_vocab


# -----------------------------
# Positional utilities
# -----------------------------


def _build_children_lists(num_nodes: int, edges: List[Tuple[int, int, EdgeType]]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for src, dst, et in edges:
        if et == EdgeType.AST:
            children[src].append(dst)
    return children


def _compute_depth_and_order(root: int, children: Dict[int, List[int]]) -> Tuple[List[int], List[int]]:
    depth = [-1] * len(children)
    preorder: List[int] = [-1] * len(children)
    counter = 0

    def dfs(u: int, d: int) -> None:
        nonlocal counter
        depth[u] = d
        preorder[u] = counter
        counter += 1
        for v in children[u]:
            dfs(v, d + 1)

    dfs(root, 0)
    return depth, preorder


def _compute_sibling_index(children: Dict[int, List[int]]) -> List[int]:
    sibling_index = [-1] * len(children)
    for parent, ch in children.items():
        for idx, node_id in enumerate(ch):
            sibling_index[node_id] = idx
    return sibling_index


# -----------------------------
# Constant feature encoding
# -----------------------------


def _int_features(x: int, small_int_vocab: Vocab, bucket_bound: int = 9) -> Tuple[int, List[float]]:
    # exact id if in vocab
    exact_id = small_int_vocab.to_id(str(x))
    if exact_id == small_int_vocab.unk_id:
        # treat as out-of-vocab exact; keep features only
        exact_id = -1
    # magnitude bucket via log10(|x|)
    if x == 0:
        magnitude = 0
        residual = 0.0
    else:
        import math

        magnitude = int(math.floor(math.log10(abs(x))))
        magnitude = max(-bucket_bound, min(bucket_bound, magnitude))
        residual = x / float(10 ** magnitude)
        residual = max(-1.0, min(1.0, residual))
    sign_bit = 1.0 if x < 0 else 0.0
    zero_bit = 1.0 if x == 0 else 0.0
    parity_bit = 1.0 if (x % 2 == 0) else 0.0
    # normalize magnitude to [-1,1]
    mag_norm = magnitude / float(bucket_bound if bucket_bound > 0 else 1)
    features = [sign_bit, zero_bit, parity_bit, mag_norm, float(residual)]
    return exact_id, features


def _float_features(x: float, bucket_bound: int = 9) -> List[float]:
    import math

    if x == 0.0:
        return [0.0, 1.0, 0.0, 0.0]  # sign, zero, exp_norm, mantissa
    sign = 1.0 if x < 0 else 0.0
    m, e = math.frexp(abs(x))  # x = m * 2**e, m in [0.5,1)
    # bucket exponent by base-10 order approximately
    e10 = int(math.floor(math.log10(abs(x))))
    e10 = max(-bucket_bound, min(bucket_bound, e10))
    exp_norm = e10 / float(bucket_bound)
    mantissa = max(0.0, min(1.0, (abs(x) / (10 ** e10)) if e10 != 0 else abs(x)))
    return [sign, 0.0, exp_norm, mantissa]


def _str_features(s: str, max_len_bucket: int = 32) -> List[float]:
    length = len(s)
    # simple length bucket normalized
    lb = min(length, max_len_bucket)
    return [lb / float(max_len_bucket)]


def _list_int_features(values: List[int], small_int_vocab: Vocab, first_k: int = 8) -> Tuple[List[int], List[float], List[int]]:
    length = len(values)
    lb = min(length, 64)  # clip length bucket
    if length == 0:
        return [-1] * first_k, [0.0, 0.0, 0.0, lb / 64.0], [0] * first_k
    vmin = min(values)
    vmax = max(values)
    mean = sum(values) / float(length)
    vsum = sum(values)
    # normalize/clamp summary
    def clampf(x: float, bound: float = 1e6) -> float:
        x = max(-bound, min(bound, x))
        return float(x)

    summary = [
        clampf(vmin / 1000.0),
        clampf(vmax / 1000.0),
        clampf(mean / 1000.0),
        lb / 64.0,
    ]
    # first-K exact ids through small-int vocab (or -1)
    first_ids: List[int] = []
    mask: List[int] = []
    for i in range(first_k):
        if i < length:
            vid = small_int_vocab.to_id(str(values[i]))
            if vid == small_int_vocab.unk_id:
                vid = -1
            first_ids.append(vid)
            mask.append(1)
        else:
            first_ids.append(-1)
            mask.append(0)
    return first_ids, summary, mask


# -----------------------------
# Encoded graph container
# -----------------------------


@dataclass
class EncodedGraph:
    # Node-level categorical ids
    node_type: np.ndarray  # [N] int64
    op_id: np.ndarray  # [N] int64, -1 if N/A
    ctx_id: np.ndarray  # [N] int64, -1 if N/A
    dtype_id: np.ndarray  # [N] int64, -1 if N/A
    function_name_id: np.ndarray  # [N] int64, -1 if N/A
    attribute_name_id: np.ndarray  # [N] int64, -1 if N/A
    var_id: np.ndarray  # [N] int64, 0 is UNK for non-variable nodes too
    const_exact_int_id: np.ndarray  # [N] int64, -1 if N/A
    list_firstk_ids: np.ndarray  # [N, K] int64, -1 padded
    list_firstk_mask: np.ndarray  # [N, K] int64, 0/1

    # Node-level numeric features
    const_numeric: np.ndarray  # [N, 5] float32
    str_numeric: np.ndarray  # [N, 1] float32
    list_summary: np.ndarray  # [N, 4] float32
    position: np.ndarray  # [N, 3] float32 -> depth, preorder, sibling_idx (scaled 0..1)

    # Edges
    edge_index: np.ndarray  # [2, E] int64
    edge_type: np.ndarray  # [E] int64

    # Book-keeping
    root: int


class GraphEncoder:
    def __init__(
        self,
        operator_vocab: Optional[Vocab] = None,
        function_name_vocab: Optional[Vocab] = None,
        attribute_name_vocab: Optional[Vocab] = None,
        small_int_vocab: Optional[Vocab] = None,
        list_first_k: int = 8,
        max_var_vocab_size: int = 512,
    ) -> None:
        self.op_vocab = operator_vocab or default_operator_vocab()
        if function_name_vocab is None or attribute_name_vocab is None or small_int_vocab is None:
            _, fnv, av, siv = build_default_vocabs_from_programs()
            self.function_vocab = function_name_vocab or fnv
            self.attribute_vocab = attribute_name_vocab or av
            self.small_int_vocab = small_int_vocab or siv
        else:
            self.function_vocab = function_name_vocab
            self.attribute_vocab = attribute_name_vocab
            self.small_int_vocab = small_int_vocab
        self.list_first_k = list_first_k
        self.max_var_vocab_size = max_var_vocab_size

        # ctx mapping for variable read/write
        self.ctx_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "Load": 2, "Store": 3, "Del": 4}

        # dtype mapping for constants
        self.dtype_to_id: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "bool": 2,
            "int": 3,
            "float": 4,
            "str": 5,
        }

        self.edge_type_to_id: Dict[EdgeType, int] = {
            EdgeType.AST: 0,
            EdgeType.SYMBOL: 1,
            EdgeType.NEXT_SIBLING: 2,
        }

    def vocab_sizes(self) -> Dict[str, int]:
        """Return sizes for all categorical vocabularies, aligned with encoder logic."""
        return {
            "node_type": len(ASTNodeType),
            "op": len(self.op_vocab),
            "ctx": len(self.ctx_to_id),
            "dtype": len(self.dtype_to_id),
            "fn": len(self.function_vocab),
            "attr": len(self.attribute_vocab),
            # var_id space is per-graph dynamic; provide a safe cap configured in the encoder
            "var": self.max_var_vocab_size,
            "const_exact": len(self.small_int_vocab),
        }

    def encode(self, graph: Dict[str, Any]) -> EncodedGraph:
        nodes: List[Dict[str, Any]] = graph["nodes"]
        edges: List[Tuple[int, int, EdgeType]] = graph["edges"]
        root: int = graph["root"]

        num_nodes = len(nodes)
        children = _build_children_lists(num_nodes, edges)
        depth, preorder = _compute_depth_and_order(root, children)
        sibling_index = _compute_sibling_index(children)

        # arrays to fill (pure Python lists)
        node_type_ids = [-1 for _ in range(num_nodes)]
        op_ids = [-1 for _ in range(num_nodes)]
        ctx_ids = [-1 for _ in range(num_nodes)]
        dtype_ids = [-1 for _ in range(num_nodes)]
        fn_ids = [-1 for _ in range(num_nodes)]
        attr_ids = [-1 for _ in range(num_nodes)]
        var_ids = [0 for _ in range(num_nodes)]  # 0 is UNK
        const_exact_ids = [-1 for _ in range(num_nodes)]
        list_firstk_ids = [[-1 for _ in range(self.list_first_k)] for _ in range(num_nodes)]
        list_firstk_mask = [[0 for _ in range(self.list_first_k)] for _ in range(num_nodes)]

        const_numeric = [[0.0 for _ in range(5)] for _ in range(num_nodes)]  # int: 5 features; float: 4 (+ one spare)
        str_numeric = [[0.0] for _ in range(num_nodes)]
        list_summary = [[0.0 for _ in range(4)] for _ in range(num_nodes)]
        position = [[0.0, 0.0, 0.0] for _ in range(num_nodes)]

        # map ASTNodeType to stable ids (use enum order)
        node_type_to_id: Dict[ASTNodeType, int] = {nt: i for i, nt in enumerate(ASTNodeType)}

        for i, n in enumerate(nodes):
            ntype: ASTNodeType = n["type"]
            node_type_ids[i] = node_type_to_id[ntype]

            # position
            d = depth[i] if depth[i] >= 0 else 0
            pre = preorder[i] if preorder[i] >= 0 else 0
            sib = sibling_index[i] if sibling_index[i] >= 0 else 0
            # scale roughly to 0..1 using per-graph maxima (avoid divide-by-zero later)
            position[i] = [float(d), float(pre), float(sib)]

            if ntype in (ASTNodeType.BINARY_OPERATION, ASTNodeType.UNARY_OPERATION, ASTNodeType.BOOLEAN_OPERATION):
                token = n.get("op")
                op_ids[i] = self.op_vocab.to_id(token)
            elif ntype == ASTNodeType.COMPARISON:
                token = n.get("op")
                if token is not None:
                    op_ids[i] = self.op_vocab.to_id(token)
                else:
                    # Multiple ops stored under 'ops'; keep the first as op_id for simplicity
                    ops_list = n.get("ops", [])
                    if ops_list:
                        op_ids[i] = self.op_vocab.to_id(ops_list[0])

            if ntype == ASTNodeType.VARIABLE:
                # var_id already canonicalized per-program
                var_ids[i] = int(n.get("var_id", 0))
                ctx = str(n.get("ctx", "<UNK>"))
                ctx_ids[i] = self.ctx_to_id.get(ctx, self.ctx_to_id["<UNK>"])

            if ntype == ASTNodeType.FUNCTION_CALL:
                fname = n.get("function")
                fn_ids[i] = self.function_vocab.to_id(fname)

            if ntype == ASTNodeType.ATTRIBUTE:
                attr = n.get("attr")
                attr_ids[i] = self.attribute_vocab.to_id(attr)

            if ntype == ASTNodeType.CONSTANT:
                dtype = str(n.get("dtype", "<UNK>"))
                dtype_ids[i] = self.dtype_to_id.get(dtype, self.dtype_to_id["<UNK>"])
                val = n.get("value", None)
                if dtype == "int" and isinstance(val, int):
                    ex_id, feats = _int_features(val, self.small_int_vocab)
                    const_exact_ids[i] = ex_id if ex_id >= 0 else -1
                    for j, v in enumerate(feats):
                        const_numeric[i][j] = float(v)
                elif dtype == "float" and isinstance(val, (float, int)):
                    feats = _float_features(float(val))
                    for j, v in enumerate(feats):
                        const_numeric[i][j] = float(v)
                elif dtype == "bool" and isinstance(val, bool):
                    # encode as [is_true, is_false, parity, mag, residual] placeholder bits
                    const_numeric[i][0] = 1.0 if val else 0.0
                    const_numeric[i][1] = 0.0 if val else 1.0
                elif dtype == "str" and isinstance(val, str):
                    feats = _str_features(val)
                    str_numeric[i][0] = float(feats[0])

            if ntype == ASTNodeType.LIST:
                # Attempt to reconstruct immediate child constants under this list to approximate values when possible
                # This is a best-effort; the true values should be recovered by decoding structure and child constants
                # Here we only provide helpful features
                # The actual elements live as child nodes connected by AST edges; we cannot easily gather just ints without walking children
                # So we compute features by scanning children indices
                int_vals: List[int] = []
                for src, dst, et in edges:
                    if et == EdgeType.AST and src == i:
                        child = nodes[dst]
                        if child.get("type") == ASTNodeType.CONSTANT and child.get("dtype") == "int":
                            v = child.get("value")
                            if isinstance(v, int):
                                int_vals.append(v)
                first_ids, summary, mask = _list_int_features(int_vals, self.small_int_vocab, first_k=self.list_first_k)
                for j, v in enumerate(first_ids):
                    list_firstk_ids[i][j] = int(v)
                for j, v in enumerate(mask):
                    list_firstk_mask[i][j] = int(v)
                for j, v in enumerate(summary):
                    list_summary[i][j] = float(v)

        # edge tensors
        edge_index = [[0 for _ in range(len(edges))], [0 for _ in range(len(edges))]]
        edge_type = [0 for _ in range(len(edges))]
        for ei, (src, dst, et) in enumerate(edges):
            edge_index[0][ei] = int(src)
            edge_index[1][ei] = int(dst)
            edge_type[ei] = int(self.edge_type_to_id[et])

        # scale position columns to [0,1] per-graph
        for col in range(3):
            maxv = max(position[row][col] for row in range(num_nodes)) if num_nodes > 0 else 0.0
            if maxv > 0:
                for row in range(num_nodes):
                    position[row][col] = position[row][col] / maxv

        # Convert to numpy arrays with consistent dtypes
        node_type_arr = np.asarray(node_type_ids, dtype=np.int64)
        op_id_arr = np.asarray(op_ids, dtype=np.int64)
        ctx_id_arr = np.asarray(ctx_ids, dtype=np.int64)
        dtype_id_arr = np.asarray(dtype_ids, dtype=np.int64)
        fn_id_arr = np.asarray(fn_ids, dtype=np.int64)
        attr_id_arr = np.asarray(attr_ids, dtype=np.int64)
        var_id_arr = np.asarray(var_ids, dtype=np.int64)
        const_exact_arr = np.asarray(const_exact_ids, dtype=np.int64)
        list_firstk_ids_arr = np.asarray(list_firstk_ids, dtype=np.int64)
        list_firstk_mask_arr = np.asarray(list_firstk_mask, dtype=np.int64)
        const_numeric_arr = np.asarray(const_numeric, dtype=np.float32)
        str_numeric_arr = np.asarray(str_numeric, dtype=np.float32)
        list_summary_arr = np.asarray(list_summary, dtype=np.float32)
        position_arr = np.asarray(position, dtype=np.float32)
        edge_index_arr = np.asarray(edge_index, dtype=np.int64)
        edge_type_arr = np.asarray(edge_type, dtype=np.int64)

        return EncodedGraph(
            node_type=node_type_arr,
            op_id=op_id_arr,
            ctx_id=ctx_id_arr,
            dtype_id=dtype_id_arr,
            function_name_id=fn_id_arr,
            attribute_name_id=attr_id_arr,
            var_id=var_id_arr,
            const_exact_int_id=const_exact_arr,
            list_firstk_ids=list_firstk_ids_arr,
            list_firstk_mask=list_firstk_mask_arr,
            const_numeric=const_numeric_arr,
            str_numeric=str_numeric_arr,
            list_summary=list_summary_arr,
            position=position_arr,
            edge_index=edge_index_arr,
            edge_type=edge_type_arr,
            root=root,
        )


__all__ = [
    "Vocab",
    "default_operator_vocab",
    "build_default_vocabs_from_programs",
    "EncodedGraph",
    "GraphEncoder",
    "BatchedEncodedGraph",
    "collate_encoded_graphs",
    "to_torch_dict",
    "to_pyg_data",
    "collate_to_pyg_batch",
    "NodeFeatureBuilder",
]


# -----------------------------
# Batching utilities
# -----------------------------


@dataclass
class BatchedEncodedGraph:
    # Concatenated node features
    node_type: np.ndarray  # [N]
    op_id: np.ndarray  # [N]
    ctx_id: np.ndarray  # [N]
    dtype_id: np.ndarray  # [N]
    function_name_id: np.ndarray  # [N]
    attribute_name_id: np.ndarray  # [N]
    var_id: np.ndarray  # [N]
    const_exact_int_id: np.ndarray  # [N]
    list_firstk_ids: np.ndarray  # [N, K]
    list_firstk_mask: np.ndarray  # [N, K]
    const_numeric: np.ndarray  # [N, 5]
    str_numeric: np.ndarray  # [N, 1]
    list_summary: np.ndarray  # [N, 4]
    position: np.ndarray  # [N, 3]

    # Edges
    edge_index: np.ndarray  # [2, E]
    edge_type: np.ndarray  # [E]

    # Graph bookkeeping
    graph_id: np.ndarray  # [N] which graph each node belongs to
    node_ptr: np.ndarray  # [G+1] cumulative node offsets
    num_graphs: int


def collate_encoded_graphs(graphs: List[EncodedGraph]) -> BatchedEncodedGraph:
    if not graphs:
        raise ValueError("No graphs provided to collate")

    num_graphs = len(graphs)
    node_counts = [g.node_type.shape[0] for g in graphs]
    # Ensure consistent K across graphs
    k_first = graphs[0].list_firstk_ids.shape[1]
    for g in graphs:
        if g.list_firstk_ids.shape[1] != k_first:
            raise ValueError("Inconsistent list_firstk_ids width across graphs")

    # Concatenate node-level arrays
    node_type = np.concatenate([g.node_type for g in graphs], axis=0)
    op_id = np.concatenate([g.op_id for g in graphs], axis=0)
    ctx_id = np.concatenate([g.ctx_id for g in graphs], axis=0)
    dtype_id = np.concatenate([g.dtype_id for g in graphs], axis=0)
    function_name_id = np.concatenate([g.function_name_id for g in graphs], axis=0)
    attribute_name_id = np.concatenate([g.attribute_name_id for g in graphs], axis=0)
    var_id = np.concatenate([g.var_id for g in graphs], axis=0)
    const_exact_int_id = np.concatenate([g.const_exact_int_id for g in graphs], axis=0)
    list_firstk_ids = np.concatenate([g.list_firstk_ids for g in graphs], axis=0)
    list_firstk_mask = np.concatenate([g.list_firstk_mask for g in graphs], axis=0)
    const_numeric = np.concatenate([g.const_numeric for g in graphs], axis=0)
    str_numeric = np.concatenate([g.str_numeric for g in graphs], axis=0)
    list_summary = np.concatenate([g.list_summary for g in graphs], axis=0)
    position = np.concatenate([g.position for g in graphs], axis=0)

    # Reindex and concatenate edges
    total_edges = sum(g.edge_index.shape[1] for g in graphs)
    edge_index = np.zeros((2, total_edges), dtype=np.int64)
    edge_type = np.zeros((total_edges,), dtype=np.int64)

    node_ptr = np.zeros((num_graphs + 1,), dtype=np.int64)
    for i in range(num_graphs):
        node_ptr[i + 1] = node_ptr[i] + node_counts[i]

    e_cursor = 0
    for gi, g in enumerate(graphs):
        offset = node_ptr[gi]
        e_count = g.edge_index.shape[1]
        if e_count > 0:
            edge_index[:, e_cursor : e_cursor + e_count] = g.edge_index + offset
            edge_type[e_cursor : e_cursor + e_count] = g.edge_type
            e_cursor += e_count

    graph_id = np.concatenate([
        np.full((n,), gi, dtype=np.int64) for gi, n in enumerate(node_counts)
    ])

    return BatchedEncodedGraph(
        node_type=node_type,
        op_id=op_id,
        ctx_id=ctx_id,
        dtype_id=dtype_id,
        function_name_id=function_name_id,
        attribute_name_id=attribute_name_id,
        var_id=var_id,
        const_exact_int_id=const_exact_int_id,
        list_firstk_ids=list_firstk_ids,
        list_firstk_mask=list_firstk_mask,
        const_numeric=const_numeric,
        str_numeric=str_numeric,
        list_summary=list_summary,
        position=position,
        edge_index=edge_index,
        edge_type=edge_type,
        graph_id=graph_id,
        node_ptr=node_ptr,
        num_graphs=num_graphs,
    )


def to_torch_dict(obj: Any, device: Optional[str] = None, non_blocking: bool = False) -> Dict[str, Any]:
    """Convert EncodedGraph or BatchedEncodedGraph numpy arrays to a dict of torch tensors.

    Keeps python ints as-is (e.g., root, num_graphs).
    """
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyTorch is required for to_torch_dict but is not available") from e

    tensors: Dict[str, Any] = {}
    for key, value in obj.__dict__.items():
        if isinstance(value, np.ndarray):
            tensors[key] = torch.as_tensor(value, device=device)
        else:
            tensors[key] = value
    return tensors


def to_pyg_data(g: EncodedGraph, device: Optional[str] = None) -> Any:
    """Convert an EncodedGraph to torch_geometric.data.Data.

    - x is not constructed here; you can build model-specific node features from fields.
    - Stores node/edge categorical and numeric arrays as tensors on Data.
    """

    td = to_torch_dict(g, device=device)
    try:
        from torch_geometric.data import Data  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("torch_geometric is required for to_pyg_data") from e
    data = Data()
    # Required for PyG
    data.num_nodes = int(td["node_type"].shape[0])
    data.edge_index = td["edge_index"]
    data.edge_type = td["edge_type"]

    # Node attributes
    for key in [
        "node_type",
        "op_id",
        "ctx_id",
        "dtype_id",
        "function_name_id",
        "attribute_name_id",
        "var_id",
        "const_exact_int_id",
        "list_firstk_ids",
        "list_firstk_mask",
        "const_numeric",
        "str_numeric",
        "list_summary",
        "position",
    ]:
        setattr(data, key, td[key])

    data.root = g.root  # keep as python int meta
    return data


def collate_to_pyg_batch(graphs: List[EncodedGraph], device: Optional[str] = None) -> Any:
    """Collate a list of EncodedGraph into a torch_geometric.data.Batch via Data.list -> Batch.from_data_list.
    Reindexes edges using PyG utilities and builds batch vector.
    """
    try:
        from torch_geometric.data import Batch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("torch_geometric is required for collate_to_pyg_batch") from e

    data_list = [to_pyg_data(g, device=device) for g in graphs]
    batch = Batch.from_data_list(data_list, follow_batch=[])
    return batch
