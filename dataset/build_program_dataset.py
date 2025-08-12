from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Local imports
from dataset.programs import get_program_registry, ProgramSpecification
from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, collate_encoded_graphs
from dataset.grammar_actions import simplified_ast_to_graph, graph_to_actions, Action


def _random_identifier_pool() -> List[str]:
    base = [
        "x", "y", "z", "u", "v", "w",
        "a", "b", "c", "d", "e", "f",
        "i", "j", "k", "m", "n", "t",
        "arr", "vals", "nums", "seq", "data",
    ]
    # Add suffix variants to enlarge pool
    pool: List[str] = []
    for name in base:
        pool.append(name)
        for s in (1, 2, 3):
            pool.append(f"{name}{s}")
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in pool:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _augment_code_parameter_names(code: str, seed: int) -> str:
    """Rename function name, parameters, and simple loop indices deterministically.

    - Only touches the top-level function definition and Name/arg occurrences bound to parameters
    - Optionally renames for-loop indices 'i'/'j' style to reduce name bias
    """
    import ast

    rng = random.Random(seed)
    tree = ast.parse(code)

    # Find top-level function def
    fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = node
            break
    if fn is None:
        return code

    # Build rename mapping for parameters
    param_names = [arg.arg for arg in fn.args.args]
    pool = _random_identifier_pool()
    rng.shuffle(pool)

    # Avoid renaming to existing local names if easily detectable
    existing_names: set[str] = set()
    class LocalNameCollector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> Any:  # type: ignore[override]
            existing_names.add(node.id)
    LocalNameCollector().visit(fn)

    new_names: List[str] = []
    for _ in param_names:
        while pool and pool[0] in existing_names:
            pool.pop(0)
        new_names.append(pool.pop(0) if pool else _)

    rename_map: Dict[str, str] = {old: new for old, new in zip(param_names, new_names)}

    # Also optionally rename simple loop indices (i/j/k) that are not parameters
    loop_index_candidates = ["i", "j", "k"]
    for_target_renames: Dict[str, str] = {}

    class Renamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # type: ignore[override]
            # Rename function itself to "program" or a deterministic variant
            node.name = "program"

            # Rename parameters
            for arg in node.args.args:
                if arg.arg in rename_map:
                    arg.arg = rename_map[arg.arg]
            self.generic_visit(node)
            return node

        def visit_For(self, node: ast.For) -> Any:  # type: ignore[override]
            # Handle simple Name targets only
            if isinstance(node.target, ast.Name) and node.target.id in loop_index_candidates and node.target.id not in rename_map:
                # Assign a new fresh name
                replacement = None
                while pool:
                    cand = pool.pop(0)
                    if cand not in existing_names and cand not in rename_map.values():
                        replacement = cand
                        break
                if replacement is not None:
                    for_target_renames[node.target.id] = replacement
                    node.target.id = replacement

            self.generic_visit(node)
            return node

        def visit_Name(self, node: ast.Name) -> Any:  # type: ignore[override]
            if node.id in rename_map:
                node.id = rename_map[node.id]
            elif node.id in for_target_renames:
                node.id = for_target_renames[node.id]
            return node

    new_tree = Renamer().visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_code = ast.unparse(new_tree)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: if unparse fails, return original
        return code

    return new_code


def _action_to_jsonable(a: Action) -> Dict[str, Any]:
    return {"kind": str(a.kind), "value": a.value}


def generate_program_dataset(output_dir: str, num_samples: int = 200, seed: int = 123) -> None:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Registry of small programs
    registry = get_program_registry()
    names = registry.list_names()
    if not names:
        raise RuntimeError("No programs found in registry")

    # Encodings holder to collate at the end
    encoder = GraphEncoder()
    encoded_graphs = []

    samples_path = Path(output_dir) / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for idx in range(num_samples):
            # Pick a spec uniformly
            spec_name = rng.choice(names)
            spec: ProgramSpecification = registry.get(spec_name)  # type: ignore[assignment]
            assert spec is not None

            base_code = spec.implementation
            # Augment deterministically based on idx and seed
            aug_code = _augment_code_parameter_names(base_code, seed=seed + idx)

            # Build two graphs: full AST for encoding and simplified one for actions
            try:
                ast_graph = ASTSimplifier.ast_to_graph(aug_code)
            except Exception:
                # If reconstruction fails for some edge case, fall back to original code
                ast_graph = ASTSimplifier.ast_to_graph(base_code)

            try:
                simp_graph = simplified_ast_to_graph(aug_code)
            except Exception:
                simp_graph = simplified_ast_to_graph(base_code)

            # Actions
            actions: List[Action] = graph_to_actions(simp_graph)
            actions_json = [_action_to_jsonable(a) for a in actions]

            # Encode for GNN
            enc = encoder.encode(ast_graph)
            encoded_graphs.append(enc)

            # Persist JSONL record (compact for readability)
            record: Dict[str, Any] = {
                "id": idx,
                "program_name": spec_name,
                "code": aug_code,
                "actions": actions_json,
                # For compactness, store counts not full arrays for graphs; graphs are recoverable from code
                "num_nodes": len(ast_graph["nodes"]),
                "num_edges": len(ast_graph["edges"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Collate encodings and save as NPZ for fast training loads
    batched = collate_encoded_graphs(encoded_graphs)
    npz_path = Path(output_dir) / "encoded.npz"
    np.savez_compressed(
        npz_path,
        node_type=batched.node_type,
        op_id=batched.op_id,
        ctx_id=batched.ctx_id,
        dtype_id=batched.dtype_id,
        function_name_id=batched.function_name_id,
        attribute_name_id=batched.attribute_name_id,
        var_id=batched.var_id,
        const_exact_int_id=batched.const_exact_int_id,
        list_firstk_ids=batched.list_firstk_ids,
        list_firstk_mask=batched.list_firstk_mask,
        const_numeric=batched.const_numeric,
        str_numeric=batched.str_numeric,
        list_summary=batched.list_summary,
        position=batched.position,
        edge_index=batched.edge_index,
        edge_type=batched.edge_type,
        graph_id=batched.graph_id,
        node_ptr=batched.node_ptr,
        num_graphs=np.array([batched.num_graphs], dtype=np.int64),
    )

    # Save a tiny manifest
    manifest = {
        "num_samples": num_samples,
        "source_programs": names,
        "samples_jsonl": str(samples_path.name),
        "encoded_npz": str(npz_path.name),
        "notes": "Each JSONL record contains augmented code and action sequence; tensors are in encoded.npz",
    }
    with open(Path(output_dir) / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small program dataset (code, actions, encodings)")
    parser.add_argument("--out", type=str, default="data/programs-200", help="Output directory")
    parser.add_argument("--n", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    generate_program_dataset(args.out, num_samples=args.n, seed=args.seed)
    print(f"Saved dataset to {args.out}")


if __name__ == "__main__":
    main()


