from dataset.programs import ProgramRegistry, ProgramSpecification
from typing import List, Dict, Any
import random



def _random_identifier_pool() -> List[str]:
    """Generate pool of single-letter variable names compatible with grammar.py"""
    # Only single-letter names to match grammar.py VARIABLE definition
    return [chr(c) for c in range(ord('a'), ord('z') + 1)]


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


def augment_registry(base_registry: ProgramRegistry, num_samples: int, seed: int) -> ProgramRegistry:
    """Create a new registry with code-augmented programs."""

    rng = random.Random(seed)
    augmented_registry = ProgramRegistry()

    names = base_registry.list_names()
    if not names:
        raise RuntimeError("No programs found in base registry")

    # Generate augmented programs
    for idx in range(num_samples):
        # Pick a spec uniformly
        spec_name = rng.choice(names)
        base_spec = base_registry.get(spec_name)
        assert base_spec is not None

        # Augment the code
        augmented_code = _augment_code_parameter_names(
            base_spec.implementation,
            seed=seed + idx
        )

        # Create new spec with augmented code
        aug_spec = ProgramSpecification(
            name=f"{spec_name}_aug_{idx:06d}",
            description=base_spec.description,
            inputs=base_spec.inputs,
            outputs=base_spec.outputs,
            implementation=augmented_code,
            base_examples=base_spec.base_examples
        )

        augmented_registry.register(aug_spec)

    return augmented_registry