#!/usr/bin/env python3
"""
Test suite for AST to Python code reconstruction functionality.
Tests the complete pipeline: code -> AST -> reconstructed code -> execution
"""

from typing import Dict, Any
import multiprocessing as _mp
import re as _re
import logging
from dataset.ast import ASTSimplifier
from dataset.cfg import CFGCodeGenerator

log = logging.getLogger(__name__)

def _worker_run(code_str: str, inp: Any, q: Any) -> None:
    try:
        ns: Dict[str, Any] = {}
        exec(code_str, ns)
        m = _re.search(r"def\s+(\w+)\(", code_str)
        # Prefer canonical name 'program' if available; otherwise fallback to first def
        func = ns.get("program") or (ns.get(m.group(1)) if m else None)
        if func is None:
            q.put(("err", "function_not_found"))
            return
        if isinstance(inp, list):
            out = func(*inp)
        else:
            out = func(inp)
        q.put(("ok", out))
    except Exception as e:  # pragma: no cover - surfaced in parent
        q.put(("err", str(e)))


def run_with_timeout(code_str: str, inp: Any, timeout_s: float = 0.1) -> Any:
    # Reuse a lighter-weight process start if available (fork is set in conftest)
    ctx = _mp.get_context()
    q: Any = ctx.Queue()
    p = ctx.Process(target=_worker_run, args=(code_str, inp, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(0.1)
        raise TimeoutError("execution timed out")
    if q.empty():
        raise RuntimeError("no result returned")
    tag, payload = q.get()
    if tag == "ok":
        return payload
    raise RuntimeError(payload)


def test_registry_programs_roundtrip():
    """Roundtrip all registry program implementations through the mini-language pipeline.
    """
    from dataset.programs import get_program_registry

    registry = get_program_registry()

    for name, spec in registry.programs.items():
        code = spec.implementation
        log.info(f"Testing program '{name}'")
        # Build simplified graph compatible with grammar actions

        ast_graph = ASTSimplifier.ast_to_graph(code)

        # Generate code using CFG
        generator = CFGCodeGenerator()
        reconstructed_code = generator.generate_code_from_ast_graph(ast_graph)

        log.info(f"Final code:\n{reconstructed_code}")
        # The reconstructed code should be syntactically valid Python
        try:
            import ast as _pyast
            _pyast.parse(reconstructed_code)
        except SyntaxError as e:
            import pytest
            pytest.fail(f"Reconstructed code not parseable for program '{name}': {e}\nCode:\n{reconstructed_code}")

        # Execute reconstructed program against all base examples with a timeout to avoid infinite loops
        for ex in spec.base_examples:
            try:
                result = run_with_timeout(reconstructed_code, ex.input, timeout_s=1.0)
            except TimeoutError as te:
                import pytest
                pytest.fail(f"Program '{name}' timed out for input {ex.input} within 1s: {te}\nCode:\n{reconstructed_code}")
            assert result == ex.output, f"Program '{name}' failed example {ex.input}: expected {ex.output}, got {result}\nCode:\n{reconstructed_code}"
            log.info(f"\tTesting input {ex.input} ✅")
