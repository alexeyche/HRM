import importlib
import inspect
from typing import Any, Dict
import multiprocessing as mp
import re as _re
import sys
import logging

log = logging.getLogger(__name__)


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)

    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


# Optimize multiprocessing startup for test helper processes
try:
    if sys.platform != "win32":
        log.info("Setting multiprocessing start method to fork")
        mp.set_start_method("fork", force=True)
    else:
        log.info("Using default multiprocessing start method")
except RuntimeError:
    # Start method may already be set by the runner; ignore
    log.info("Multiprocessing start method already set by the runner; ignoring")




def _worker_run(code_str: str, inp: Any, q: Any) -> None:
    """Top-level worker for multiprocessing (required for spawn on macOS)."""
    try:
        ns: Dict[str, Any] = {}
        exec(code_str, ns)
        m = _re.search(r"def\s+(\w+)\(", code_str)
        func = ns.get("program") or (ns.get(m.group(1)) if m else None)
        if func is None:
            q.put(("err", "function_not_found"))
            return
        if isinstance(inp, list):
            out = func(*inp)
        else:
            out = func(inp)
        q.put(("ok", out))
    except Exception as e:  # pragma: no cover
        q.put(("err", str(e)))


def run_with_timeout(code_str: str, inp: Any, timeout_s: float = 0.1) -> Any:
    ctx = mp.get_context()
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