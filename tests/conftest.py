import os
import sys
import logging
import multiprocessing as _mp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _add_project_root_to_path() -> None:
    # tests/ -> project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_add_project_root_to_path()






# Optimize multiprocessing startup for test helper processes
try:
    if sys.platform != "win32":
        _mp.set_start_method("fork", force=True)
except RuntimeError:
    # Start method may already be set by the runner; ignore
    pass
