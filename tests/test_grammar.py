import types
from typing import List

import pytest

from dataset.grammar import get_cfg, sample_programs, realize_program
from nltk.parse.generate import generate
from nltk import Nonterminal


LEVELS = [
    "ALL",
    "LEVEL1.1",
    "LEVEL1.2",
    "LEVEL2.1",
    "LEVEL2.2",
    "LEVEL3.1",
    "LEVEL3.2",
    "LEVEL4.1",
]


def test_cfg_builds():
    cfg = get_cfg()
    assert cfg is not None
    # Basic sanity: a few known nonterminals exist
    lhs_symbols = {prod.lhs().symbol() for prod in cfg.productions()}
    assert "ALL" in lhs_symbols
    assert "INITIALIZATION" in lhs_symbols
    assert "FOR_LOOP" in lhs_symbols


@pytest.mark.parametrize("level", LEVELS)
def test_generation_has_at_least_one_sentence(level: str):
    cfg = get_cfg()
    start = Nonterminal(level if level == "ALL" else level.replace(".", "_"))
    # Try to enumerate some derivations
    sentences = list(generate(cfg, start=start, depth=20, n=100))
    assert len(sentences) > 0, f"No sentences generated for level {level}"


@pytest.mark.parametrize("level", LEVELS)
def test_sample_programs_exec(level: str):
    programs: List[str] = sample_programs(n=1, level=level, max_depth=25, seed=123)
    assert len(programs) == 1
    code = programs[0]

    # Attempt to execute generated code in a restricted namespace
    local_vars = {}
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        pytest.fail(f"Generated program for {level} failed to execute: {e}\n\n{code}")


