import pytest

from dataset.programs import get_program_registry, test_all_programs as _test_all_programs_impl


def test_all_programs():
    registry = get_program_registry()
    results = _test_all_programs_impl(registry)
    passed = sum(results.values())
    total = len(results)

    assert passed == total, f"Expected {total} programs to pass, but only {passed} passed"

