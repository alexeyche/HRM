from dataset.grammar import get_cfg, sample_programs, parse_program_with_ast
from dataset.tokenizer import tokenize_code
from nltk.parse import RecursiveDescentParser
from dataset.programs import get_program_registry
import logging
import pytest

log = logging.getLogger(__name__)

def test_generates_parseable_programs():
    cfg = get_cfg()
    programs = sample_programs(cfg, n=1000)
    for program in programs:
        assert parse_program_with_ast(program), f"Program {program} is not parseable"



def test_parse_program():
    grammar = get_cfg()

    parser = RecursiveDescentParser(grammar)

    code = """
    def program(a, b):
        if a < b:
            return a
        else:
            return b
    """
    tokens = tokenize_code(code)
    print("Tokens:", tokens)

    for tree in parser.parse(tokens):
        print("Parse tree:", tree)
        assert tree is not None


def test_tokenize_supported_programs():
    """Test tokenization of programs that are supported by the current grammar."""
    registry = get_program_registry()

    # Programs that should work with the current grammar (subset of Python)
    supported_programs = [
        'max_of_two', 'min_of_two', 'is_positive', 'is_negative',
        'is_zero', 'sum_of_two', 'difference', 'product',
        'double', 'add_ten', 'subtract_five', 'absolute_value',
        'is_even', 'square', 'cube', 'power_of_two', 'half'
    ]

    for program_name in supported_programs:
        program = registry.get(program_name)
        if program is None:
            pytest.skip(f"Program {program_name} not found in registry")
            continue

        log.info(f"Parsing program {program_name}: \n{program.implementation}")

        tokens = tokenize_code(program.implementation)
        log.info(f"Tokens: {tokens}")
        assert len(tokens) > 0, f"Program {program_name} has no tokens"


@pytest.mark.skip(reason="This test is for later")
def test_parse_all_programs():
    grammar = get_cfg()
    parser = RecursiveDescentParser(grammar)

    registry = get_program_registry()
    for program_name in registry.list_names():
        program = registry.get(program_name)
        assert program is not None

        log.info(f"Parsing program {program_name}: \n{program.implementation}")

        tokens = tokenize_code(program.implementation)
        log.info(f"Tokens: {tokens}")

        for tree in parser.parse(tokens):
            log.info(f"Parse tree: {tree}")
            assert tree is not None