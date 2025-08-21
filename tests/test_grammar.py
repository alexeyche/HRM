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


def test_tokenize_all_programs():
    registry = get_program_registry()

    for program_name in registry.list_names():
        program = registry.get(program_name)
        assert program is not None

        log.info(f"Parsing program {program_name}: \n{program.implementation}")

        tokens = tokenize_code(program.implementation)
        log.info(f"Tokens: {tokens}")
        assert len(tokens) > 0, f"Program {program_name} has no tokens"


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