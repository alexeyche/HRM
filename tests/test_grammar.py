from dataset.grammar import get_cfg, get_parser, sample_programs, parse_program_with_ast, parse_tokens_to_productions
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
    parser = get_parser()

    code = """
    def program(a, b):
        if a < b:
            return a
        else:
            return b
    """
    tokens = tokenize_code(code)
    result = list(parser.parse(tokens))
    assert len(result) > 0, "No parse trees found"
    for tree in result:
        assert tree is not None
        assert tree.height() > 0, "Parse tree has no height"


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
    parser = get_parser()

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


def test_parse_tokens_to_productions_all_programs():
    cfg = get_cfg()
    registry = get_program_registry()

    failed_programs = []
    for program_name in registry.list_names():
        program = registry.get(program_name)
        assert program is not None

        # log.info(f"Testing parse_tokens_to_productions for program {program_name}: \n{program.implementation}")

        tokens = tokenize_code(program.implementation)
        # log.info(f"Tokens: {tokens}")

        # Call parse_tokens_to_productions and check results
        try:
            production_sequence, terminal_requirements = parse_tokens_to_productions(tokens, cfg)
        except ValueError as e:
            failed_programs.append(f"{program_name}: \n{program.implementation}\n{e}\n")
            continue

        # log.info(f"Production sequence length: {len(production_sequence)}")
        # log.info(f"Terminal requirements length: {len(terminal_requirements)}")

        # Basic assertions
        assert isinstance(production_sequence, list), f"Production sequence should be a list for {program_name}"
        assert isinstance(terminal_requirements, list), f"Terminal requirements should be a list for {program_name}"
        assert len(production_sequence) > 0, f"Production sequence should not be empty for {program_name}"
        assert len(terminal_requirements) > 0, f"Terminal requirements should not be empty for {program_name}"

        # Check that production sequence contains valid tuples
        for i, (nonterminal, prod_idx) in enumerate(production_sequence):
            assert hasattr(nonterminal, 'symbol'), f"First element should be a Nonterminal at position {i} for {program_name}"
            assert isinstance(prod_idx, int), f"Second element should be an int at position {i} for {program_name}"
            assert prod_idx >= 0, f"Production index should be non-negative at position {i} for {program_name}"

        # Check that terminal requirements contain valid tuples
        for i, (terminal_type, target_value) in enumerate(terminal_requirements):
            assert isinstance(terminal_type, str), f"Terminal type should be a string at position {i} for {program_name}"
            assert isinstance(target_value, str), f"Target value should be a string at position {i} for {program_name}"
            assert terminal_type != "UNKNOWN", f"Terminal type should not be UNKNOWN at position {i} for {program_name}, got '{target_value}'"

        # log.info(f"✓ Successfully parsed {program_name} with {len(production_sequence)} productions and {len(terminal_requirements)} terminals")

    assert len(failed_programs) == 0, f"Failed to parse {len(failed_programs)} programs: \n{''.join(failed_programs)}"