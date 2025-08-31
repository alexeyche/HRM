from dataset.grammar import get_cfg, get_parser, sample_programs, parse_program_with_ast, parse_tokens_to_productions
from dataset.tokenizer import tokenize_code
from nltk.parse import RecursiveDescentParser
from dataset.programs import get_program_registry
from dataset.augment_programs import augment_registry
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

        # log.info(f"Parsing program {program_name}: \n{program.implementation}")

        tokens = tokenize_code(program.implementation)
        # log.info(f"Tokens: {tokens}")
        assert len(tokens) > 0, f"Program {program_name} has no tokens"


def test_parse_all_programs():
    parser = get_parser()

    failed_programs = []
    registry = get_program_registry()
    for program_name in registry.list_names():
        program = registry.get(program_name)
        assert program is not None

        # log.info(f"Parsing program {program_name}: \n{program.implementation}")

        tokens = tokenize_code(program.implementation)
        # log.info(f"Tokens: {tokens}")

        try:
            for tree in parser.parse(tokens):
                # log.info(f"Parse tree: {tree}")
                assert tree is not None
        except ValueError as e:
            failed_programs.append(f"{program_name}: \n{program.implementation}\n{e}\n")
            continue

    assert len(failed_programs) == 0, f"Failed to parse {len(failed_programs)} programs: \n{''.join(failed_programs)}"


def test_parse_tokens_to_productions_all_programs():
    cfg = get_cfg()
    registry = augment_registry(get_program_registry(), num_samples=500, seed=42)
    # registry = get_program_registry()

    failed_programs = []
    production_sequence_length = 0.0
    tokens_length = 0.0
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

        # log.info(f"Production sequence length: {len(production_sequence)}")
        production_sequence_length += len(production_sequence)
        tokens_length += len(tokens)

        # Check that production sequence contains valid tuples
        for i, (nonterminal, prod_idx) in enumerate(production_sequence):
            assert hasattr(nonterminal, 'symbol'), f"First element should be a Nonterminal at position {i} for {program_name}"
            assert isinstance(prod_idx, int), f"Second element should be an int at position {i} for {program_name}"
            assert prod_idx >= 0, f"Production index should be non-negative at position {i} for {program_name}"

        # Check that terminal requirements contain valid tuples (now with context)
        for i, terminal_req in enumerate(terminal_requirements):
            if len(terminal_req) == 3:
                terminal_type, target_value, context = terminal_req
                assert isinstance(context, (list, type(None))), f"Context should be a list or None at position {i} for {program_name}"
            else:
                # Backward compatibility
                terminal_type, target_value = terminal_req

            assert isinstance(terminal_type, str), f"Terminal type should be a string at position {i} for {program_name}"
            assert isinstance(target_value, str), f"Target value should be a string at position {i} for {program_name}"
            assert terminal_type != "UNKNOWN", f"Terminal type should not be UNKNOWN at position {i} for {program_name}, got '{target_value}'"

        # log.info(f"✓ Successfully parsed {program_name} with {len(production_sequence)} productions and {len(terminal_requirements)} terminals")

    assert len(failed_programs) == 0, f"Failed to parse {len(failed_programs)} programs: \n{''.join(failed_programs)}"
    log.info(f"Average production sequence length: {production_sequence_length / len(registry.list_names())}")
    log.info(f"Average tokens length: {tokens_length / len(registry.list_names())}")


def test_batched_parser_all_programs():
    from dataset.grammar import get_batched_parser

    # Get the batched parser
    batched_parser = get_batched_parser()
    registry = augment_registry(get_program_registry(), num_samples=100, seed=42)  # Smaller sample for torch-struct tests

    failed_programs = []
    batch_sequences = []
    program_names = []

    # Collect all program token sequences
    for program_name in registry.list_names():
        program = registry.get(program_name)
        assert program is not None

        tokens = tokenize_code(program.implementation)
        batch_sequences.append(tokens)
        program_names.append(program_name)

    if not batch_sequences:
        pytest.skip("No programs to test")

    # Test batch parsing
    try:
        # Parse in smaller batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(batch_sequences), batch_size):
            batch = batch_sequences[i:i+batch_size]
            batch_names = program_names[i:i+batch_size]

            # log.info(f"Testing batch {i//batch_size + 1} with {len(batch)} programs")

            try:
                results = batched_parser.parse_batch(batch)

                # Validate results structure
                assert isinstance(results, dict), "Results should be a dictionary"
                assert 'batch_marginals' in results, "Results should contain batch_marginals"
                assert 'batch_partitions' in results, "Results should contain batch_partitions"
                assert 'batch_entropies' in results, "Results should contain batch_entropies"
                assert 'lengths' in results, "Results should contain lengths"

                # Validate batch dimensions
                assert len(results['batch_marginals']) == len(batch), f"Marginals count mismatch for batch starting at {i}"
                assert results['batch_partitions'].shape[0] == len(batch), f"Partitions batch size mismatch for batch starting at {i}"
                assert results['batch_entropies'].shape[0] == len(batch), f"Entropies batch size mismatch for batch starting at {i}"
                assert results['lengths'].shape[0] == len(batch), f"Lengths batch size mismatch for batch starting at {i}"

                # Validate individual results
                for j, (seq, name) in enumerate(zip(batch, batch_names)):
                    # Check that marginals have correct shape
                    if j < len(results['batch_marginals']):
                        marginals = results['batch_marginals'][j]
                        assert marginals.shape[0] == len(seq), f"Marginals sequence length mismatch for {name}: expected {len(seq)}, got {marginals.shape[0]}"

                        # Check that marginals are valid probabilities (non-negative)
                        assert (marginals >= 0).all(), f"Marginals should be non-negative for {name}"

                    # Check partition function is positive
                    if j < results['batch_partitions'].shape[0]:
                        partition = results['batch_partitions'][j]
                        # Note: In log-space, partition could be negative, so we just check it's finite
                        assert partition.isfinite(), f"Partition function should be finite for {name}"

                    # Check entropy is non-negative and finite
                    if j < results['batch_entropies'].shape[0]:
                        entropy = results['batch_entropies'][j]
                        assert entropy.isfinite(), f"Entropy should be finite for {name}"
                        # Note: Entropy could be negative in some edge cases, so we don't assert >= 0

                    # Check that actual length matches expected
                    if j < results['lengths'].shape[0]:
                        length = results['lengths'][j].item()
                        assert length == len(seq), f"Length mismatch for {name}: expected {len(seq)}, got {length}"

            except Exception as e:
                failed_programs.extend([f"{name}: {str(e)}" for name in batch_names])
                continue

    except Exception as e:
        failed_programs.append(f"Batch parsing failed: {str(e)}")

    # Report results
    if failed_programs:
        log.warning(f"Failed to parse {len(failed_programs)} programs with batched parser")
        for failure in failed_programs[:5]:  # Show first 5 failures
            log.warning(f"  {failure}")
        if len(failed_programs) > 5:
            log.warning(f"  ... and {len(failed_programs) - 5} more failures")

    # Allow some failures for torch-struct compatibility, but most should work
    success_rate = (len(batch_sequences) - len(failed_programs)) / len(batch_sequences)
    assert success_rate == 1.0, f"Batched parser success rate too low: {success_rate:.2%}. Failed programs: {len(failed_programs)}/{len(batch_sequences)}"

    log.info(f"Batched parser success rate: {success_rate:.2%} ({len(batch_sequences) - len(failed_programs)}/{len(batch_sequences)} programs)")