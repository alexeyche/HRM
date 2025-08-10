#!/usr/bin/env python3
"""
End-to-end test suite for grammar actions with programs from the registry.
Tests the full stack: program -> AST -> actions -> graph -> validation
"""

import pytest
from typing import Dict, Any, List

from dataset.programs import get_program_registry
from dataset.ast import ASTNodeType, EdgeType
from dataset.grammar_actions import (
    graph_to_actions,
    actions_to_graph,
    Action,
    ActionKind,
    simplified_ast_to_graph
)


class TestGrammarActionsEndToEnd:
    """Test class for end-to-end grammar actions functionality"""

    @pytest.fixture(scope="class")
    def registry(self):
        """Get the program registry for testing"""
        return get_program_registry()

    @pytest.mark.parametrize("program_name", [
        "sum_of_two",      # def program(a, b): return a + b
        "product",         # def program(a, b): return a * b
        "is_positive",     # def program(n): return n > 0
        "double",          # def program(n): return n * 2
        "is_even",         # def program(n): return n % 2 == 0
        "square"           # def program(n): return n * n
    ])
    def test_program_roundtrip(self, registry, program_name):
        """Test that each program can be converted to actions and back to graph successfully"""
        self._test_single_program_roundtrip(registry, program_name)

    def _test_single_program_roundtrip(self, registry, program_name: str):
        """Test a single program through the full grammar actions stack"""
        # Get the program
        program = registry.get(program_name)
        assert program is not None, f"Program '{program_name}' not found in registry"

        code = program.implementation

        # Step 1: Convert to simplified AST graph
        original_graph = simplified_ast_to_graph(code)
        assert len(original_graph['nodes']) > 0, "AST graph should have nodes"
        assert 'root' in original_graph, "AST graph should have root"

        # Step 2: Convert AST to actions
        actions = graph_to_actions(original_graph)
        assert len(actions) > 0, "Should generate actions"

        # Step 3: Convert actions back to graph
        reconstructed_graph = actions_to_graph(actions)
        assert len(reconstructed_graph['nodes']) > 0, "Reconstructed graph should have nodes"

        # Step 4: Validate roundtrip
        self._validate_roundtrip(original_graph, reconstructed_graph, program_name)

    def _validate_roundtrip(self, original_graph: Dict[str, Any],
                           reconstructed_graph: Dict[str, Any], program_name: str):
        """Validate that the roundtrip conversion preserves essential structure"""
        # Check node count
        original_nodes = len(original_graph['nodes'])
        reconstructed_nodes = len(reconstructed_graph['nodes'])
        assert original_nodes == reconstructed_nodes, (
            f"Node count mismatch for {program_name}: "
            f"{original_nodes} vs {reconstructed_nodes}"
        )

        # Check node types (ignoring order)
        original_types = [str(n['type']) for n in original_graph['nodes']]
        reconstructed_types = [str(n['type']) for n in reconstructed_graph['nodes']]
        assert sorted(original_types) == sorted(reconstructed_types), (
            f"Node type mismatch for {program_name}:\n"
            f"  Original: {original_types}\n"
            f"  Reconstructed: {reconstructed_types}"
        )

        # Check that essential structure is preserved
        self._check_essential_structure(original_graph, reconstructed_graph, program_name)

    def _check_essential_structure(self, original_graph: Dict[str, Any],
                                 reconstructed_graph: Dict[str, Any], program_name: str):
        """Check that essential structural elements are preserved"""
        # Check function definition exists
        original_fn = next(n for n in original_graph['nodes']
                          if n['type'] == ASTNodeType.FUNCTION_DEF)
        reconstructed_fn = next(n for n in reconstructed_graph['nodes']
                              if n['type'] == ASTNodeType.FUNCTION_DEF)

        # Check return statement exists
        original_ret = next(n for n in original_graph['nodes']
                           if n['type'] == ASTNodeType.RETURN)
        reconstructed_ret = next(n for n in reconstructed_graph['nodes']
                               if n['type'] == ASTNodeType.RETURN)

        # Check that operators are preserved
        self._check_operators_preserved(original_graph, reconstructed_graph, program_name)

    def _check_operators_preserved(self, original_graph: Dict[str, Any],
                                 reconstructed_graph: Dict[str, Any], program_name: str):
        """Check that operators in expressions are preserved"""
        # Check binary operations
        original_binops = [n for n in original_graph['nodes']
                          if n['type'] == ASTNodeType.BINARY_OPERATION]
        reconstructed_binops = [n for n in reconstructed_graph['nodes']
                              if n['type'] == ASTNodeType.BINARY_OPERATION]

        if original_binops:
            assert len(reconstructed_binops) == len(original_binops), (
                f"Binary operation count mismatch for {program_name}"
            )
            for orig, recon in zip(original_binops, reconstructed_binops):
                assert orig.get('op') == recon.get('op'), (
                    f"Binary operator mismatch for {program_name}: "
                    f"{orig.get('op')} vs {recon.get('op')}"
                )

        # Check comparisons
        original_comps = [n for n in original_graph['nodes']
                         if n['type'] == ASTNodeType.COMPARISON]
        reconstructed_comps = [n for n in reconstructed_graph['nodes']
                             if n['type'] == ASTNodeType.COMPARISON]

        if original_comps:
            assert len(reconstructed_comps) == len(original_comps), (
                f"Comparison count mismatch for {program_name}"
            )
            for orig, recon in zip(original_comps, reconstructed_comps):
                assert orig.get('op') == recon.get('op'), (
                    f"Comparison operator mismatch for {program_name}: "
                    f"{orig.get('op')} vs {recon.get('op')}"
                )

    def test_action_generation_consistency(self, registry):
        """Test that action generation is consistent across similar programs"""
        # Test that similar programs generate similar action patterns
        sum_program = registry.get("sum_of_two")
        product_program = registry.get("product")

        assert sum_program is not None and product_program is not None

        sum_actions = graph_to_actions(simplified_ast_to_graph(sum_program.implementation))
        product_actions = graph_to_actions(simplified_ast_to_graph(product_program.implementation))

        # Both should have the same action structure (just different operators)
        assert len(sum_actions) == len(product_actions), "Similar programs should generate same number of actions"

        # Check action kinds match
        sum_kinds = [a.kind for a in sum_actions]
        product_kinds = [a.kind for a in product_actions]
        assert sum_kinds == product_kinds, "Similar programs should generate same action kinds"

        # Only the operator should differ
        sum_op = next(a.value for a in sum_actions if a.kind == ActionKind.SET_OP)
        product_op = next(a.value for a in product_actions if a.kind == ActionKind.SET_OP)
        assert sum_op == "+" and product_op == "*", "Operators should be preserved"

    def test_edge_cases(self, registry):
        """Test edge cases and error conditions"""
        # Test with a program that has no return value
        # (This would require extending our grammar actions to handle this case)

        # Test that invalid actions are rejected
        with pytest.raises(ValueError):
            # Try to create an invalid action sequence
            invalid_actions = [Action(ActionKind.EOS)]  # Just EOS without proper structure
            actions_to_graph(invalid_actions)

    @pytest.mark.parametrize("test_name,code", [
        ("simple_variable", "def program(n): return n"),
        ("simple_constant", "def program(): return 42"),
        ("binary_op", "def program(a, b): return a + b"),
        ("comparison", "def program(n): return n > 0"),
        ("nested_binary", "def program(a, b, c): return a + b * c"),
    ])
    def test_grammar_coverage(self, registry, test_name, code):
        """Test that our grammar actions cover the intended subset of Python"""
        # Test that we can handle all the basic expression types we claim to support
        try:
            graph = simplified_ast_to_graph(code)
            actions = graph_to_actions(graph)
            reconstructed = actions_to_graph(actions)

            # Basic validation
            assert len(graph['nodes']) == len(reconstructed['nodes'])

        except Exception as e:
            pytest.fail(f"Failed to process {test_name}: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
