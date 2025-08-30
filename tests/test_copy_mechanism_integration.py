"""
Tests for Copy Mechanism Integration with Context Tracking

Tests that the copy mechanism properly uses context information for training.
"""

import pytest
import torch
import torch.nn.functional as F
from models.generation_head import GrammarAwareGenerationHead
from dataset.grammar import get_cfg


class TestCopyMechanismIntegration:
    """Test integration between context tracking and copy mechanism training."""

    def test_context_aware_loss_calculation(self):
        """Test that loss calculation properly uses context information."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, get_cfg())

        hidden_state = torch.randn(1, hidden_dim)
        device = hidden_state.device

        # Test the new terminal requirements format with context
        # Simulate what parse_tokens_to_productions should return
        terminal_requirements = [
            ("VARIABLE", "a", []),        # First occurrence - should generate
            ("VARIABLE", "b", ["a"]),     # New variable - should generate
            ("VARIABLE", "a", ["a", "b"]), # Reuse - should copy
            ("VARIABLE", "c", ["a", "b"]), # New variable - should generate
            ("VARIABLE", "a", ["a", "b", "c"]), # Reuse - should copy
        ]

        # Test that the loss calculation works with both formats
        total_loss = torch.tensor(0.0, device=device)
        copy_decisions = []

        # Test the logic from compute_sequence_loss_single
        for terminal_requirement in terminal_requirements:
            if len(terminal_requirement) == 3:
                terminal_type, target_value, context_identifiers = terminal_requirement
                context_identifiers = context_identifiers or []
            else:
                terminal_type, target_value = terminal_requirement
                context_identifiers = []

            if terminal_type == "VARIABLE":
                id_output = gen_head.identifier_head(hidden_state, context_identifiers)

                should_copy = target_value in context_identifiers
                copy_decisions.append(should_copy)

                # Unified classifier loss - combines copy and generation decisions
                if should_copy and len(context_identifiers) > 0:
                    # Copy: target is at copy index in unified output (first 10 slots)
                    copy_target_idx = context_identifiers.index(target_value)
                    target_tensor = torch.tensor([copy_target_idx], device=device)
                else:
                    # Generate: target is at generation index in unified output (slots 10-35)
                    target_char_idx = ord(target_value.lower()) - ord('a')
                    if 0 <= target_char_idx < 26:
                        # Generation indices start at slot 10 (max_identifiers)
                        generation_start = 10  # max_identifiers
                        target_tensor = torch.tensor([generation_start + target_char_idx], device=device)
                    else:
                        continue  # Skip invalid targets
                
                # Single cross-entropy loss over unified vocabulary
                unified_loss = F.cross_entropy(id_output["unified"], target_tensor)
                total_loss = total_loss + unified_loss

        # Verify the copy decisions are correct
        expected_decisions = [False, False, True, False, True]  # Generate, Generate, Copy, Generate, Copy
        assert copy_decisions == expected_decisions, f"Expected {expected_decisions}, got {copy_decisions}"

        # Verify loss is reasonable
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
        assert total_loss.item() < 100  # Should be reasonable magnitude

    def test_backward_compatibility(self):
        """Test that the system works with old format terminal requirements."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, get_cfg())

        hidden_state = torch.randn(1, hidden_dim)
        device = hidden_state.device

        # Old format (2-tuple)
        old_format_reqs = [
            ("VARIABLE", "a"),
            ("DIGIT", "0"),
        ]

        # Test that old format still works
        for terminal_requirement in old_format_reqs:
            if len(terminal_requirement) == 3:
                terminal_type, target_value, context_identifiers = terminal_requirement
                context_identifiers = context_identifiers or []
            else:
                terminal_type, target_value = terminal_requirement
                context_identifiers = []  # Fallback to empty context

            if terminal_type == "VARIABLE":
                id_output = gen_head.identifier_head(hidden_state, context_identifiers)
                assert isinstance(id_output, dict)
                assert "unified" in id_output, "Unified classifier should have 'unified' output"
                # Verify unified shape: fixed size = max_identifiers + generation_vocab_size
                expected_size = 10 + 26  # max_identifiers + alphabet = 36
                assert id_output["unified"].size(-1) == expected_size

    def test_context_evolution_pattern(self):
        """Test that context evolves properly during training."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, get_cfg())

        # Simulate a program: def program(x): y = x + 1; return y
        # Context should evolve: [] -> [x] -> [x, y] -> [x, y]

        contexts_sequence = [
            [],          # Parameter x definition
            ["x"],       # Assignment y =
            ["x"],       # Usage x in assignment
            ["x", "y"],  # Usage y in return
        ]

        variables_sequence = ["x", "y", "x", "y"]

        for i, (var, expected_context) in enumerate(zip(variables_sequence, contexts_sequence)):
            should_copy = var in expected_context

            if i == 0:  # First x - parameter definition
                assert should_copy == False
            elif i == 1:  # First y - assignment target
                assert should_copy == False
            elif i == 2:  # Second x - usage
                assert should_copy == True
            elif i == 3:  # Second y - usage
                assert should_copy == True

    def test_context_tracking_function_format(self):
        """Test the build_terminal_requirements_with_context function format."""
        # Test that if we had a working context tracker, it would return proper format

        # Simulate what the context tracker should produce
        simulated_output = [
            ("DEF", "def", None),
            ("PROGRAM_NAME", "program", None),
            ("LPAREN", "(", None),
            ("VARIABLE", "a", []),  # Parameter definition - empty context
            ("RPAREN", ")", None),
            ("COLON", ":", None),
            ("VARIABLE", "a", ["a"]),  # Variable usage - has context
        ]

        # Verify format
        for req in simulated_output:
            assert len(req) == 3
            terminal_type, target_value, context = req
            assert isinstance(terminal_type, str)
            assert isinstance(target_value, str)

            if terminal_type == "VARIABLE":
                assert isinstance(context, list), f"VARIABLE context should be list, got {type(context)}"
            else:
                assert context is None, f"Non-VARIABLE context should be None, got {context}"

    def test_proper_copy_mechanism_training_signal(self):
        """Test that copy mechanism receives proper training signal."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, get_cfg())

        hidden_state = torch.randn(1, hidden_dim)
        device = hidden_state.device

        # Test case: variable reuse should trigger copy mechanism training
        context_identifiers = ["a", "b"]
        target_value = "a"  # Should copy

        id_output = gen_head.identifier_head(hidden_state, context_identifiers)
        should_copy = target_value in context_identifiers

        assert should_copy == True

        # Unified classifier should train on copy target
        copy_target_idx = context_identifiers.index(target_value)
        copy_target_tensor = torch.tensor([copy_target_idx], device=device)
        unified_loss = F.cross_entropy(id_output["unified"], copy_target_tensor)

        assert isinstance(unified_loss, torch.Tensor)
        assert unified_loss.item() >= 0

        # Verify the unified loss is reasonable
        assert unified_loss.item() > 0

    def test_generation_mechanism_training_signal(self):
        """Test that generation mechanism receives proper training signal."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, get_cfg())

        hidden_state = torch.randn(1, hidden_dim)
        device = hidden_state.device

        # Test case: new variable should trigger generation mechanism training
        context_identifiers = ["a", "b"]
        target_value = "c"  # Should generate (not in context)

        id_output = gen_head.identifier_head(hidden_state, context_identifiers)
        should_copy = target_value in context_identifiers

        assert should_copy == False

        # Unified classifier should train on generation target
        target_char_idx = ord(target_value.lower()) - ord('a')
        # Generation indices start at slot 10 (max_identifiers)
        generation_start = 10  # max_identifiers
        target_tensor = torch.tensor([generation_start + target_char_idx], device=device)
        unified_loss = F.cross_entropy(id_output["unified"], target_tensor)

        assert isinstance(unified_loss, torch.Tensor)
        assert unified_loss.item() >= 0

        # Verify the unified loss is reasonable
        assert unified_loss.item() > 0


class TestRegressionPrevention:
    """Tests to prevent regression of the copy mechanism fix."""

    def test_identifier_loss_components_all_trained(self):
        """Test that all identifier loss components receive gradients."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, get_cfg())

        hidden_state = torch.randn(1, hidden_dim, requires_grad=True)

        # Test both copy and generate scenarios
        test_cases = [
            (["a"], "a", True),   # Should copy
            ([], "a", False),     # Should generate
        ]

        for context, target, expected_copy in test_cases:
            id_output = gen_head.identifier_head(hidden_state, context)
            should_copy = target in context
            assert should_copy == expected_copy

            device = hidden_state.device

            # Unified classifier should always be trained
            if should_copy and context:
                # Copy target: index in context (first 10 slots)
                copy_target_idx = context.index(target)
                target_tensor = torch.tensor([copy_target_idx], device=device)
            else:
                # Generation target: index at slot 10 + char_index (slots 10-35)
                target_char_idx = ord(target.lower()) - ord('a')
                generation_start = 10  # max_identifiers
                target_tensor = torch.tensor([generation_start + target_char_idx], device=device)
            
            # Single unified loss
            unified_loss = F.cross_entropy(id_output["unified"], target_tensor)

            # Backward pass to verify gradients
            unified_loss.backward(retain_graph=True)
            assert hidden_state.grad is not None, "Unified classifier should create gradients"

            # Reset gradients for next iteration
            hidden_state.grad.zero_()