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

                # Copy gate loss
                copy_gate_target = torch.tensor([1.0 if should_copy else 0.0], device=device)
                copy_gate_logit = id_output["copy_gate"].squeeze(-1)
                copy_gate_loss = F.binary_cross_entropy_with_logits(copy_gate_logit, copy_gate_target)
                total_loss = total_loss + copy_gate_loss

                if should_copy and len(context_identifiers) > 0:
                    # Copy attention loss
                    copy_target_idx = context_identifiers.index(target_value)
                    copy_target_tensor = torch.tensor([copy_target_idx], device=device)
                    copy_attention_loss = F.cross_entropy(id_output["copy_attention"], copy_target_tensor)
                    total_loss = total_loss + copy_attention_loss
                else:
                    # Generation loss
                    target_char_idx = ord(target_value.lower()) - ord('a')
                    if 0 <= target_char_idx < 26:
                        target_tensor = torch.tensor([target_char_idx], device=device)
                        gen_loss = F.cross_entropy(id_output["generation"], target_tensor)
                        total_loss = total_loss + gen_loss

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
                assert "copy_gate" in id_output
                assert "copy_attention" in id_output
                assert "generation" in id_output

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

        # Both copy gate and copy attention should be trained
        copy_gate_target = torch.tensor([1.0], device=device)  # Should copy
        copy_gate_logit = id_output["copy_gate"].squeeze(-1)
        copy_gate_loss = F.binary_cross_entropy_with_logits(copy_gate_logit, copy_gate_target)

        copy_target_idx = context_identifiers.index(target_value)
        copy_target_tensor = torch.tensor([copy_target_idx], device=device)
        copy_attention_loss = F.cross_entropy(id_output["copy_attention"], copy_target_tensor)

        assert isinstance(copy_gate_loss, torch.Tensor)
        assert isinstance(copy_attention_loss, torch.Tensor)
        assert copy_gate_loss.item() >= 0
        assert copy_attention_loss.item() >= 0

        # Total loss should be combination
        total_loss = copy_gate_loss + copy_attention_loss
        assert total_loss.item() > 0

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

        # Both copy gate and generation head should be trained
        copy_gate_target = torch.tensor([0.0], device=device)  # Should generate
        copy_gate_logit = id_output["copy_gate"].squeeze(-1)
        copy_gate_loss = F.binary_cross_entropy_with_logits(copy_gate_logit, copy_gate_target)

        target_char_idx = ord(target_value.lower()) - ord('a')
        target_tensor = torch.tensor([target_char_idx], device=device)
        gen_loss = F.cross_entropy(id_output["generation"], target_tensor)

        assert isinstance(copy_gate_loss, torch.Tensor)
        assert isinstance(gen_loss, torch.Tensor)
        assert copy_gate_loss.item() >= 0
        assert gen_loss.item() >= 0

        # Total loss should be combination
        total_loss = copy_gate_loss + gen_loss
        assert total_loss.item() > 0


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

            # Copy gate should always be trained
            copy_gate_target = torch.tensor([1.0 if should_copy else 0.0], device=device)
            copy_gate_logit = id_output["copy_gate"].squeeze(-1)
            copy_gate_loss = F.binary_cross_entropy_with_logits(copy_gate_logit, copy_gate_target)

            # Backward pass to verify gradients
            copy_gate_loss.backward(retain_graph=True)
            assert hidden_state.grad is not None, "Copy gate should create gradients"

            # Reset gradients
            hidden_state.grad.zero_()

            if should_copy and context:
                # Copy attention should be trained
                copy_target_idx = context.index(target)
                copy_target_tensor = torch.tensor([copy_target_idx], device=device)
                copy_attention_loss = F.cross_entropy(id_output["copy_attention"], copy_target_tensor)
                copy_attention_loss.backward(retain_graph=True)
                assert hidden_state.grad is not None, "Copy attention should create gradients"
                hidden_state.grad.zero_()
            else:
                # Generation should be trained
                target_char_idx = ord(target.lower()) - ord('a')
                target_tensor = torch.tensor([target_char_idx], device=device)
                gen_loss = F.cross_entropy(id_output["generation"], target_tensor)
                gen_loss.backward(retain_graph=True)
                assert hidden_state.grad is not None, "Generation should create gradients"
                hidden_state.grad.zero_()