"""
Test script for generation head loss functions.

Tests that the loss functions properly propagate gradients and optimize in the correct direction.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from models.generation_head import GrammarAwareGenerationHead
from models.ast_autoencoder import ASTAutoencoder, ASTAutoencoderTrainer
from dataset.grammar import get_cfg


def test_basic_loss_computation():
    """Test that loss computation works and returns proper tensors."""
    print("Testing basic loss computation...")

    # Create generation head
    hidden_dim = 32
    grammar = get_cfg()
    gen_head = GrammarAwareGenerationHead(hidden_dim, grammar)

    # Create fake context embeddings
    batch_size = 2
    context_embeddings = torch.randn(batch_size, 1, hidden_dim)

    # Simple target token sequences (must match grammar)
    target_tokens = [
        ["def", "program", "(", "a", ")", ":", "<NEWLINE>", "<INDENT>", "return", "1", "<NEWLINE>", "<DEDENT>"],
        ["def", "program", "(", "a", ")", ":", "<NEWLINE>", "<INDENT>", "return", "2", "<NEWLINE>", "<DEDENT>"]
    ]

    # Compute loss
    loss_dict = gen_head.compute_sequence_loss(context_embeddings, target_tokens)

    # Check that all components are tensors with gradients
    assert isinstance(loss_dict["total_loss"], torch.Tensor), "total_loss should be a tensor"
    assert loss_dict["total_loss"].requires_grad, "total_loss should require gradients"
    assert loss_dict["total_loss"].item() >= 0.0, "Loss should be non-negative"

    print(f"✓ Basic loss computation works: {loss_dict['total_loss'].item():.4f}")


def test_gradient_flow():
    """Test that gradients flow through the loss function."""
    print("\nTesting gradient flow...")

    hidden_dim = 32
    gen_head = GrammarAwareGenerationHead(hidden_dim)

    # Create a simple linear layer to test gradients on
    linear = nn.Linear(hidden_dim, hidden_dim)
    optimizer = torch.optim.SGD(list(gen_head.parameters()) + list(linear.parameters()), lr=0.1)

    context_embeddings = linear(torch.randn(1, 1, hidden_dim))
    target_tokens = [["def", "program", "(", "x", ")", ":", "<NEWLINE>", "<INDENT>", "return", "5", "<NEWLINE>", "<DEDENT>"]]

    # Forward pass
    loss_dict = gen_head.compute_sequence_loss(context_embeddings, target_tokens)
    loss = loss_dict["total_loss"]

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    has_gradients = False
    for param in gen_head.parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "No gradients found in generation head parameters"
    print("✓ Gradients flow correctly through generation head")

    # Check linear layer gradients too
    assert linear.weight.grad is not None, "Linear layer should have gradients"
    print("✓ Gradients flow back to input embeddings: ", linear.weight.grad)


def test_loss_optimization_direction():
    """Test with 5 manual examples that loss decreases during optimization."""
    print("\nTesting loss optimization with 5 manual examples...")

    # Create autoencoder with small dimensions for fast testing
    model = ASTAutoencoder(hidden_dim=32, max_decode_steps=20)
    trainer = ASTAutoencoderTrainer(model, device=torch.device('cpu'))

    # 5 simple test programs with increasing complexity
    test_programs = [
        "def program ( a ) :\n    return a\n",           # Simple identity
        "def program ( a ) :\n    return 1\n",           # Return constant
        "def program ( a ) :\n    return a + 1\n",       # Simple arithmetic
        "def program ( a , b ) :\n    return a + b\n",   # Two parameters
        "def program ( x ) :\n    return x * 2\n"        # Different operation
    ]

    print(f"Testing optimization on {len(test_programs)} programs...")

    # Create a simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_losses = []
    final_losses = []

    # Run optimization for each program individually
    for i, program in enumerate(test_programs):
        print(f"\nProgram {i+1}: {program}")

        # Convert program to graph and create batch
        try:
            from dataset.ast_converter import program_to_graph
            from torch_geometric.data import Batch

            graph = program_to_graph(program)
            batched_graph = Batch.from_data_list([graph])

            # Initial loss
            with torch.no_grad():
                result = model(batched_graph, decode=True, max_steps=10, temperature=0.1)
                initial_loss_dict = trainer.reconstruction_loss(
                    [program], result['programs'], result['latent']
                )
                initial_loss = initial_loss_dict['total_loss'].item()
                initial_losses.append(initial_loss)

            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Initial reconstruction: {repr(result['programs'][0][:50])}")  # First 50 chars

            # Training steps
            for step in range(10):  # Just a few steps to see if loss decreases
                optimizer.zero_grad()

                result = model(batched_graph, decode=True, max_steps=10, temperature=1.0)  # Higher temp for training
                loss_dict = trainer.reconstruction_loss(
                    [program], result['programs'], result['latent']
                )

                loss = loss_dict['total_loss']
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if step % 3 == 0:
                    print(f"  Step {step}: loss = {loss.item():.4f}")

            # Final loss
            with torch.no_grad():
                result = model(batched_graph, decode=True, max_steps=10, temperature=0.1)
                final_loss_dict = trainer.reconstruction_loss(
                    [program], result['programs'], result['latent']
                )
                final_loss = final_loss_dict['total_loss'].item()
                final_losses.append(final_loss)

            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Final reconstruction: {repr(result['programs'][0][:50])}")  # First 50 chars
            print(f"  Loss change: {final_loss - initial_loss:.4f} {'✓' if final_loss < initial_loss else '⚠'}")

        except Exception as e:
            print(f"  Error processing program {i+1}: {e}")
            initial_losses.append(float('inf'))
            final_losses.append(float('inf'))

    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")

    improvements = 0
    for i, (initial, final) in enumerate(zip(initial_losses, final_losses)):
        if final < initial:
            improvements += 1
            status = "✓ IMPROVED"
        else:
            status = "⚠ NO IMPROVEMENT"

        print(f"Program {i+1}: {initial:.4f} → {final:.4f} ({final-initial:+.4f}) {status}")

    print(f"\nResult: {improvements}/{len(test_programs)} programs showed loss improvement")

    # Assert that at least some programs improved
    assert improvements >= 2, f"Expected at least 2 programs to show improvement, got {improvements}"
    print(f"✓ Loss optimization working: {improvements} programs improved!")

