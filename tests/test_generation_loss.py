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
    
    # Simple target token sequences
    target_tokens = [
        ["def", "program", "a", "return", "1"],
        ["def", "program", "b", "return", "2"]
    ]
    
    # Compute loss
    loss_dict = gen_head.compute_sequence_loss(context_embeddings, target_tokens)
    
    # Check that all components are tensors with gradients
    assert isinstance(loss_dict["total_loss"], torch.Tensor), "total_loss should be a tensor"
    assert loss_dict["total_loss"].requires_grad, "total_loss should require gradients"
    assert loss_dict["total_loss"].item() >= 0.0, "Loss should be non-negative"
    
    print(f"✓ Basic loss computation works: {loss_dict['total_loss'].item():.4f}")
    return loss_dict


def test_gradient_flow():
    """Test that gradients flow through the loss function."""
    print("\nTesting gradient flow...")
    
    hidden_dim = 32
    gen_head = GrammarAwareGenerationHead(hidden_dim)
    
    # Create a simple linear layer to test gradients on
    linear = nn.Linear(hidden_dim, hidden_dim)
    optimizer = torch.optim.SGD(list(gen_head.parameters()) + list(linear.parameters()), lr=0.1)
    
    context_embeddings = linear(torch.randn(1, 1, hidden_dim))
    target_tokens = [["def", "program", "x", "return", "5"]]
    
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
    print("✓ Gradients flow back to input embeddings")


def test_loss_optimization_direction():
    """Test with 5 manual examples that loss decreases during optimization."""
    print("\nTesting loss optimization with 5 manual examples...")
    
    # Create autoencoder with small dimensions for fast testing
    model = ASTAutoencoder(hidden_dim=32, max_decode_steps=20)
    trainer = ASTAutoencoderTrainer(model, device=torch.device('cpu'))
    
    # 5 simple test programs with increasing complexity
    test_programs = [
        "def program ( a ) : return a",           # Simple identity
        "def program ( a ) : return 1",           # Return constant
        "def program ( a ) : return a + 1",       # Simple arithmetic
        "def program ( a , b ) : return a + b",   # Two parameters
        "def program ( x ) : return x * 2"        # Different operation
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
                    [program], result['programs'], result['latent'], use_generation_loss=True
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
                    [program], result['programs'], result['latent'], use_generation_loss=True
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
                    [program], result['programs'], result['latent'], use_generation_loss=True
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


def test_component_losses():
    """Test individual loss components work properly."""
    print("\nTesting individual loss components...")
    
    gen_head = GrammarAwareGenerationHead(32)
    context = torch.randn(1, 1, 32)
    
    # Test different token types
    test_cases = [
        (["a"], "identifier"),           # Single identifier  
        (["1"], "number"),              # Single number
        (["True"], "boolean"),          # Boolean literal
        (["def", "program"], "keywords"), # Keywords
    ]
    
    for tokens, description in test_cases:
        loss_dict = gen_head.compute_sequence_loss(context, [tokens])
        total_loss = loss_dict["total_loss"].item()
        
        print(f"  {description:12}: loss = {total_loss:.4f}")
        assert total_loss >= 0.0, f"Loss should be non-negative for {description}"
        assert not torch.isnan(loss_dict["total_loss"]), f"Loss should not be NaN for {description}"
    
    print("✓ All component losses computed correctly")


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING GENERATION HEAD LOSS FUNCTIONS")
    print("="*60)
    
    try:
        test_basic_loss_computation()
        test_gradient_flow()
        test_component_losses()
        test_loss_optimization_direction()  # The main test
        
        print(f"\n{'='*60}")
        print("🎉 ALL TESTS PASSED!")
        print("✓ Loss computation works correctly")
        print("✓ Gradients propagate properly") 
        print("✓ Loss optimizes in the right direction")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()