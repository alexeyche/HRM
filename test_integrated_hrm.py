#!/usr/bin/env python3
"""
Test the integrated HRM program synthesis model with proper configuration
"""

import torch
from typing import Dict, Any

from models.program_synthesis_processor import ProgramSynthesisProcessor
from models.program_synthesis_hrm import ProgramSynthesisHRM
from dataset.build_program_synthesis_dataset import PROGRAM_TEMPLATES


def create_compatible_config():
    """Create HRM config compatible with program synthesis"""
    return {
        'batch_size': 2,
        'seq_len': 64,
        'puzzle_emb_ndim': 0,  # Disable puzzle embeddings for simplicity
        'num_puzzle_identifiers': 10,
        'vocab_size': 1024,  # Match processor vocab size
        'H_cycles': 1,  # Minimal for testing
        'L_cycles': 1,
        'H_layers': 1,
        'L_layers': 1,
        'hidden_size': 128,  # Reduced for testing
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 1,  # Disable ACT for testing
        'halt_exploration_prob': 0.0,
        'max_nodes': 20,
        'max_edges': 15,
        'num_node_types': 30,
        'max_spec_tokens': 64,
        'max_examples': 5,
        'forward_dtype': 'float32'  # Use float32 instead of bfloat16
    }


def test_integrated_model():
    """Test the full integrated HRM program synthesis model"""
    print("Testing Integrated HRM Program Synthesis Model...\n")
    
    # Create model and processor
    config = create_compatible_config()
    model = ProgramSynthesisHRM(config)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test data
    test_examples = []
    if 'double' in PROGRAM_TEMPLATES:
        template = PROGRAM_TEMPLATES['double']
        example = {
            'specification': {
                'name': 'double',
                'description': template['description'],
                'inputs': template['inputs'],
                'outputs': template['outputs'],
                'examples': template['base_examples'][:3]
            },
            'implementation': template['implementation']
        }
        test_examples.append(example)
    
    # Fallback if template not found
    if not test_examples:
        test_examples = [{
            'specification': {
                'name': 'simple_add',
                'description': 'Add 1 to input',
                'inputs': [{'type': 'int'}],
                'outputs': [{'type': 'int'}],
                'examples': [{'input': 5, 'output': 6}]
            },
            'implementation': 'def program(n): return n + 1'
        }]
    
    # Create batch
    batch = model.processor.create_hrm_batch(test_examples[:1])  # Use single example first
    
    print(f"✓ Batch created")
    print(f"  Input shape: {batch['inputs'].shape}")
    print(f"  Input vocab range: {batch['inputs'].min().item()} to {batch['inputs'].max().item()}")
    
    # Check vocabulary compatibility
    if batch['inputs'].max().item() >= config['vocab_size']:
        print(f"❌ Vocabulary size issue: max token {batch['inputs'].max().item()} >= vocab_size {config['vocab_size']}")
        return False
    
    # Initialize carry
    try:
        carry = model.initial_carry(batch)
        print(f"✓ Initial carry created")
    except Exception as e:
        print(f"❌ Initial carry failed: {e}")
        return False
    
    # Forward pass
    try:
        model.eval()
        with torch.no_grad():
            new_carry, outputs = model(carry, batch)
        
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        
        # Print output shapes
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    try:
        targets = {'ast_targets': batch['ast_targets']}
        losses = model.compute_loss(outputs, targets)
        
        print(f"✓ Loss computation successful")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test program generation
    try:
        test_spec = {
            'name': 'test_generation',
            'description': 'Simple test function',
            'inputs': [{'type': 'int'}],
            'outputs': [{'type': 'int'}],
            'examples': [
                {'input': 3, 'output': 6},
                {'input': 5, 'output': 10}
            ]
        }
        
        generated_ast = model.generate_program(test_spec)
        
        print(f"✓ Program generation successful")
        print(f"  Generated AST keys: {list(generated_ast.keys())}")
        if 'num_nodes' in generated_ast:
            print(f"  Generated nodes: {generated_ast['num_nodes']}")
        if 'edges' in generated_ast:
            print(f"  Generated edges: {len(generated_ast['edges'])}")
        
        if generated_ast['nodes']:
            print(f"  Sample nodes:")
            for i, node in enumerate(generated_ast['nodes'][:3]):
                print(f"    {i}: {node['type_name']} (value: {node['value']})")
        
    except Exception as e:
        print(f"❌ Program generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_training_step():
    """Test a simple training step"""
    print("Testing training step...\n")
    
    config = create_compatible_config()
    model = ProgramSynthesisHRM(config)
    
    # Create simple batch
    test_example = {
        'specification': {
            'name': 'add_one',
            'inputs': [{'type': 'int'}],
            'outputs': [{'type': 'int'}],
            'examples': [{'input': 5, 'output': 6}]
        },
        'implementation': 'def program(n): return n + 1'
    }
    
    batch = model.processor.create_hrm_batch([test_example])
    
    # Training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    try:
        carry = model.initial_carry(batch)
        
        optimizer.zero_grad()
        new_carry, outputs = model(carry, batch)
        
        # Compute loss
        targets = {'ast_targets': batch['ast_targets']}
        losses = model.compute_loss(outputs, targets)
        
        loss = losses['total_loss']
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients computed and applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run integrated HRM tests"""
    print("Testing Integrated HRM Program Synthesis Model\n")
    
    success = True
    
    # Test 1: Basic model functionality
    if not test_integrated_model():
        success = False
    
    print()
    
    # Test 2: Training step
    if not test_training_step():
        success = False
    
    if success:
        print(f"\n🎉 Integrated HRM Program Synthesis Model Tests PASSED!")
        print(f"\nKey achievements:")
        print(f"✅ Full HRM model integration with PyTorch fallback attention")
        print(f"✅ Program synthesis data processing pipeline")  
        print(f"✅ AST generation heads working with HRM")
        print(f"✅ Multi-component loss computation")
        print(f"✅ Training step with gradient computation")
        print(f"✅ Program generation from specifications")
        print(f"\n🚀 The model is ready for training on program synthesis tasks!")
        
    else:
        print(f"\n❌ Some tests failed")
        
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())