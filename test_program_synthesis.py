#!/usr/bin/env python3
"""
Test script for program synthesis input/output processing

This script tests the integration of program synthesis data processing
with the HRM model architecture.
"""

import os
import json
import torch
from typing import Dict, Any

from models.program_synthesis_processor import ProgramSynthesisProcessor, ProgramSynthesisDataset
from models.program_synthesis_hrm import ProgramSynthesisHRM, ProgramSynthesisHRMConfig
from dataset.build_program_synthesis_dataset import PROGRAM_TEMPLATES


def create_test_examples() -> list[Dict[str, Any]]:
    """Create test examples for program synthesis"""
    examples = []

    # Use a few simple templates from the dataset
    test_templates = ['sum_up_to_n', 'max_of_two', 'absolute_value', 'is_even']

    for name in test_templates:
        if name in PROGRAM_TEMPLATES:
            template = PROGRAM_TEMPLATES[name]
            example = {
                'specification': {
                    'name': name,
                    'description': template['description'],
                    'inputs': template['inputs'],
                    'outputs': template['outputs'],
                    'examples': template['base_examples'][:5]  # Use first 5 examples
                },
                'implementation': template['implementation']
            }
            examples.append(example)

    return examples


def test_processor():
    """Test the ProgramSynthesisProcessor"""
    print("Testing ProgramSynthesisProcessor...")

    processor = ProgramSynthesisProcessor(
        max_examples=5,
        max_spec_tokens=64,
        max_nodes=30,
        max_edges=25,
        vocab_size=1024
    )

    # Test specification encoding
    test_spec = {
        'name': 'sum_up_to_n',
        'description': 'Sum up all numbers up to N',
        'inputs': [{'type': 'int', 'description': 'The input number N'}],
        'outputs': [{'type': 'int', 'description': 'The sum'}],
        'examples': [
            {'input': 5, 'output': 15},
            {'input': 10, 'output': 55}
        ]
    }

    encoded = processor.encode_specification(test_spec)
    print(f"Encoded specification shape: {encoded.shape}")
    print(f"First 10 tokens: {encoded[:10].tolist()}")

    # Test AST processing
    implementation = """def program(n):
    return sum(range(1, n + 1))"""

    ast_data = processor.process_ast_target(implementation)
    print(f"AST processing results:")
    for key, value in ast_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {value}")

    # Test batch creation
    examples = create_test_examples()
    batch = processor.create_hrm_batch(examples[:2])
    print(f"Batch creation results:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    print(f"    {subkey}: shape {subvalue.shape}, dtype {subvalue.dtype}")

    print("✓ ProgramSynthesisProcessor tests passed\n")


def test_hrm_model():
    """Test the ProgramSynthesisHRM model"""
    print("Testing ProgramSynthesisHRM model...")

    # Create model config
    config = {
        'batch_size': 2,
        'seq_len': 64,
        'puzzle_emb_ndim': 0,  # Disable puzzle embeddings for simplicity
        'num_puzzle_identifiers': 100,
        'vocab_size': 1024,
        'H_cycles': 2,
        'L_cycles': 3,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 256,
        'expansion': 2.0,
        'num_heads': 8,
        'pos_encodings': 'rope',
        'halt_max_steps': 1,  # Disable ACT for simplicity
        'halt_exploration_prob': 0.0,
        'max_nodes': 30,
        'max_edges': 25,
        'num_node_types': 30,
        'max_spec_tokens': 64,
        'max_examples': 5
    }

    # Create model
    model = ProgramSynthesisHRM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create test batch
    examples = create_test_examples()
    batch = model.processor.create_hrm_batch(examples[:2])

    # Initialize carry
    carry = model.initial_carry(batch)
    print(f"Initial carry created")

    # Forward pass
    model.eval()
    with torch.no_grad():
        new_carry, outputs = model(carry, batch)

    print(f"Forward pass outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

    # Test loss computation
    targets = {'ast_targets': batch['ast_targets']}
    losses = model.compute_loss(outputs, targets)
    print(f"Loss computation results:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")

    print("✓ ProgramSynthesisHRM model tests passed\n")


def test_generation():
    """Test program generation from specification"""
    print("Testing program generation...")

    # Create simple model config
    config = {
        'batch_size': 1,
        'seq_len': 64,
        'puzzle_emb_ndim': 0,
        'num_puzzle_identifiers': 100,
        'vocab_size': 1024,
        'H_cycles': 1,
        'L_cycles': 1,
        'H_layers': 1,
        'L_layers': 1,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 1,
        'halt_exploration_prob': 0.0,
        'max_nodes': 30,
        'max_edges': 25,
        'num_node_types': 30,
        'max_spec_tokens': 64,
        'max_examples': 5
    }

    model = ProgramSynthesisHRM(config)

    # Test specification
    test_spec = {
        'name': 'simple_add',
        'description': 'Add two numbers',
        'inputs': [
            {'type': 'int', 'description': 'First number'},
            {'type': 'int', 'description': 'Second number'}
        ],
        'outputs': [{'type': 'int', 'description': 'Sum'}],
        'examples': [
            {'input': [2, 3], 'output': 5},
            {'input': [10, 5], 'output': 15}
        ]
    }

    # Generate program
    result = model.generate_program(test_spec)
    print(f"Generated AST:")
    print(f"  Nodes: {len(result['nodes'])}")
    print(f"  Edges: {len(result['edges'])}")

    for i, node in enumerate(result['nodes'][:5]):  # Show first 5 nodes
        print(f"  Node {i}: type={node['type_name']}, value={node['value']}")

    print("✓ Program generation tests passed\n")


def test_integration():
    """Integration test with dataset loading"""
    print("Testing dataset integration...")

    # Create temporary test data
    test_data_dir = "/tmp/test_program_synthesis"
    os.makedirs(test_data_dir, exist_ok=True)

    test_file = os.path.join(test_data_dir, "test_data.json")
    test_examples = create_test_examples()

    with open(test_file, 'w') as f:
        json.dump(test_examples, f, indent=2)

    # Test dataset loading
    processor = ProgramSynthesisProcessor()
    dataset = ProgramSynthesisDataset(
        data_path=test_file,
        processor=processor,
        max_examples=100
    )

    print(f"Dataset loaded with {len(dataset)} examples")

    # Test batch creation
    batch = dataset.get_random_batch(batch_size=2)
    print(f"Random batch created with keys: {list(batch.keys())}")

    # Clean up
    os.remove(test_file)
    os.rmdir(test_data_dir)

    print("✓ Dataset integration tests passed\n")


def main():
    """Run all tests"""
    print("Running Program Synthesis Input/Output Processing Tests\n")

    try:
        test_processor()
        test_hrm_model()
        test_generation()
        test_integration()

        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())