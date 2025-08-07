#!/usr/bin/env python3
"""
Debug Q-learning in more detail - check HRM outputs
"""

import torch
from models.program_synthesis_hrm import ProgramSynthesisHRM
from program_synthesis_dataset import ProgramSynthesisDataset, ProgramSynthesisDatasetConfig


def debug_hrm_q_outputs():
    """Debug HRM Q-learning outputs directly"""
    print("Debugging HRM Q-learning outputs...")
    
    # Create config with Q-learning enabled
    config = {
        'batch_size': 1,
        'seq_len': 64,
        'puzzle_emb_ndim': 0,
        'num_puzzle_identifiers': 10,
        'vocab_size': 1024,
        'H_cycles': 1,
        'L_cycles': 1,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 4,
        'halt_exploration_prob': 0.2,  # Higher exploration for testing
        'max_nodes': 20,
        'max_edges': 15,
        'num_node_types': 30,
        'max_spec_tokens': 64,
        'max_examples': 5,
        'forward_dtype': 'float32'
    }
    
    # Create model - just the HRM part
    hrm_model = ProgramSynthesisHRM(config)
    hrm_model.train()  # Set to training mode
    
    print(f"✓ HRM model created in training mode")
    print(f"✓ Halt max steps: {config['halt_max_steps']}")
    print(f"✓ Exploration prob: {config['halt_exploration_prob']}")
    
    # Get a single training example
    dataset_config = ProgramSynthesisDatasetConfig(
        seed=42,
        dataset_path="data/program-synthesis-100",
        rank=0,
        num_replicas=1
    )
    
    train_dataset = ProgramSynthesisDataset(dataset_config, "train")
    
    # Get first example
    for set_name, batch, global_batch_size in train_dataset:
        print(f"✓ Got training example")
        break
    
    # Test HRM forward pass directly
    carry = hrm_model.initial_carry(batch)
    
    print(f"\n=== HRM Forward Pass ===")
    step = 0
    while True:
        print(f"\nStep {step + 1}:")
        print(f"  Carry halted: {carry.halted}")
        print(f"  Carry steps: {carry.steps}")
        
        # Forward through HRM
        new_carry, outputs = hrm_model(carry, batch)
        
        print(f"  HRM output keys: {list(outputs.keys())}")
        
        # Check Q-learning specific outputs
        if 'q_halt_logits' in outputs:
            print(f"  q_halt_logits: {outputs['q_halt_logits'].item():.6f}")
        if 'q_continue_logits' in outputs:
            print(f"  q_continue_logits: {outputs['q_continue_logits'].item():.6f}")
        if 'target_q_continue' in outputs:
            print(f"  target_q_continue: {outputs['target_q_continue'].item():.6f}")
            print("  ✓ Q-learning targets generated!")
        else:
            print("  ✗ NO target_q_continue found")
        
        # Check computed losses
        targets = {'ast_targets': batch.get('ast_targets', {})}
        losses = hrm_model.compute_loss(outputs, targets)
        print(f"  Computed losses: {list(losses.keys())}")
        if 'q_loss' in losses:
            print(f"  q_loss value: {losses['q_loss'].item():.6f}")
        
        step += 1
        carry = new_carry
        
        if carry.halted.all() or step >= 6:
            print(f"\nFinished after {step} steps (halted: {carry.halted.all()})")
            break


if __name__ == "__main__":
    debug_hrm_q_outputs()