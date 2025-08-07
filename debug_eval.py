#!/usr/bin/env python3
"""
Debug evaluation to see what's happening with metrics
"""

import torch
from models.program_synthesis_processor import ProgramSynthesisProcessor
from models.program_synthesis_hrm import ProgramSynthesisHRM
from models.losses import ProgramSynthesisLossHead
from program_synthesis_dataset import ProgramSynthesisDataset, ProgramSynthesisDatasetConfig


def debug_evaluation():
    """Debug evaluation metrics"""
    print("Debugging evaluation metrics...")
    
    # Create config
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
        'halt_exploration_prob': 0.0,
        'max_nodes': 20,
        'max_edges': 15,
        'num_node_types': 30,
        'max_spec_tokens': 64,
        'max_examples': 5,
        'forward_dtype': 'float32'
    }
    
    # Create model
    model = ProgramSynthesisHRM(config)
    loss_head = ProgramSynthesisLossHead(model)
    
    print(f"✓ Model created")
    
    # Create test dataset
    dataset_config = ProgramSynthesisDatasetConfig(
        seed=42,
        dataset_path="data/program-synthesis-100",
        rank=0,
        num_replicas=1
    )
    
    test_dataset = ProgramSynthesisDataset(dataset_config, "test")
    print(f"✓ Test dataset loaded with {len(test_dataset.examples)} examples")
    
    # Set model to eval mode
    loss_head.eval()
    
    # Test evaluation on a few examples
    with torch.no_grad():
        for i, (set_name, batch, global_batch_size) in enumerate(test_dataset):
            if i >= 3:  # Just test first 3 examples
                break
                
            print(f"\n=== Example {i+1} ===")
            print(f"Set name: {set_name}")
            print(f"Global batch size: {global_batch_size}")
            
            # Initialize carry
            carry = loss_head.initial_carry(batch)
            print(f"Carry halted initially: {carry.halted}")
            print(f"Carry steps initially: {carry.steps}")
            
            # Run evaluation loop
            step = 0
            while True:
                carry, loss, metrics, preds, all_finish = loss_head(
                    carry=carry, 
                    batch=batch, 
                    return_keys=[]
                )
                
                step += 1
                print(f"  Step {step}:")
                print(f"    Loss: {loss.item():.4f}")
                print(f"    Metrics: {[(k, v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()]}")
                print(f"    All finished: {all_finish}")
                print(f"    Carry halted: {carry.halted}")
                print(f"    Carry steps: {carry.steps}")
                
                if all_finish:
                    break
                    
                if step > 10:  # Safety break
                    print("    Breaking due to too many steps")
                    break
            
            print(f"Total steps for example {i+1}: {step}")


if __name__ == "__main__":
    debug_evaluation()