#!/usr/bin/env python3
"""
Debug Q-learning to see why q_loss is flat
"""

import torch
from models.program_synthesis_processor import ProgramSynthesisProcessor
from models.program_synthesis_hrm import ProgramSynthesisHRM
from models.losses import ProgramSynthesisLossHead
from program_synthesis_dataset import ProgramSynthesisDataset, ProgramSynthesisDatasetConfig


def debug_q_learning():
    """Debug Q-learning loss computation"""
    print("Debugging Q-learning loss...")
    
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
        'halt_exploration_prob': 0.1,  # Enable exploration for Q-learning
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
    print(f"✓ Halt max steps: {config['halt_max_steps']}")
    print(f"✓ Exploration prob: {config['halt_exploration_prob']}")
    
    # Create train dataset (for training mode)
    dataset_config = ProgramSynthesisDatasetConfig(
        seed=42,
        dataset_path="data/program-synthesis-100",
        rank=0,
        num_replicas=1
    )
    
    train_dataset = ProgramSynthesisDataset(dataset_config, "train")
    print(f"✓ Train dataset loaded with {len(train_dataset.examples)} examples")
    
    # Set model to training mode (important for Q-learning)
    loss_head.train()
    print("✓ Model set to training mode")
    
    # Test Q-learning on training examples
    with torch.no_grad():
        for i, (set_name, batch, global_batch_size) in enumerate(train_dataset):
            if i >= 3:  # Just test first 3 examples
                break
                
            print(f"\n=== Training Example {i+1} ===")
            print(f"Set name: {set_name}")
            
            # Initialize carry
            carry = loss_head.initial_carry(batch)
            
            # Enable gradients for this forward pass to see Q-learning
            step = 0
            final_outputs = None
            final_metrics = None
            with torch.enable_grad():
                while True:
                    carry, loss, metrics, preds, all_finish = loss_head(
                        carry=carry, 
                        batch=batch, 
                        return_keys=[]
                    )
                    
                    step += 1
                    if all_finish:
                        final_outputs = carry
                        final_metrics = metrics
                        print(f"  Finished after {step} steps")
                        break
                        
                    if step > 10:  # Safety break
                        print("    Breaking due to too many steps")
                        final_outputs = carry
                        final_metrics = metrics
                        break
            
            print(f"  Final halted state: {final_outputs.halted}")
            print(f"  Final steps: {final_outputs.steps}")
            
            # Check if Q-learning components are present
            print(f"  Metrics keys: {list(final_metrics.keys())}")
            
            # Look for Q-learning related metrics
            q_metrics = {k: v for k, v in final_metrics.items() if 'q_' in k.lower()}
            if q_metrics:
                print(f"  Q-learning metrics found:")
                for k, v in q_metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    print(f"    {k}: {val}")
            else:
                print(f"  NO Q-learning metrics found!")
    
    # Test with explicit training mode check
    print(f"\n=== Model Training State Check ===")
    print(f"Model training mode: {loss_head.training}")
    print(f"Inner HRM training mode: {loss_head.model.hrm.training}")


if __name__ == "__main__":
    debug_q_learning()