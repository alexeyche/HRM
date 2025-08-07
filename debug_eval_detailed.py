#!/usr/bin/env python3
"""
Debug detailed evaluation to understand flat metrics issue
"""

import torch
from models.program_synthesis_processor import ProgramSynthesisProcessor
from models.program_synthesis_hrm import ProgramSynthesisHRM
from models.losses import ProgramSynthesisLossHead
from program_synthesis_dataset import ProgramSynthesisDataset, ProgramSynthesisDatasetConfig


def debug_evaluation_aggregation():
    """Debug evaluation metrics aggregation"""
    print("Debugging evaluation metrics aggregation...")
    
    # Create simplified config
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
    
    # Collect all metrics from evaluation examples (like real training does)
    accumulated_metrics = {}
    total_count = 0
    
    with torch.no_grad():
        for i, (set_name, batch, global_batch_size) in enumerate(test_dataset):
            if i >= 5:  # Just test first 5 examples
                break
                
            print(f"\n=== Example {i+1} ===")
            print(f"Set name: {set_name}")
            
            # Initialize carry and run evaluation steps
            carry = loss_head.initial_carry(batch)
            
            # Run until completion (mimicking evaluation loop)
            step = 0
            final_metrics = None
            while True:
                carry, loss, metrics, preds, all_finish = loss_head(
                    carry=carry, 
                    batch=batch, 
                    return_keys=[]
                )
                
                step += 1
                if all_finish:
                    final_metrics = metrics
                    print(f"  Final metrics after {step} steps:")
                    for k, v in final_metrics.items():
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        print(f"    {k}: {val:.6f}")
                    break
                    
                if step > 10:  # Safety break
                    print("    Breaking due to too many steps")
                    final_metrics = metrics
                    break
            
            # Accumulate metrics (like the real evaluation does)
            if final_metrics:
                for k, v in final_metrics.items():
                    if k not in accumulated_metrics:
                        accumulated_metrics[k] = 0.0
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    accumulated_metrics[k] += val
                
                total_count += final_metrics.get('count', 1)
    
    print(f"\n=== ACCUMULATED METRICS (like evaluation does) ===")
    print(f"Total count: {total_count}")
    
    # Apply same normalization as real evaluation
    count = max(accumulated_metrics.pop("count", total_count), 1)
    normalized_metrics = {}
    for k, v in accumulated_metrics.items():
        normalized_metrics[f"eval/{k}"] = v / count
        print(f"eval/{k}: {v / count:.6f}")
    
    print("\n=== ANALYSIS ===")
    print("If these metrics are identical across different runs,")
    print("it suggests the model weights aren't changing between evaluations.")


if __name__ == "__main__":
    debug_evaluation_aggregation()