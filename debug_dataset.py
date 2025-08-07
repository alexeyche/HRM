#!/usr/bin/env python3
"""
Debug dataset loading to see what's happening with evaluation data
"""

from program_synthesis_dataset import ProgramSynthesisDataset, ProgramSynthesisDatasetConfig

def debug_dataset():
    """Debug dataset loading"""
    print("Testing dataset loading...")
    
    # Test train dataset
    train_config = ProgramSynthesisDatasetConfig(
        seed=42,
        dataset_path="data/program-synthesis-100",
        rank=0,
        num_replicas=1
    )
    
    train_dataset = ProgramSynthesisDataset(train_config, "train")
    print(f"✓ Train dataset loaded with {len(train_dataset.examples)} examples")
    
    # Test test dataset 
    test_dataset = ProgramSynthesisDataset(train_config, "test")
    print(f"✓ Test dataset loaded with {len(test_dataset.examples)} examples")
    
    # Test first few items from each
    print("\n=== Train dataset items ===")
    count = 0
    for set_name, batch, global_batch_size in train_dataset:
        if count >= 2:
            break
        print(f"Set name: {set_name}, Global batch size: {global_batch_size}")
        print(f"Batch keys: {list(batch.keys())}")
        print(f"AST targets keys: {list(batch.get('ast_targets', {}).keys())}")
        count += 1
    
    print("\n=== Test dataset items ===")
    count = 0
    for set_name, batch, global_batch_size in test_dataset:
        if count >= 2:
            break
        print(f"Set name: {set_name}, Global batch size: {global_batch_size}")
        print(f"Batch keys: {list(batch.keys())}")
        print(f"AST targets keys: {list(batch.get('ast_targets', {}).keys())}")
        count += 1

if __name__ == "__main__":
    debug_dataset()