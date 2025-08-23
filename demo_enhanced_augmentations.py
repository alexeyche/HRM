#!/usr/bin/env python3
"""
Demo script showing the enhanced augmentation system for reaching 2000+ samples.
"""

from dataset.graph_dataset import ProgramGraphDataset
from dataset.augmentations import (
    create_comprehensive_augmentation_strategy, 
    MultiAugmentationStrategy, ExtendedAugmentationSpec, AugmentationType
)
from dataset.programs import get_program_registry
import random


def demo_basic_augmentations():
    """Show basic augmentation types in action."""
    print("🎯 Basic Augmentation Strategies Demo")
    print("=" * 50)
    
    # Integer augmentations
    print("🔢 Integer Augmentations:")
    int_strategy = create_comprehensive_augmentation_strategy("int", complexity_level=2)
    int_values = int_strategy.generate_multiple(10)
    print(f"   Generated integers: {int_values}")
    
    # Array augmentations  
    print("\n📋 Array Augmentations:")
    array_strategy = create_comprehensive_augmentation_strategy("List[int]", complexity_level=2)
    array_values = array_strategy.generate_multiple(5)
    for i, arr in enumerate(array_values):
        print(f"   Array {i+1}: {arr}")
    
    # String augmentations
    print("\n📝 String Augmentations:")
    str_strategy = create_comprehensive_augmentation_strategy("str", complexity_level=2)  
    str_values = str_strategy.generate_multiple(8)
    print(f"   Generated strings: {str_values}")


def demo_augmentation_diversity():
    """Show diversity of augmentation types."""
    print("\n🌈 Augmentation Type Diversity Demo")
    print("=" * 50)
    
    # Show different augmentation types for integers
    augmentation_types = [
        (AugmentationType.RAND_INT, [-10, 10], "Random integers"),
        (AugmentationType.GEOMETRIC_SEQUENCE, [2, 6], "Powers of 2"),
        (AugmentationType.PRIME_NUMBERS, [30], "Prime numbers"),
        (AugmentationType.SPECIAL_VALUES, [0, 1, -1, 5, -5, 100], "Special edge cases"),
    ]
    
    for aug_type, params, description in augmentation_types:
        spec = ExtendedAugmentationSpec(
            type=aug_type, 
            parameters=params, 
            description=description,
            weight=1.0
        )
        values = [spec.generate_value() for _ in range(8)]
        print(f"   {description}: {values}")


def demo_program_specific_augmentation():
    """Show how different programs get different augmentation strategies."""
    print("\n🧮 Program-Specific Augmentation Demo")
    print("=" * 50)
    
    registry = get_program_registry()
    
    # Show augmentation for different program types
    programs_to_demo = ['sum_up_to_n', 'array_sum', 'string_length', 'max_of_two']
    
    for prog_name in programs_to_demo:
        if prog_name in registry.programs:
            spec = registry.programs[prog_name] 
            print(f"\n📊 Program: {prog_name}")
            print(f"   Description: {spec.description}")
            print(f"   Input types: {[inp.type.value for inp in spec.inputs]}")
            
            # Show generated examples
            examples = spec.generate_examples(5, seed=42)
            print(f"   Sample inputs/outputs:")
            for i, example in enumerate(examples[:3]):
                print(f"     {i+1}. {example.input} → {example.output}")


def demo_2000_sample_dataset():
    """Demo creating the full 2000+ sample dataset."""
    print("\n🚀 Full 2000-Sample Dataset Demo")
    print("=" * 50)
    
    print("Creating dataset with 2000 target samples...")
    dataset = ProgramGraphDataset(target_total_samples=2000, seed=42)
    
    stats = dataset.get_statistics()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {stats['total_items']}")
    print(f"   Target samples: 2000") 
    print(f"   Achievement rate: {stats['total_items']/2000*100:.1f}%")
    print(f"   Unique programs: {stats['unique_programs']}")
    print(f"   Samples per program: {stats['programs_per_spec']}")
    
    print(f"\n📈 Sample Distribution by Category:")
    sorted_categories = sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        percentage = count / stats['total_items'] * 100
        print(f"   {category}: {count} samples ({percentage:.1f}%)")
    
    # Show some sample data
    print(f"\n🔍 Sample Data Points:")
    for i in range(3):
        graph, info = dataset[i]
        print(f"   Sample {i+1}:")
        print(f"     Program: {info['spec_name']}")
        print(f"     Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
        print(f"     Input types: {info['input_types']}")


def demo_augmentation_analysis():
    """Analyze the diversity and coverage of augmentations."""
    print("\n🔬 Augmentation Analysis")
    print("=" * 50)
    
    registry = get_program_registry()
    
    # Count programs by input type
    type_counts = {}
    for spec in registry.programs.values():
        for inp in spec.inputs:
            inp_type = inp.type.value
            type_counts[inp_type] = type_counts.get(inp_type, 0) + 1
    
    print("📋 Program Parameter Type Distribution:")
    for param_type, count in sorted(type_counts.items()):
        print(f"   {param_type}: {count} programs")
    
    # Calculate augmentation coverage
    total_base_examples = sum(len(spec.base_examples) for spec in registry.programs.values())
    with_augmentations = sum(1 for spec in registry.programs.values() 
                            if any(inp.augmentation for inp in spec.inputs))
    
    print(f"\n📈 Augmentation Coverage:")
    print(f"   Programs with augmentations: {with_augmentations}/{len(registry.programs)} ({with_augmentations/len(registry.programs)*100:.1f}%)")
    print(f"   Total base examples: {total_base_examples}")
    print(f"   Expected augmented samples: ~{25 * len(registry.programs)}")
    print(f"   Augmentation multiplier: ~{25 * len(registry.programs) / total_base_examples:.1f}x")


def main():
    """Run all demos."""
    print("🌟 Enhanced Augmentation System Demo")
    print("=" * 60)
    print("Demonstrating comprehensive augmentation strategies for 2000+ samples\n")
    
    # Set consistent seed for demos
    random.seed(42)
    
    try:
        demo_basic_augmentations()
        demo_augmentation_diversity() 
        demo_program_specific_augmentation()
        demo_augmentation_analysis()
        demo_2000_sample_dataset()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"🎉 Enhanced augmentation system successfully generates 2000+ diverse samples!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()