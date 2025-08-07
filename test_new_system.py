#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset'))

from programs import get_program_registry, ProgramSpecification, test_all_programs
from build_program_synthesis_dataset import PROGRAM_TEMPLATES, generate_program_examples

def test_new_system():
    """Test the new Pydantic-based system"""
    print("Testing new Pydantic-based system...")

    # Test the registry
    registry = get_program_registry()
    print(f"Registry has {len(registry.programs)} programs")

    # Test a specific program
    spec = registry.get("sum_up_to_n")
    if spec:
        print(f"Found program: {spec.name}")
        print(f"Description: {spec.description}")
        print(f"Inputs: {[inp.type.value for inp in spec.inputs]}")
        print(f"Outputs: {[out.type.value for out in spec.outputs]}")

        # Test example generation
        examples = spec.generate_examples(10, seed=42)
        print(f"Generated {len(examples)} examples")
        for i, ex in enumerate(examples[:3]):
            print(f"  Example {i+1}: {ex.input} -> {ex.output}")

    # Test legacy compatibility
    print(f"\nLegacy PROGRAM_TEMPLATES has {len(PROGRAM_TEMPLATES)} templates")

    # Test the generate_program_examples function
    if "sum_up_to_n" in PROGRAM_TEMPLATES:
        template = PROGRAM_TEMPLATES["sum_up_to_n"]
        examples = generate_program_examples("sum_up_to_n", template, 5)
        print(f"Generated {len(examples)} examples via legacy function")
        for i, ex in enumerate(examples[:3]):
            print(f"  Example {i+1}: {ex['input']} -> {ex['output']}")

    print("\nTest completed successfully!")

    # Test all programs in the registry
    print("\n" + "="*50)
    print("TESTING ALL PROGRAMS IN REGISTRY")
    print("="*50)
    test_all_programs()

if __name__ == "__main__":
    test_new_system()
