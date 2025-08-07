#!/usr/bin/env python3

import os
import yaml
from programs import get_program_registry, ParameterType

def create_yaml_files():
    """Create YAML files for all programs in the registry"""
    registry = get_program_registry()

    # Create the program_definitions directory if it doesn't exist
    os.makedirs('program_definitions', exist_ok=True)

    # Create individual YAML files for each program
    for name, spec in registry.programs.items():
        yaml_file = f'program_definitions/{name}.yaml'

        # Convert to dictionary format and fix enum serialization
        program_data = spec.model_dump()

        # Fix enum serialization
        for input_param in program_data['inputs']:
            if 'type' in input_param and hasattr(input_param['type'], 'value'):
                input_param['type'] = input_param['type'].value

        for output_param in program_data['outputs']:
            if 'type' in output_param and hasattr(output_param['type'], 'value'):
                output_param['type'] = output_param['type'].value

        # Write to YAML file
        with open(yaml_file, 'w') as f:
            yaml.dump(program_data, f, default_flow_style=False, indent=2)

        print(f"Created {yaml_file}")

    # Create a combined YAML file with all programs
    combined_data = {
        'programs': []
    }

    for spec in registry.programs.values():
        program_data = spec.model_dump()

        # Fix enum serialization
        for input_param in program_data['inputs']:
            if 'type' in input_param and hasattr(input_param['type'], 'value'):
                input_param['type'] = input_param['type'].value

        for output_param in program_data['outputs']:
            if 'type' in output_param and hasattr(output_param['type'], 'value'):
                output_param['type'] = output_param['type'].value

        combined_data['programs'].append(program_data)

    with open('program_definitions/all_programs.yaml', 'w') as f:
        yaml.dump(combined_data, f, default_flow_style=False, indent=2)

    print("Created program_definitions/all_programs.yaml")

if __name__ == "__main__":
    create_yaml_files()
