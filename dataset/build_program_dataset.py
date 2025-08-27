from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch_geometric.data import Data

# Local imports
from dataset.programs import get_program_registry, ProgramSpecification, ProgramRegistry
from dataset.graph_dataset import ProgramGraphDataset
from dataset.augment_programs import augment_registry

def _data_to_dict(data: Data) -> Dict[str, Any]:
    """Convert PyTorch Geometric Data object to a safe dictionary for serialization."""
    data_dict = {}

    # Get all attributes from the Data object
    for key, value in data.items():
        data_dict[key] = value

    # Also capture any additional attributes that might not be in the main storage
    # Skip computed properties that might fail on empty objects
    skip_attrs = {
        'node_offsets', 'edge_offsets', 'batch', 'ptr', 'face', 'edge_weight',
        'pos', 'norm', 'train_mask', 'val_mask', 'test_mask'
    }

    for attr_name in dir(data):
        if (not attr_name.startswith('_') and
            attr_name not in data_dict and
            attr_name not in skip_attrs and
            not callable(getattr(data, attr_name, None))):

            try:
                attr_value = getattr(data, attr_name)
                # Only store tensor, string, int, float, bool, or None values
                if isinstance(attr_value, (torch.Tensor, str, int, float, bool, type(None))):
                    data_dict[attr_name] = attr_value
            except (TypeError, AttributeError, RuntimeError):
                # Skip attributes that can't be accessed safely
                continue

    return data_dict


def _make_info_safe(info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert program info dictionary to contain only basic Python types."""
    from dataset.programs import Example

    safe_info = {}
    for key, value in info.items():
        if key == 'examples' and isinstance(value, list):
            # Convert Example objects to dictionaries
            safe_examples = []
            for example in value:
                if isinstance(example, Example):
                    safe_examples.append({
                        'input': example.input,
                        'output': example.output
                    })
                else:
                    safe_examples.append(example)
            safe_info[key] = safe_examples
        else:
            # Keep other values as-is (they should be basic types)
            safe_info[key] = value

    return safe_info


def _dict_to_data(data_dict: Dict[str, Any]) -> Data:
    """Convert a dictionary back to PyTorch Geometric Data object."""
    # Create new Data object
    data = Data()

    # Set all attributes from the dictionary
    for key, value in data_dict.items():
        setattr(data, key, value)

    return data


def generate_program_dataset(output_dir: str, num_samples: int = 200, seed: int = 123) -> None:
    """Generate a materialized program dataset with graph representations.

    Creates a dataset directory with:
    - Individual .pt files for each graph-program pair
    - metadata.json with dataset statistics and information
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a custom registry with code augmentation
    registry = get_program_registry()
    augmented_registry = augment_registry(registry, num_samples, seed)

    # Create dataset with augmented programs
    dataset = ProgramGraphDataset(
        registry=augmented_registry,
        programs_per_spec=1,  # Each augmented program is unique
        examples_per_program=3,
        seed=seed,
        cache_dir=None  # We'll save manually
    )

    print(f"Generated dataset with {len(dataset)} items")

    # Save each item as individual .pt file
    graphs_dir = Path(output_dir) / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    metadata = {
        "num_samples": len(dataset),
        "seed": seed,
        "examples_per_program": dataset.examples_per_program,
        "program_stats": dataset.get_statistics(),
        "files": []
    }

    for idx in range(len(dataset)):
        graph, info = dataset[idx]

        # Save graph and info
        filename = f"sample_{idx:06d}.pt"
        filepath = graphs_dir / filename

        # Convert Data object to safe dictionary for serialization
        graph_dict = _data_to_dict(graph)
        safe_info = _make_info_safe(info)

        torch.save({
            'graph_dict': graph_dict,
            'info': safe_info,
            'index': idx
        }, filepath)

        metadata["files"].append({
            "filename": filename,
            "spec_name": info["spec_name"],
            "description": info["description"],
            "num_nodes": graph.x.shape[0] if hasattr(graph, 'x') and graph.x is not None else 0,
            "num_edges": graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} samples")

    # Save metadata
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"- {len(dataset)} graph files in graphs/")
    print(f"- metadata.json with dataset information")


def load_sample(filepath: str) -> tuple[Data, Dict[str, Any], int]:
    """Load a single sample from a .pt file with safe Data reconstruction.

    Args:
        filepath: Path to the .pt file

    Returns:
        Tuple of (Data object, program info dict, sample index)
    """
    # Load with weights_only=True for security (no unsafe globals needed)
    saved_data = torch.load(filepath, weights_only=True)

    # Reconstruct Data object from dictionary
    graph = _dict_to_data(saved_data['graph_dict'])
    info = saved_data['info']
    index = saved_data['index']

    return graph, info, index


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a materialized program graph dataset")
    parser.add_argument("--out", type=str, default="data/programs-200", help="Output directory")
    parser.add_argument("--n", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    generate_program_dataset(args.out, num_samples=args.n, seed=args.seed)


__all__ = [
    'generate_program_dataset',
    'load_sample'
]


if __name__ == "__main__":
    main()


