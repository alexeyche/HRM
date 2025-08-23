"""
Graph Dataset for Program Synthesis

Dataset class that converts programs from programs.py into graph representations
for training graph neural networks.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import json
import os
from pathlib import Path

from dataset.programs import (
    ProgramSpecification, ProgramRegistry, get_program_registry, Example
)
from dataset.ast_converter import program_to_graph, ASTToGraphConverter


class ProgramGraphDataset(Dataset):
    """
    Dataset that yields (graph, program_info) pairs.
    
    Converts program implementations from ProgramRegistry into graph representations
    suitable for training Graph Neural Networks.
    """
    
    def __init__(
        self,
        registry: Optional[ProgramRegistry] = None,
        programs_per_spec: int = 5,
        examples_per_program: int = 3,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
        converter_kwargs: Optional[Dict[str, Any]] = None,
        target_total_samples: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            registry: Program registry to use (default: get_program_registry())
            programs_per_spec: How many graph instances to generate per program spec
            examples_per_program: How many input/output examples to include
            seed: Random seed for reproducibility
            cache_dir: Directory to cache processed graphs
            converter_kwargs: Additional arguments for ASTToGraphConverter
            target_total_samples: If provided, automatically calculate programs_per_spec to reach target
        """
        self.registry = registry or get_program_registry()
        
        # Auto-calculate programs_per_spec if target is specified
        if target_total_samples is not None:
            num_programs = len(self.registry.programs)
            self.programs_per_spec = max(1, target_total_samples // num_programs)
            print(f"Auto-calculated {self.programs_per_spec} samples per program for {target_total_samples} target samples")
        else:
            self.programs_per_spec = programs_per_spec
            
        self.examples_per_program = examples_per_program
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.converter_kwargs = converter_kwargs or {}
        self.target_total_samples = target_total_samples
        
        # Initialize converter
        self.converter = ASTToGraphConverter()
        
        # Build dataset items
        self._build_dataset()
        
        # Setup caching if enabled
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_dataset(self):
        """Build the dataset by processing all program specifications."""
        self.items: List[Dict[str, Any]] = []
        
        for spec_name, spec in self.registry.programs.items():
            # Generate examples for this program specification
            examples = spec.generate_examples(
                self.examples_per_program, 
                seed=self.seed
            )
            
            # Create multiple graph instances per specification
            for instance_idx in range(self.programs_per_spec):
                item = {
                    'spec_name': spec_name,
                    'spec': spec,
                    'examples': examples,
                    'instance_idx': instance_idx,
                    'program_code': spec.implementation,
                    'description': spec.description,
                    'input_types': [inp.type for inp in spec.inputs],
                    'output_types': [out.type for out in spec.outputs],
                }
                self.items.append(item)
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[Data, Dict[str, Any]]:
        """
        Get a (graph, program_info) pair.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (PyTorch Geometric Data object, program information dict)
        """
        if idx >= len(self.items):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.items)}")
        
        item = self.items[idx]
        
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{item['spec_name']}_{item['instance_idx']}.pt"
            if cache_file.exists():
                try:
                    cached_data = torch.load(cache_file)
                    return cached_data['graph'], cached_data['info']
                except Exception:
                    # Cache corrupted, continue with generation
                    pass
        
        # Convert program to graph
        try:
            graph = program_to_graph(item['program_code'])
        except Exception as e:
            # If conversion fails, create empty graph with error info
            print(f"Warning: Failed to convert program {item['spec_name']}: {e}")
            graph = Data(
                x=torch.empty(0, 6),  # Match feature dimension from converter
                edge_index=torch.empty(2, 0, dtype=torch.long),
                conversion_error=str(e)
            )
        
        # Add program-specific information to graph
        graph.spec_name = item['spec_name']
        graph.description = item['description']
        graph.program_code = item['program_code']
        
        # Create program info dictionary
        program_info = {
            'spec_name': item['spec_name'],
            'description': item['description'],
            'examples': item['examples'],
            'input_types': [t.value for t in item['input_types']],
            'output_types': [t.value for t in item['output_types']],
            'program_code': item['program_code'],
            'instance_idx': item['instance_idx'],
        }
        
        # Cache the result if caching is enabled
        if self.cache_dir:
            cache_file = self.cache_dir / f"{item['spec_name']}_{item['instance_idx']}.pt"
            try:
                torch.save({
                    'graph': graph,
                    'info': program_info
                }, cache_file)
            except Exception:
                # Ignore caching errors
                pass
        
        return graph, program_info
    
    def get_program_names(self) -> List[str]:
        """Get list of all program names in the dataset."""
        return list(set(item['spec_name'] for item in self.items))
    
    def get_examples_for_program(self, program_name: str) -> List[Example]:
        """Get examples for a specific program."""
        for item in self.items:
            if item['spec_name'] == program_name:
                return item['examples']
        raise ValueError(f"Program {program_name} not found in dataset")
    
    def filter_by_category(self, categories: List[str]) -> 'ProgramGraphDataset':
        """
        Create a filtered dataset containing only programs from specified categories.
        
        Categories are inferred from program names (e.g., 'array_*', 'string_*').
        """
        filtered_items = []
        
        for item in self.items:
            program_name = item['spec_name']
            program_category = program_name.split('_')[0] if '_' in program_name else 'misc'
            
            if program_category in categories:
                filtered_items.append(item)
        
        # Create new dataset with filtered items
        new_dataset = ProgramGraphDataset.__new__(ProgramGraphDataset)
        new_dataset.registry = self.registry
        new_dataset.programs_per_spec = self.programs_per_spec
        new_dataset.examples_per_program = self.examples_per_program
        new_dataset.seed = self.seed
        new_dataset.cache_dir = self.cache_dir
        new_dataset.converter_kwargs = self.converter_kwargs
        new_dataset.converter = self.converter
        new_dataset.items = filtered_items
        
        return new_dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        program_names = self.get_program_names()
        categories = {}
        
        for name in program_names:
            category = name.split('_')[0] if '_' in name else 'misc'
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_items': len(self.items),
            'unique_programs': len(program_names),
            'programs_per_spec': self.programs_per_spec,
            'examples_per_program': self.examples_per_program,
            'categories': categories,
            'program_names': sorted(program_names)
        }


class ProgramGraphCollator:
    """
    Collate function for batching graph data.
    
    Handles variable-sized graphs and creates proper batches for PyTorch Geometric.
    """
    
    def __init__(self, add_program_info: bool = True):
        """
        Initialize collator.
        
        Args:
            add_program_info: Whether to include program info in batch
        """
        self.add_program_info = add_program_info
    
    def __call__(self, batch: List[Tuple[Data, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Collate a batch of (graph, program_info) pairs.
        
        Args:
            batch: List of (Data, program_info) tuples
            
        Returns:
            Dictionary with batched data
        """
        from torch_geometric.data import Batch
        
        graphs, program_infos = zip(*batch)
        
        # Batch the graphs
        try:
            batched_graphs = Batch.from_data_list(list(graphs))
        except Exception as e:
            # If batching fails, create empty batch
            print(f"Warning: Failed to batch graphs: {e}")
            batched_graphs = Batch(
                x=torch.empty(0, 6),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                batch=torch.empty(0, dtype=torch.long)
            )
        
        result = {'graphs': batched_graphs}
        
        if self.add_program_info:
            result['program_infos'] = list(program_infos)
            
            # Add useful batch-level information
            result['spec_names'] = [info['spec_name'] for info in program_infos]
            result['batch_size'] = len(batch)
        
        return result


def create_train_val_datasets(
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    **dataset_kwargs
) -> Tuple[ProgramGraphDataset, ProgramGraphDataset]:
    """
    Create train and validation datasets.
    
    Args:
        train_ratio: Fraction of programs to use for training
        seed: Random seed for split
        **dataset_kwargs: Arguments passed to ProgramGraphDataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    # Get all program names
    registry = dataset_kwargs.get('registry') or get_program_registry()
    all_programs = list(registry.programs.keys())
    random.shuffle(all_programs)
    
    # Split programs
    split_idx = int(len(all_programs) * train_ratio)
    train_programs = all_programs[:split_idx]
    val_programs = all_programs[split_idx:]
    
    # Create filtered registries
    train_registry = ProgramRegistry()
    val_registry = ProgramRegistry()
    
    for name in train_programs:
        train_registry.register(registry.programs[name])
    
    for name in val_programs:
        val_registry.register(registry.programs[name])
    
    # Create datasets
    train_dataset = ProgramGraphDataset(registry=train_registry, **dataset_kwargs)
    val_dataset = ProgramGraphDataset(registry=val_registry, **dataset_kwargs)
    
    return train_dataset, val_dataset


def demo_dataset():
    """Demo function to show dataset usage."""
    print("Creating ProgramGraphDataset...")
    
    # Create dataset with a few programs
    dataset = ProgramGraphDataset(programs_per_spec=2, examples_per_program=2)
    
    print(f"Dataset size: {len(dataset)}")
    print("Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nFirst few items:")
    for i in range(min(3, len(dataset))):
        graph, info = dataset[i]
        print(f"  Item {i}:")
        print(f"    Program: {info['spec_name']}")
        print(f"    Description: {info['description']}")
        print(f"    Graph nodes: {graph.x.shape[0] if hasattr(graph, 'x') else 0}")
        print(f"    Graph edges: {graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0}")
        print()


if __name__ == "__main__":
    demo_dataset()