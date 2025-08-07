"""
Dataset adapter for program synthesis to work with the existing pretrain.py infrastructure
"""

from typing import Dict, Any, Optional, Generator, Tuple
import os
import json
import torch
from dataclasses import dataclass

from models.program_synthesis_processor import ProgramSynthesisProcessor
from dataset.common import PuzzleDatasetMetadata


@dataclass
class ProgramSynthesisDatasetConfig:
    """Configuration for program synthesis dataset"""
    seed: int
    dataset_path: str
    rank: int = 0
    num_replicas: int = 1


class ProgramSynthesisDataset:
    """Dataset adapter for program synthesis to work with pretrain.py"""
    
    def __init__(self, config: ProgramSynthesisDatasetConfig, split: str):
        self.config = config
        self.split = split
        self.rank = config.rank
        self.num_replicas = config.num_replicas
        
        # Load dataset
        self.examples = self._load_examples()
        
        # Create processor  
        self.processor = ProgramSynthesisProcessor(
            max_examples=5,
            max_spec_tokens=64,
            max_nodes=20,
            max_edges=15,
            vocab_size=1024
        )
        
        # Create metadata compatible with pretrain.py
        self.metadata = PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            vocab_size=1024,
            seq_len=64,
            num_puzzle_identifiers=len(self.examples),
            total_groups=len(self.examples),
            mean_puzzle_examples=1.0,  # Each program is one example
            sets=['train'] if split == 'train' else ['eval']
        )
    
    def _load_examples(self):
        """Load program synthesis examples from dataset path"""
        import glob
        import yaml
        
        # Load from split directory (train/ or test/)
        split_dir = os.path.join(self.config.dataset_path, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Could not find split directory at {split_dir}")
        
        examples = []
        
        # Find all program*.yaml files
        yaml_files = glob.glob(os.path.join(split_dir, "program*.yaml"))
        
        for yaml_file in yaml_files:
            # Extract program number
            basename = os.path.basename(yaml_file)
            program_num = basename.replace('program', '').replace('.yaml', '')
            
            # Load specification from YAML
            with open(yaml_file, 'r') as f:
                spec = yaml.safe_load(f)
            
            # Load implementation from corresponding Python file
            py_file = os.path.join(split_dir, f"program{program_num}.py")
            if os.path.exists(py_file):
                with open(py_file, 'r') as f:
                    implementation = f.read()
                
                # Create example in expected format
                example = {
                    'specification': {
                        'name': f'program{program_num}',
                        'description': spec.get('description', ''),
                        'inputs': spec.get('inputs', []),
                        'outputs': spec.get('outputs', []),
                        'examples': spec.get('examples', [])
                    },
                    'implementation': implementation
                }
                examples.append(example)
        
        # Distributed sampling: take every num_replicas-th example starting from rank
        if self.num_replicas > 1:
            examples = examples[self.rank::self.num_replicas]
        
        return examples
    
    def __iter__(self) -> Generator[Tuple[str, Dict[str, torch.Tensor], int], None, None]:
        """
        Yield batches in format expected by pretrain.py:
        (set_name, batch_dict, global_batch_size)
        """
        # For simplicity, yield one example at a time
        # In production, you'd want proper batching
        
        # Map 'test' directory to 'eval' metrics naming for user preference
        set_name = 'train' if self.split == 'train' else 'eval'
        
        for example in self.examples:
            # Convert to HRM batch format
            batch = self.processor.create_hrm_batch([example])
            
            # Add labels for compatibility (empty for program synthesis)
            batch['labels'] = torch.full((1, 64), -100, dtype=torch.long)
            
            yield set_name, batch, 1  # batch_size = 1


def PuzzleDataset(config: ProgramSynthesisDatasetConfig, split: str, **kwargs):
    """
    Factory function that mimics the signature expected by pretrain.py
    """
    # Ignore extra kwargs that might be passed from pretrain.py
    return ProgramSynthesisDataset(config, split)