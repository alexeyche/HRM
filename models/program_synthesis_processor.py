"""
Program Synthesis Input/Output Processing for HRM

This module handles input and output processing for the program synthesis tasks,
adapting the multi-modal specification format to HRM's expected input structure
and processing graph-based AST outputs.
"""

from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from dataset.build_program_synthesis_dataset import TYPE_VOCAB, NodeType, ASTSimplifier


@dataclass
class ProgramSynthesisExample:
    """Single program synthesis training example"""
    specification: Dict[str, Any]  # Program spec with types, examples, description
    target_ast: Dict[str, Any]     # Target AST representation
    flattened_ast: torch.Tensor    # Flattened AST tensor for training


class ProgramSynthesisProcessor:
    """Processes program synthesis data for HRM model input/output"""
    
    def __init__(self, 
                 max_examples: int = 10,
                 max_spec_tokens: int = 64,
                 max_nodes: int = 30,
                 max_edges: int = 25,
                 vocab_size: int = 1024):
        self.max_examples = max_examples
        self.max_spec_tokens = max_spec_tokens
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.vocab_size = vocab_size
        
        # AST processor
        self.ast_simplifier = ASTSimplifier(max_nodes=max_nodes, max_edges=max_edges)
        
        # Vocabulary mappings
        self.type_vocab = TYPE_VOCAB.copy()
        self.reverse_type_vocab = {v: k for k, v in self.type_vocab.items()}
        
        # Special tokens for specification encoding
        self.special_tokens = {
            '<PAD>': 0,
            '<START>': 1, 
            '<END>': 2,
            '<SEP>': 3,
            '<INPUT>': 4,
            '<OUTPUT>': 5,
            '<EXAMPLE>': 6,
            '<TYPE>': 7,
            '<DESC>': 8,
        }
        
        # Build unified vocabulary (combining special tokens, types, and values)
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build unified vocabulary for specification encoding"""
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        current_id = max(self.special_tokens.values()) + 1
        
        # Add type tokens
        for type_name in self.type_vocab.keys():
            if type_name not in self.token_to_id:
                self.token_to_id[type_name] = current_id
                self.id_to_token[current_id] = type_name
                current_id += 1
        
        # Add common value tokens (numbers, basic strings)
        common_values = [str(i) for i in range(-100, 101)]  # Numbers -100 to 100
        common_values.extend(['True', 'False', 'null', 'empty'])
        
        for value in common_values:
            if value not in self.token_to_id and current_id < self.vocab_size:
                self.token_to_id[value] = current_id
                self.id_to_token[current_id] = value
                current_id += 1
    
    def encode_specification(self, spec: Dict[str, Any]) -> torch.Tensor:
        """
        Encode program specification to token sequence
        
        Format: <START> <INPUT> type1 <OUTPUT> type2 <DESC> desc_tokens 
                <EXAMPLE> input_val <SEP> output_val ... <END>
        """
        tokens = []
        
        # Start token
        tokens.append(self.token_to_id['<START>'])
        
        # Input types
        tokens.append(self.token_to_id['<INPUT>'])
        for inp in spec.get('inputs', []):
            type_name = inp.get('type', 'int')
            if type_name in self.token_to_id:
                tokens.append(self.token_to_id[type_name])
        
        # Output types  
        tokens.append(self.token_to_id['<OUTPUT>'])
        for out in spec.get('outputs', []):
            type_name = out.get('type', 'int')
            if type_name in self.token_to_id:
                tokens.append(self.token_to_id[type_name])
        
        # Description (simplified - just add DESC token)
        tokens.append(self.token_to_id['<DESC>'])
        
        # Examples (up to max_examples)
        examples = spec.get('examples', [])[:self.max_examples]
        for example in examples:
            tokens.append(self.token_to_id['<EXAMPLE>'])
            
            # Input value
            input_val = example.get('input')
            if isinstance(input_val, list):
                for val in input_val:
                    val_str = str(val)
                    if val_str in self.token_to_id:
                        tokens.append(self.token_to_id[val_str])
            else:
                val_str = str(input_val)
                if val_str in self.token_to_id:
                    tokens.append(self.token_to_id[val_str])
            
            tokens.append(self.token_to_id['<SEP>'])
            
            # Output value
            output_val = example.get('output')
            if isinstance(output_val, list):
                for val in output_val:
                    val_str = str(val)
                    if val_str in self.token_to_id:
                        tokens.append(self.token_to_id[val_str])
            else:
                val_str = str(output_val)
                if val_str in self.token_to_id:
                    tokens.append(self.token_to_id[val_str])
        
        # End token
        tokens.append(self.token_to_id['<END>'])
        
        # Pad or truncate to max_spec_tokens
        if len(tokens) < self.max_spec_tokens:
            tokens.extend([self.token_to_id['<PAD>']] * (self.max_spec_tokens - len(tokens)))
        else:
            tokens = tokens[:self.max_spec_tokens]
            tokens[-1] = self.token_to_id['<END>']  # Ensure end token
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def process_ast_target(self, implementation: str) -> Dict[str, torch.Tensor]:
        """
        Process target AST implementation into graph representation
        
        Returns:
            Dict with node_exists, adjacency, node_types, node_values tensors
        """
        # Generate sparse AST representation
        sparse_ast = self.ast_simplifier.ast_to_sparse_representation(implementation)
        
        # Convert to fixed-size tensors for training
        num_nodes = sparse_ast['num_nodes']
        
        # Node existence mask
        node_exists = torch.zeros(self.max_nodes, dtype=torch.bool)
        node_exists[:num_nodes] = True
        
        # Node types (padded)
        node_types = torch.zeros(self.max_nodes, dtype=torch.long)
        node_types[:num_nodes] = torch.from_numpy(sparse_ast['node_types'])
        
        # Node values (padded)
        node_values = torch.zeros(self.max_nodes, dtype=torch.long) 
        node_values[:num_nodes] = torch.from_numpy(sparse_ast['node_values'])
        
        # Node parameters (padded)
        node_params = torch.zeros(self.max_nodes, dtype=torch.long)
        node_params[:num_nodes] = torch.from_numpy(sparse_ast['node_params'])
        
        # Adjacency matrix
        adjacency = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.bool)
        if len(sparse_ast['edge_list']) > 0:
            edges = sparse_ast['edge_list']
            valid_edges = edges[edges[:, 1] < self.max_nodes]  # Filter edges within bounds
            if len(valid_edges) > 0:
                adjacency[valid_edges[:, 0], valid_edges[:, 1]] = True
        
        return {
            'node_exists': node_exists,
            'adjacency': adjacency, 
            'node_types': node_types,
            'node_values': node_values,
            'node_params': node_params,
            'num_nodes': torch.tensor(num_nodes, dtype=torch.long)
        }
    
    def create_hrm_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Convert program synthesis examples to HRM model input format
        
        Args:
            examples: List of program synthesis examples from dataset
            
        Returns:
            Batch dict compatible with HRM model
        """
        batch_size = len(examples)
        
        # Process specifications
        inputs = torch.zeros((batch_size, self.max_spec_tokens), dtype=torch.long)
        
        # Process AST targets 
        ast_targets = {
            'node_exists': torch.zeros((batch_size, self.max_nodes), dtype=torch.bool),
            'adjacency': torch.zeros((batch_size, self.max_nodes, self.max_nodes), dtype=torch.bool),
            'node_types': torch.zeros((batch_size, self.max_nodes), dtype=torch.long),
            'node_values': torch.zeros((batch_size, self.max_nodes), dtype=torch.long),
            'node_params': torch.zeros((batch_size, self.max_nodes), dtype=torch.long),
            'num_nodes': torch.zeros(batch_size, dtype=torch.long)
        }
        
        # Puzzle identifiers (simple enumeration for now)
        puzzle_identifiers = torch.arange(batch_size, dtype=torch.long)
        
        for i, example in enumerate(examples):
            # Encode specification
            inputs[i] = self.encode_specification(example['specification'])
            
            # Process target AST
            ast_data = self.process_ast_target(example['implementation'])
            for key in ast_targets:
                ast_targets[key][i] = ast_data[key]
        
        return {
            'inputs': inputs,
            'puzzle_identifiers': puzzle_identifiers,
            'ast_targets': ast_targets
        }
    
    def decode_ast_output(self, 
                         node_exists: torch.Tensor,
                         adjacency: torch.Tensor, 
                         node_types: torch.Tensor,
                         node_values: torch.Tensor) -> Dict[str, Any]:
        """
        Decode HRM AST output back to interpretable format
        
        Args:
            node_exists: [max_nodes] boolean mask
            adjacency: [max_nodes, max_nodes] adjacency matrix
            node_types: [max_nodes] node type indices
            node_values: [max_nodes] node values
            
        Returns:
            Dict representation of generated AST
        """
        # Find valid nodes
        valid_indices = torch.where(node_exists)[0].tolist()
        
        if not valid_indices:
            return {'nodes': [], 'edges': []}
        
        # Extract valid node information
        nodes = []
        for idx in valid_indices:
            node_type = node_types[idx].item()
            node_value = node_values[idx].item()
            
            nodes.append({
                'id': idx,
                'type': node_type,
                'type_name': self._node_type_to_name(node_type),
                'value': node_value
            })
        
        # Extract edges
        edges = []
        adj_np = adjacency.cpu().numpy()
        for parent in valid_indices:
            for child in valid_indices:
                if adj_np[parent, child]:
                    edges.append((parent, child))
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(valid_indices)
        }
    
    def _node_type_to_name(self, node_type: int) -> str:
        """Convert node type index to human-readable name"""
        type_names = {
            NodeType.FUNC_DEF: 'function',
            NodeType.RETURN: 'return',
            NodeType.FOR_LOOP: 'for',
            NodeType.WHILE_LOOP: 'while', 
            NodeType.IF_STMT: 'if',
            NodeType.ELSE_STMT: 'else',
            NodeType.VAR_PARAM: 'param',
            NodeType.VAR_LOCAL: 'local_var',
            NodeType.VAR_ITER: 'iter_var',
            NodeType.VAR_TEMP: 'temp_var',
            NodeType.OP_ADD: '+',
            NodeType.OP_SUB: '-',
            NodeType.OP_MUL: '*',
            NodeType.OP_DIV: '/',
            NodeType.OP_MOD: '%',
            NodeType.OP_POW: '**',
            NodeType.OP_NEG: 'neg',
            NodeType.OP_ABS: 'abs',
            NodeType.OP_EQ: '==',
            NodeType.OP_NE: '!=',
            NodeType.OP_LT: '<',
            NodeType.OP_LE: '<=',
            NodeType.OP_GT: '>',
            NodeType.OP_GE: '>=',
            NodeType.OP_BUILTIN_SUM: 'sum',
            NodeType.OP_BUILTIN_RANGE: 'range',
            NodeType.OP_BUILTIN_LEN: 'len',
            NodeType.OP_BUILTIN_MIN: 'min',
            NodeType.CONST_INT: 'const_int',
            NodeType.CONST_BOOL: 'const_bool'
        }
        return type_names.get(node_type, f'unknown_{node_type}')


class ProgramSynthesisDataset:
    """Dataset wrapper for program synthesis training with HRM"""
    
    def __init__(self, 
                 data_path: str,
                 processor: ProgramSynthesisProcessor,
                 max_examples: int = 1000):
        self.data_path = data_path
        self.processor = processor
        self.max_examples = max_examples
        
        # Load data
        self.examples = self._load_examples()
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load program synthesis examples from data path"""
        import os
        import json
        
        examples = []
        
        # Look for JSON files in the data directory
        if os.path.isdir(self.data_path):
            for filename in os.listdir(self.data_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.data_path, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            examples.extend(data)
                        else:
                            examples.append(data)
        elif os.path.isfile(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]
        
        # Limit examples
        if len(examples) > self.max_examples:
            examples = examples[:self.max_examples]
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]
    
    def create_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Create batch from given indices"""
        batch_examples = [self.examples[i] for i in indices]
        return self.processor.create_hrm_batch(batch_examples)
    
    def get_random_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get random batch of examples"""
        indices = torch.randperm(len(self.examples))[:batch_size].tolist()
        return self.create_batch(indices)


def create_program_synthesis_loss(ast_targets: Dict[str, torch.Tensor], 
                                ast_predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Create multi-component loss for program synthesis training
    
    Args:
        ast_targets: Target AST tensors
        ast_predictions: Predicted AST tensors
        
    Returns:
        Combined loss tensor
    """
    losses = []
    
    # Node existence loss
    if 'node_exists' in ast_predictions:
        node_exist_loss = F.binary_cross_entropy_with_logits(
            ast_predictions['node_exists'], 
            ast_targets['node_exists'].float()
        )
        losses.append(node_exist_loss)
    
    # Node type classification loss
    if 'node_types' in ast_predictions:
        # Mask out non-existent nodes
        mask = ast_targets['node_exists']
        if mask.any():
            node_type_loss = F.cross_entropy(
                ast_predictions['node_types'][mask],
                ast_targets['node_types'][mask]
            )
            losses.append(node_type_loss)
    
    # Adjacency matrix loss
    if 'adjacency' in ast_predictions:
        adj_loss = F.binary_cross_entropy_with_logits(
            ast_predictions['adjacency'],
            ast_targets['adjacency'].float()
        )
        losses.append(adj_loss)
    
    # Node value regression loss (for constants)
    if 'node_values' in ast_predictions:
        mask = ast_targets['node_exists']
        if mask.any():
            value_loss = F.mse_loss(
                ast_predictions['node_values'][mask].float(),
                ast_targets['node_values'][mask].float()
            )
            losses.append(value_loss)
    
    # Combine losses
    if losses:
        return sum(losses) / len(losses)
    else:
        return torch.tensor(0.0, requires_grad=True)