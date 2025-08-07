#!/usr/bin/env python3
"""
Standalone Program Synthesis Model Test

This demonstrates the core program synthesis functionality without FlashAttention dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import json
import math

# Import our data processing components
from models.program_synthesis_processor import ProgramSynthesisProcessor
from dataset.build_program_synthesis_dataset import PROGRAM_TEMPLATES


class StandaloneProgramSynthesisModel(nn.Module):
    """Standalone program synthesis model for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Core parameters
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.max_nodes = config['max_nodes']
        self.num_node_types = config['num_node_types']
        self.seq_len = config['seq_len']
        
        # Input processing
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_emb = nn.Embedding(self.seq_len, self.hidden_size)
        
        # Transformer layers (simplified)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config['num_heads'],
                dim_feedforward=int(self.hidden_size * config['expansion']),
                dropout=0.1,
                batch_first=True
            )
            for _ in range(config['num_layers'])
        ])
        
        # AST Generation Heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Node-level predictions
        self.node_exist_head = nn.Linear(self.hidden_size, 1)
        self.node_type_head = nn.Linear(self.hidden_size, self.num_node_types)
        self.node_value_head = nn.Linear(self.hidden_size, 1)
        
        # Global predictions
        self.num_nodes_head = nn.Linear(self.hidden_size, self.max_nodes)
        
        # Adjacency prediction (simplified - use global representation)
        self.adjacency_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.max_nodes * self.max_nodes)
        )
    
    def forward(self, inputs: torch.Tensor, targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            inputs: [batch_size, seq_len] token indices
            targets: Optional targets for loss computation
            
        Returns:
            Dict of predictions and optionally loss
        """
        batch_size, seq_len = inputs.shape
        
        # Input embeddings
        token_emb = self.embed_tokens(inputs)  # [batch_size, seq_len, hidden_size]
        pos_ids = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(pos_ids)
        
        x = token_emb + pos_emb
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)  # [batch_size, seq_len, hidden_size]
        
        # Global representation for sequence-level predictions
        global_repr = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_size]
        
        # Number of nodes prediction
        num_nodes_logits = self.num_nodes_head(global_repr)  # [batch_size, max_nodes]
        
        # Node-level predictions (use first max_nodes positions)
        node_positions = min(self.max_nodes, seq_len)
        node_hidden = x[:, :node_positions, :]  # [batch_size, max_nodes, hidden_size]
        
        # Pad if needed
        if node_positions < self.max_nodes:
            padding = torch.zeros(batch_size, self.max_nodes - node_positions, self.hidden_size, 
                                device=x.device, dtype=x.dtype)
            node_hidden = torch.cat([node_hidden, padding], dim=1)
        
        # Node predictions
        node_exist_logits = self.node_exist_head(node_hidden).squeeze(-1)  # [batch_size, max_nodes]
        node_type_logits = self.node_type_head(node_hidden)  # [batch_size, max_nodes, num_node_types]
        node_value_preds = self.node_value_head(node_hidden).squeeze(-1)  # [batch_size, max_nodes]
        
        # Adjacency predictions (simplified - use global representation)
        adj_flat = self.adjacency_head(global_repr)  # [batch_size, max_nodes * max_nodes]
        adjacency_logits = adj_flat.view(batch_size, self.max_nodes, self.max_nodes)
        
        outputs = {
            'num_nodes_logits': num_nodes_logits,
            'node_exists': node_exist_logits,
            'node_types': node_type_logits,
            'node_values': node_value_preds,
            'adjacency': adjacency_logits,
            'hidden_states': x
        }
        
        # Compute loss if targets provided
        if targets is not None:
            losses = self.compute_loss(outputs, targets)
            outputs.update(losses)
        
        return outputs
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-component loss"""
        losses = {}
        
        # Node existence loss
        if 'node_exists' in targets:
            node_exist_loss = F.binary_cross_entropy_with_logits(
                predictions['node_exists'], 
                targets['node_exists'].float()
            )
            losses['node_exist_loss'] = node_exist_loss
        
        # Node type loss (only for existing nodes)
        if 'node_types' in targets and 'node_exists' in targets:
            mask = targets['node_exists']
            if mask.any():
                masked_pred = predictions['node_types'][mask]
                masked_target = targets['node_types'][mask]
                node_type_loss = F.cross_entropy(masked_pred, masked_target)
                losses['node_type_loss'] = node_type_loss
        
        # Adjacency loss
        if 'adjacency' in targets:
            adj_loss = F.binary_cross_entropy_with_logits(
                predictions['adjacency'],
                targets['adjacency'].float()
            )
            losses['adjacency_loss'] = adj_loss
        
        # Node value loss (for existing nodes)
        if 'node_values' in targets and 'node_exists' in targets:
            mask = targets['node_exists']
            if mask.any():
                value_loss = F.mse_loss(
                    predictions['node_values'][mask],
                    targets['node_values'][mask].float()
                )
                losses['node_value_loss'] = value_loss
        
        # Number of nodes loss
        if 'num_nodes' in targets:
            num_nodes_loss = F.cross_entropy(
                predictions['num_nodes_logits'],
                targets['num_nodes']
            )
            losses['num_nodes_loss'] = num_nodes_loss
        
        # Total loss
        if losses:
            losses['total_loss'] = sum(losses.values())
        
        return losses
    
    def generate_program(self, specification: Dict[str, Any], processor: ProgramSynthesisProcessor) -> Dict[str, Any]:
        """Generate program from specification"""
        self.eval()
        
        with torch.no_grad():
            # Encode specification
            encoded_spec = processor.encode_specification(specification)
            inputs = encoded_spec.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            outputs = self(inputs)
            
            # Convert to predictions
            predictions = {
                'node_exists': torch.sigmoid(outputs['node_exists']) > 0.5,
                'adjacency': torch.sigmoid(outputs['adjacency']) > 0.5,
                'node_types': outputs['node_types'].argmax(dim=-1),
                'node_values': outputs['node_values'].round().long(),
                'num_nodes': outputs['num_nodes_logits'].argmax(dim=-1)
            }
            
            # Decode AST
            result = processor.decode_ast_output(
                predictions['node_exists'][0],
                predictions['adjacency'][0],
                predictions['node_types'][0],
                predictions['node_values'][0]
            )
            
            return result


def test_standalone_model():
    """Test the standalone model"""
    print("Testing Standalone Program Synthesis Model...\n")
    
    # Model configuration
    config = {
        'vocab_size': 512,
        'hidden_size': 128,
        'max_nodes': 20,
        'num_node_types': 30,
        'seq_len': 64,
        'num_heads': 8,
        'num_layers': 2,
        'expansion': 2.0
    }
    
    # Create model and processor
    model = StandaloneProgramSynthesisModel(config)
    processor = ProgramSynthesisProcessor(
        max_examples=5,
        max_spec_tokens=config['seq_len'],
        max_nodes=config['max_nodes'],
        vocab_size=config['vocab_size']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test data
    test_examples = []
    simple_templates = ['double', 'is_even', 'absolute_value']
    
    for name in simple_templates:
        if name in PROGRAM_TEMPLATES:
            template = PROGRAM_TEMPLATES[name]
            example = {
                'specification': {
                    'name': name,
                    'description': template['description'],
                    'inputs': template['inputs'],
                    'outputs': template['outputs'],
                    'examples': template['base_examples'][:3]
                },
                'implementation': template['implementation']
            }
            test_examples.append(example)
    
    if not test_examples:
        # Fallback examples
        test_examples = [{
            'specification': {
                'name': 'add_one',
                'description': 'Add 1 to input',
                'inputs': [{'type': 'int'}],
                'outputs': [{'type': 'int'}],
                'examples': [{'input': 5, 'output': 6}]
            },
            'implementation': 'def program(n): return n + 1'
        }]
    
    # Create batch
    batch = processor.create_hrm_batch(test_examples[:2])
    
    print(f"Created batch with {len(test_examples[:2])} examples")
    print(f"Input shape: {batch['inputs'].shape}")
    
    # Forward pass
    model.train()
    outputs = model(batch['inputs'], batch['ast_targets'])
    
    print(f"\nForward pass results:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            if key.endswith('_loss') or key == 'total_loss':
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: shape {value.shape}")
    
    # Test generation
    print(f"\nTesting program generation...")
    test_spec = {
        'name': 'simple_double',
        'description': 'Double the input number',
        'inputs': [{'type': 'int', 'description': 'Input number'}],
        'outputs': [{'type': 'int', 'description': 'Doubled number'}],
        'examples': [
            {'input': 3, 'output': 6},
            {'input': 5, 'output': 10}
        ]
    }
    
    generated_ast = model.generate_program(test_spec, processor)
    
    print(f"Generated AST:")
    print(f"  Nodes: {generated_ast['num_nodes']}")
    print(f"  Edges: {len(generated_ast['edges'])}")
    
    if generated_ast['nodes']:
        print(f"  Sample nodes:")
        for i, node in enumerate(generated_ast['nodes'][:5]):
            print(f"    Node {i}: {node['type_name']} (value: {node['value']})")
    
    # Test a simple training step
    print(f"\nTesting training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    optimizer.zero_grad()
    
    outputs = model(batch['inputs'], batch['ast_targets'])
    loss = outputs['total_loss']
    
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed, loss: {loss.item():.4f}")
    
    return True


def main():
    """Run standalone model test"""
    try:
        success = test_standalone_model()
        
        if success:
            print(f"\n🎉 Standalone Program Synthesis Model Test Passed!")
            print(f"\nKey achievements:")
            print(f"✅ Model can process program specifications")
            print(f"✅ Model can generate AST predictions")
            print(f"✅ Multi-component loss computation works")
            print(f"✅ Training step executes successfully")
            print(f"✅ Program generation pipeline works")
            print(f"\nThe model is ready for training on program synthesis tasks!")
            
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())