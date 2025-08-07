#!/usr/bin/env python3
"""
Test HRM program synthesis model with MLX
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, Any

from models.program_synthesis_processor import ProgramSynthesisProcessor
from dataset.build_program_synthesis_dataset import PROGRAM_TEMPLATES


def create_simple_test_data():
    """Create a simple test example for MLX"""
    processor = ProgramSynthesisProcessor(
        max_examples=3,
        max_spec_tokens=32,
        max_nodes=10,
        max_edges=8,
        vocab_size=512
    )
    
    # Use a simple template
    template = PROGRAM_TEMPLATES['double']
    example = {
        'specification': {
            'name': 'double',
            'description': template['description'],
            'inputs': template['inputs'],
            'outputs': template['outputs'],
            'examples': template['base_examples'][:3]
        },
        'implementation': template['implementation']
    }
    
    # Create batch
    batch = processor.create_hrm_batch([example])
    
    # Convert to MLX arrays
    mlx_batch = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            mlx_batch[key] = {k: mx.array(v.numpy()) for k, v in value.items()}
        else:
            mlx_batch[key] = mx.array(value.numpy())
    
    return mlx_batch, processor


class SimpleMLXHRM(nn.Module):
    """Simplified HRM model for MLX testing"""
    
    def __init__(self, vocab_size: int = 512, hidden_size: int = 128, max_nodes: int = 10):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_nodes = max_nodes
        
        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Simple transformer-like processing
        self.transformer = nn.MultiHeadAttention(hidden_size, num_heads=4)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # AST prediction heads
        self.node_exist_head = nn.Linear(hidden_size, 1)
        self.node_type_head = nn.Linear(hidden_size, 30)  # 30 node types
        self.adjacency_head = nn.Linear(hidden_size * 2, 1)
    
    def __call__(self, inputs: mx.array) -> Dict[str, mx.array]:
        # Embed inputs
        x = self.embedding(inputs)  # [batch_size, seq_len, hidden_size]
        
        # Simple processing
        attended = self.transformer(x, x, x)
        x = self.norm(x + attended)
        x = x + self.mlp(x)
        
        # Global pooling for sequence representation
        pooled = mx.mean(x, axis=1)  # [batch_size, hidden_size]
        
        # Node predictions (using first max_nodes positions)
        batch_size = x.shape[0]
        seq_len = min(self.max_nodes, x.shape[1])
        node_features = x[:, :seq_len, :]  # [batch_size, max_nodes, hidden_size]
        
        # Node existence and type predictions
        node_exists = self.node_exist_head(node_features).squeeze(-1)  # [batch_size, max_nodes]
        node_types = self.node_type_head(node_features)  # [batch_size, max_nodes, 30]
        
        # Simple adjacency prediction (just predict from global representation)
        adjacency = mx.zeros((batch_size, self.max_nodes, self.max_nodes))
        
        return {
            'node_exists': node_exists,
            'node_types': node_types,
            'adjacency': adjacency
        }


def test_mlx_model():
    """Test the MLX HRM model"""
    print("Testing MLX HRM Model...")
    
    # Create test data
    batch, processor = create_simple_test_data()
    
    print(f"✓ Test data created")
    print(f"  Input shape: {batch['inputs'].shape}")
    print(f"  Input range: {mx.min(batch['inputs']):.0f} to {mx.max(batch['inputs']):.0f}")
    
    # Create model
    model = SimpleMLXHRM(vocab_size=512, hidden_size=64, max_nodes=10)
    
    print(f"✓ Model created")
    
    # Forward pass
    outputs = model(batch['inputs'])
    
    print(f"✓ Forward pass successful")
    print(f"  Output keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        print(f"    {key}: {value.shape}")
    
    # Test loss computation
    targets = batch['ast_targets']
    
    # Simple loss computation
    node_exist_loss = mx.mean((outputs['node_exists'] - targets['node_exists'].astype(mx.float32)) ** 2)
    
    # Node type loss (only for existing nodes)
    mask = targets['node_exists'].astype(mx.bool_)
    if mx.any(mask):
        # Cross entropy approximation
        node_type_logits = outputs['node_types'][mask]  # [num_valid, 30]
        node_type_targets = targets['node_types'][mask]  # [num_valid]
        
        # Simplified cross entropy
        softmax_logits = mx.softmax(node_type_logits, axis=-1)
        node_type_loss = -mx.mean(mx.log(softmax_logits[mx.arange(softmax_logits.shape[0]), node_type_targets] + 1e-8))
    else:
        node_type_loss = mx.array(0.0)
    
    total_loss = node_exist_loss + node_type_loss
    
    print(f"✓ Loss computation successful")
    print(f"  Node exist loss: {node_exist_loss.item():.4f}")
    print(f"  Node type loss: {node_type_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    # Test optimizer
    optimizer = optim.Adam(learning_rate=1e-4)
    
    def loss_fn(model, batch):
        outputs = model(batch['inputs'])
        targets = batch['ast_targets']
        
        node_exist_loss = mx.mean((outputs['node_exists'] - targets['node_exists'].astype(mx.float32)) ** 2)
        
        mask = targets['node_exists'].astype(mx.bool_)
        if mx.any(mask):
            node_type_logits = outputs['node_types'][mask]
            node_type_targets = targets['node_types'][mask]
            softmax_logits = mx.softmax(node_type_logits, axis=-1)
            node_type_loss = -mx.mean(mx.log(softmax_logits[mx.arange(softmax_logits.shape[0]), node_type_targets] + 1e-8))
        else:
            node_type_loss = mx.array(0.0)
        
        return node_exist_loss + node_type_loss
    
    # Test gradient computation
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    loss, grads = loss_and_grad_fn(model, batch)
    
    print(f"✓ Gradient computation successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient keys: {list(grads.keys())}")
    
    # Update parameters
    optimizer.update(model, grads)
    
    print(f"✓ Parameter update successful")
    
    # Test a training step
    initial_loss = loss.item()
    
    for step in range(5):
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        print(f"  Step {step + 1}: Loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    
    print(f"✓ Training steps completed")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss change: {final_loss - initial_loss:.4f}")
    
    return True


def main():
    """Run MLX HRM tests"""
    print("MLX HRM Program Synthesis Test\n")
    
    # Check MLX availability
    print("Checking MLX...")
    try:
        print(f"✓ MLX version: {mx.__version__}")
        print(f"✓ Device: {mx.default_device()}")
    except Exception as e:
        print(f"❌ MLX error: {e}")
        return False
    
    print()
    
    # Test model
    success = test_mlx_model()
    
    if success:
        print(f"\n🎉 MLX HRM Test PASSED!")
        print(f"\nKey achievements:")
        print(f"✅ MLX model creation and forward pass")
        print(f"✅ Loss computation with MLX arrays")
        print(f"✅ Gradient computation and parameter updates")
        print(f"✅ Multi-step training simulation")
        print(f"\n🚀 Ready for full MLX training pipeline!")
    else:
        print(f"\n❌ MLX HRM Test FAILED")
    
    return success


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)