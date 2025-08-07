#!/usr/bin/env python3
"""
Debug Q-learning gradients and loss magnitude
"""

import torch
from models.program_synthesis_hrm import ProgramSynthesisHRM
from program_synthesis_dataset import ProgramSynthesisDataset, ProgramSynthesisDatasetConfig


def debug_q_gradients():
    """Debug Q-learning gradients and optimization"""
    print("Debugging Q-learning gradients...")
    
    # Create config
    config = {
        'batch_size': 1,
        'seq_len': 64,
        'puzzle_emb_ndim': 0,
        'num_puzzle_identifiers': 10,
        'vocab_size': 1024,
        'H_cycles': 1,
        'L_cycles': 1,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 4,
        'halt_exploration_prob': 0.3,  # Higher exploration
        'max_nodes': 20,
        'max_edges': 15,
        'num_node_types': 30,
        'max_spec_tokens': 64,
        'max_examples': 5,
        'forward_dtype': 'float32'
    }
    
    # Create model
    hrm_model = ProgramSynthesisHRM(config)
    hrm_model.train()
    
    # Get a training example
    dataset_config = ProgramSynthesisDatasetConfig(
        seed=42,
        dataset_path="data/program-synthesis-100",
        rank=0,
        num_replicas=1
    )
    
    train_dataset = ProgramSynthesisDataset(dataset_config, "train")
    
    for set_name, batch, global_batch_size in train_dataset:
        break
    
    print(f"✓ Model and data ready")
    
    # Check Q-head parameters before training
    q_head = hrm_model.hrm.inner.q_head
    print(f"\n=== Q-head parameters BEFORE ===")
    print(f"Q-head weight: {q_head.weight.data}")
    print(f"Q-head bias: {q_head.bias.data}")
    print(f"Q-head weight requires_grad: {q_head.weight.requires_grad}")
    print(f"Q-head bias requires_grad: {q_head.bias.requires_grad}")
    
    # Simple optimizer for Q-head only
    q_optimizer = torch.optim.Adam(q_head.parameters(), lr=0.01)
    
    # Forward pass with gradients
    carry = hrm_model.initial_carry(batch)
    
    total_q_loss = 0
    total_other_loss = 0
    
    for step in range(4):
        # Forward
        carry, outputs = hrm_model(carry, batch)
        
        # Compute losses
        targets = {'ast_targets': batch.get('ast_targets', {})}
        losses = hrm_model.compute_loss(outputs, targets)
        
        q_loss = losses.get('q_loss', torch.tensor(0.0))
        other_losses = sum(v for k, v in losses.items() if k != 'q_loss' and k != 'total_loss')
        
        total_q_loss += q_loss.item()
        total_other_loss += other_losses.item()
        
        print(f"\nStep {step + 1}:")
        print(f"  q_halt_logits: {outputs['q_halt_logits'].item():.6f}")
        print(f"  q_continue_logits: {outputs['q_continue_logits'].item():.6f}")
        print(f"  target_q_continue: {outputs['target_q_continue'].item():.6f}")
        print(f"  q_loss: {q_loss.item():.8f}")
        print(f"  other_losses: {other_losses.item():.6f}")
        
        # Backward on Q-loss only
        if q_loss.item() > 0:
            q_optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            
            # Check if gradients exist
            if q_head.weight.grad is not None:
                grad_norm = q_head.weight.grad.norm().item()
                print(f"  Q-head weight grad norm: {grad_norm:.8f}")
            else:
                print(f"  Q-head weight grad: None")
                
            if q_head.bias.grad is not None:
                grad_norm = q_head.bias.grad.norm().item()
                print(f"  Q-head bias grad norm: {grad_norm:.8f}")
            else:
                print(f"  Q-head bias grad: None")
            
            # Update
            q_optimizer.step()
            
        if carry.halted.all():
            break
    
    print(f"\n=== Q-head parameters AFTER ===")
    print(f"Q-head weight: {q_head.weight.data}")
    print(f"Q-head bias: {q_head.bias.data}")
    
    print(f"\n=== Loss Summary ===")
    print(f"Total Q-loss: {total_q_loss:.8f}")
    print(f"Total other losses: {total_other_loss:.6f}")
    print(f"Q-loss ratio: {total_q_loss / (total_other_loss + 1e-8):.8f}")


if __name__ == "__main__":
    debug_q_gradients()