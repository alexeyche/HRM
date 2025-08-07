"""
Program Synthesis HRM Model Adapter

This module extends the HRM model to output AST graph structures for program synthesis.
It adds AST prediction heads to the existing HRM architecture.
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1Carry
)
from models.layers import CastedLinear
from models.program_synthesis_processor import ProgramSynthesisProcessor
from dataset.build_program_synthesis_dataset import NodeType


class ProgramSynthesisHRMConfig(HierarchicalReasoningModel_ACTV1Config):
    """Extended HRM config for program synthesis"""

    # AST generation parameters
    max_nodes: int = 30
    max_edges: int = 25
    num_node_types: int = 30  # From NodeType enum

    # Program synthesis specific
    max_spec_tokens: int = 64
    max_examples: int = 10

    # Loss weights
    node_exist_weight: float = 1.0
    node_type_weight: float = 2.0
    adjacency_weight: float = 1.5
    node_value_weight: float = 1.0
    q_loss_weight: float = 1000.0


class ASTGenerationHead(nn.Module):
    """Neural network heads for generating AST components"""

    def __init__(self, config: ProgramSynthesisHRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Pooling for global representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Node existence prediction (per node)
        self.node_exist_head = CastedLinear(
            config.hidden_size, 1, bias=True
        )

        # Node type prediction (per node)
        self.node_type_head = CastedLinear(
            config.hidden_size, config.num_node_types, bias=True
        )

        # Node value prediction (per node, for constants)
        self.node_value_head = CastedLinear(
            config.hidden_size, 1, bias=True
        )

        # Adjacency prediction (pairwise)
        self.adjacency_head = nn.Sequential(
            CastedLinear(config.hidden_size * 2, config.hidden_size // 2, bias=True),
            nn.ReLU(),
            CastedLinear(config.hidden_size // 2, 1, bias=True)
        )

        # Global AST properties
        self.num_nodes_head = CastedLinear(
            config.hidden_size, config.max_nodes, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate AST predictions from hidden states

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] from HRM

        Returns:
            Dict of AST prediction tensors
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Global representation for sequence-level predictions
        global_repr = self.global_pool(hidden_states.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_size]

        # Predict number of nodes (categorical distribution)
        num_nodes_logits = self.num_nodes_head(global_repr)  # [batch_size, max_nodes]

        # Node-level predictions (use first max_nodes positions)
        node_positions = min(self.config.max_nodes, seq_len)
        node_hidden = hidden_states[:, :node_positions, :]  # [batch_size, max_nodes, hidden_size]

        # Node existence predictions
        node_exist_logits = self.node_exist_head(node_hidden).squeeze(-1)  # [batch_size, max_nodes]

        # Node type predictions
        node_type_logits = self.node_type_head(node_hidden)  # [batch_size, max_nodes, num_node_types]

        # Node value predictions (for constants)
        node_value_preds = self.node_value_head(node_hidden).squeeze(-1)  # [batch_size, max_nodes]

        # Adjacency predictions (pairwise combinations)
        adjacency_logits = torch.zeros(
            batch_size, self.config.max_nodes, self.config.max_nodes,
            dtype=self.forward_dtype, device=hidden_states.device
        )

        # Compute pairwise adjacency for valid node positions
        for i in range(node_positions):
            for j in range(node_positions):
                if i != j:  # No self-loops
                    # Concatenate parent and child representations
                    pair_repr = torch.cat([
                        node_hidden[:, i, :],  # Parent
                        node_hidden[:, j, :]   # Child
                    ], dim=-1)

                    adjacency_logits[:, i, j] = self.adjacency_head(pair_repr).squeeze(-1)

        return {
            'num_nodes_logits': num_nodes_logits,
            'node_exists': node_exist_logits,
            'node_types': node_type_logits,
            'node_values': node_value_preds,
            'adjacency': adjacency_logits
        }


class ProgramSynthesisHRM(nn.Module):
    """HRM model adapted for program synthesis tasks"""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ProgramSynthesisHRMConfig(**config_dict)

        # Base HRM model
        self.hrm = HierarchicalReasoningModel_ACTV1(config_dict)

        # AST generation head
        self.ast_head = ASTGenerationHead(self.config)

        # Program synthesis processor
        self.processor = ProgramSynthesisProcessor(
            max_examples=self.config.max_examples,
            max_spec_tokens=self.config.max_spec_tokens,
            max_nodes=self.config.max_nodes,
            max_edges=self.config.max_edges,
            vocab_size=self.config.vocab_size
        )

    @property
    def puzzle_emb(self):
        """Expose puzzle embeddings from base HRM"""
        return self.hrm.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> HierarchicalReasoningModel_ACTV1Carry:
        """Initialize carry state for HRM"""
        # Extract only tensor values for HRM (exclude nested dicts like ast_targets)
        hrm_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        return self.hrm.initial_carry(hrm_batch)

    def forward(self,
                carry: HierarchicalReasoningModel_ACTV1Carry,
                batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        """
        Forward pass for program synthesis

        Args:
            carry: HRM carry state
            batch: Input batch with specifications

        Returns:
            Updated carry and outputs including AST predictions
        """
        # Forward through base HRM (extract only tensor values)
        hrm_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        new_carry, hrm_outputs = self.hrm(carry, hrm_batch)

        # Extract hidden states from the inner model
        # We need to access the hidden states before the language modeling head
        with torch.no_grad():
            # Get hidden states from the HRM inner model
            seq_info = dict(
                cos_sin=self.hrm.inner.rotary_emb() if hasattr(self.hrm.inner, "rotary_emb") else None,
            )

            # Input encoding
            input_embeddings = self.hrm.inner._input_embeddings(
                carry.current_data["inputs"],
                carry.current_data["puzzle_identifiers"]
            )

            # Get final hidden states (after reasoning)
            z_H = new_carry.inner_carry.z_H
            z_L = new_carry.inner_carry.z_L

            # Use L-level states for AST generation (more detailed)
            hidden_states = z_L[:, self.hrm.inner.puzzle_emb_len:]  # Remove puzzle embedding positions

        # Generate AST predictions
        ast_predictions = self.ast_head(hidden_states)

        # Combine outputs
        outputs = {
            **hrm_outputs,  # Original HRM outputs (Q-values, etc.)
            **ast_predictions  # AST predictions
        }

        return new_carry, outputs

    def compute_loss(self,
                    outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component loss for program synthesis

        Args:
            outputs: Model predictions
            targets: Ground truth targets

        Returns:
            Dict of individual and total losses
        """
        losses = {}

        # Extract AST targets
        ast_targets = targets.get('ast_targets', {})

        # Node existence loss
        if 'node_exists' in outputs and 'node_exists' in ast_targets:
            node_exist_loss = F.binary_cross_entropy_with_logits(
                outputs['node_exists'],
                ast_targets['node_exists'].float()
            )
            losses['node_exist_loss'] = self.config.node_exist_weight * node_exist_loss

        # Node type classification loss
        if 'node_types' in outputs and 'node_types' in ast_targets:
            # Mask out non-existent nodes
            mask = ast_targets['node_exists']  # [batch_size, max_nodes]

            if mask.any():
                # Flatten for cross entropy
                masked_predictions = outputs['node_types'][mask]  # [num_valid_nodes, num_node_types]
                masked_targets = ast_targets['node_types'][mask]  # [num_valid_nodes]

                node_type_loss = F.cross_entropy(masked_predictions, masked_targets)
                losses['node_type_loss'] = self.config.node_type_weight * node_type_loss

        # Adjacency matrix loss
        if 'adjacency' in outputs and 'adjacency' in ast_targets:
            # Only compute loss for existing nodes
            node_mask = ast_targets['node_exists']  # [batch_size, max_nodes]

            # Create adjacency mask for valid node pairs
            adj_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)  # [batch_size, max_nodes, max_nodes]

            if adj_mask.any():
                adj_loss = F.binary_cross_entropy_with_logits(
                    outputs['adjacency'][adj_mask],
                    ast_targets['adjacency'][adj_mask].float()
                )
                losses['adjacency_loss'] = self.config.adjacency_weight * adj_loss

        # Node value regression loss (for constants)
        if 'node_values' in outputs and 'node_values' in ast_targets:
            mask = ast_targets['node_exists']
            if mask.any():
                value_loss = F.mse_loss(
                    outputs['node_values'][mask],
                    ast_targets['node_values'][mask].float()
                )
                losses['node_value_loss'] = self.config.node_value_weight * value_loss

        # Number of nodes prediction loss
        if 'num_nodes_logits' in outputs and 'num_nodes' in ast_targets:
            # Clamp targets to valid range [0, max_nodes-1]
            valid_targets = torch.clamp(ast_targets['num_nodes'], 0, self.config.max_nodes - 1)
            num_nodes_loss = F.cross_entropy(
                outputs['num_nodes_logits'],
                valid_targets
            )
            losses['num_nodes_loss'] = num_nodes_loss

        # Q-learning losses from base HRM (if available)
        if 'q_halt_logits' in outputs and 'target_q_continue' in outputs:
            q_loss = F.mse_loss(
                torch.sigmoid(outputs['q_continue_logits']),
                outputs['target_q_continue']
            )
            losses['q_loss'] = self.config.q_loss_weight * q_loss

        # Total loss
        if losses:
            losses['total_loss'] = sum(losses.values())
        else:
            losses['total_loss'] = torch.tensor(0.0, requires_grad=True)

        return losses

    def generate_program(self, specification: Dict[str, any]) -> Dict[str, any]:
        """
        Generate program from specification (inference mode)

        Args:
            specification: Program specification dict

        Returns:
            Generated AST representation
        """
        self.eval()

        with torch.no_grad():
            # Create batch with single example (use dummy implementation for generation)
            example = {'specification': specification, 'implementation': 'def program(): return 0'}  # Dummy implementation
            batch = self.processor.create_hrm_batch([example])

            # Move to device
            device = next(self.parameters()).device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(device)

            # Initialize carry
            carry = self.initial_carry(batch)

            # Forward pass
            final_carry, outputs = self(carry, batch)

            # Convert logits to predictions
            predictions = {
                'node_exists': torch.sigmoid(outputs['node_exists']) > 0.5,
                'adjacency': torch.sigmoid(outputs['adjacency']) > 0.5,
                'node_types': outputs['node_types'].argmax(dim=-1),
                'node_values': outputs['node_values'].round().long(),
                'num_nodes': outputs['num_nodes_logits'].argmax(dim=-1)
            }

            # Decode to interpretable format
            batch_size = predictions['node_exists'].shape[0]
            results = []

            for i in range(batch_size):
                ast_dict = self.processor.decode_ast_output(
                    predictions['node_exists'][i],
                    predictions['adjacency'][i],
                    predictions['node_types'][i],
                    predictions['node_values'][i]
                )
                results.append(ast_dict)

            return results[0] if len(results) == 1 else results


def create_program_synthesis_model(config_dict: dict) -> ProgramSynthesisHRM:
    """Factory function to create program synthesis HRM model"""
    return ProgramSynthesisHRM(config_dict)