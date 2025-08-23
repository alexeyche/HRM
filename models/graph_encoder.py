"""
Graph Neural Network Encoder for Program Synthesis

GCN-based encoder that maps AST graphs to latent embeddings compatible 
with the GrammarAwareGenerationHead.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from dataset.ast_types import ASTNodeType, OperatorType, EdgeType
from models.common import trunc_normal_init_
from models.layers import CastedLinear


class NodeEmbedding(nn.Module):
    """
    Node embedding layer that combines different types of node features.
    
    Follows encoding strategy from ENCODING.md:
    - Node type embeddings
    - Variable/operator embeddings  
    - Literal value encoders
    - Structural features
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_node_types: int = 100,  # Max enum values
        num_operators: int = 50,
        num_variables: int = 26,    # a-z variables
        max_int_value: int = 100,
        max_str_length: int = 20
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Core embeddings
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.operator_embedding = nn.Embedding(num_operators, hidden_dim // 4)
        self.variable_embedding = nn.Embedding(num_variables, hidden_dim // 4)
        
        # Value encoders
        self.value_type_embedding = nn.Embedding(10, hidden_dim // 8)  # int, str, bool, etc.
        
        # Integer value encoder (bucketed + continuous)
        self.int_bucket_embedding = nn.Embedding(11, hidden_dim // 8)  # -5 to 5
        self.small_constant_embedding = nn.Embedding(20, hidden_dim // 8)
        self.int_continuous_proj = CastedLinear(4, hidden_dim // 8, bias=True)  # sign, zero, parity, scaled
        
        # String value encoder
        self.str_length_bucket_embedding = nn.Embedding(6, hidden_dim // 8)  # 0 to 5
        self.str_features_proj = CastedLinear(5, hidden_dim // 8, bias=True)  # char features
        
        # Boolean value encoder
        self.bool_embedding = nn.Embedding(3, hidden_dim // 8)  # True, False, N/A
        
        # Structural features
        self.depth_embedding = nn.Embedding(20, hidden_dim // 8)
        self.semantic_flag_embedding = nn.Embedding(2, hidden_dim // 8)
        
        # Final projection to combine all features
        feature_dim = (
            hidden_dim +           # node_type
            hidden_dim // 4 +      # operator
            hidden_dim // 4 +      # variable
            hidden_dim // 8 +      # value_type
            hidden_dim // 8 +      # int_bucket
            hidden_dim // 8 +      # small_constant
            hidden_dim // 8 +      # int_continuous
            hidden_dim // 8 +      # str_length_bucket
            hidden_dim // 8 +      # str_features
            hidden_dim // 8 +      # bool
            hidden_dim // 8 +      # depth
            hidden_dim // 8        # semantic_flag
        )
        
        self.output_proj = CastedLinear(feature_dim, hidden_dim, bias=True)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding layers."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                trunc_normal_init_(module.weight, std=0.02)
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Embed node features into dense vectors.
        
        Args:
            node_features: Raw node features [num_nodes, feature_dim]
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        batch_size = node_features.shape[0]
        device = node_features.device
        
        if batch_size == 0:
            return torch.empty(0, self.hidden_dim, device=device)
        
        # Extract features (matching _encode_node_features in ast_converter.py)
        node_type = node_features[:, 0].long().clamp(0, 99)           # Node type
        depth = node_features[:, 1].long().clamp(0, 19)              # Depth
        is_semantic = node_features[:, 2].long().clamp(0, 1)         # Is semantic
        variable_id = node_features[:, 3].long().clamp(-1, 25)       # Variable ID
        int_value = node_features[:, 4]                              # Int value
        value_type = node_features[:, 5].long().clamp(0, 9)          # Value type
        
        embeddings = []
        
        # Core embeddings
        embeddings.append(self.node_type_embedding(node_type))
        
        # Operator embedding (default to 0 if not present)
        operator_ids = torch.zeros_like(node_type)
        embeddings.append(self.operator_embedding(operator_ids))
        
        # Variable embedding
        valid_var_mask = variable_id >= 0
        var_ids_clamped = variable_id.clamp(0, 25)
        var_emb = self.variable_embedding(var_ids_clamped)
        var_emb = var_emb * valid_var_mask.unsqueeze(-1).float()
        embeddings.append(var_emb)
        
        # Value type embedding
        embeddings.append(self.value_type_embedding(value_type))
        
        # Integer value encoding
        int_mask = (value_type == 1).float()  # value_type 1 = int
        
        # Bucket encoding
        int_buckets = torch.zeros_like(node_type)
        valid_int_mask = int_mask.bool()
        if valid_int_mask.any():
            int_vals = int_value[valid_int_mask]
            buckets = torch.zeros_like(int_vals).long()
            nonzero_mask = int_vals != 0
            if nonzero_mask.any():
                log_vals = torch.log10(torch.abs(int_vals[nonzero_mask]) + 1e-8)
                buckets[nonzero_mask] = torch.floor(log_vals).long().clamp(-5, 5) + 5
            int_buckets[valid_int_mask] = buckets
        
        embeddings.append(self.int_bucket_embedding(int_buckets))
        
        # Small constant embedding
        small_constants = [-1, 0, 1, 2, 3, 4, 5, 10, 100]
        small_const_ids = torch.zeros_like(node_type)
        for i, const in enumerate(small_constants):
            mask = (int_value == const) & int_mask.bool()
            small_const_ids = torch.where(mask, torch.tensor(i, device=device), small_const_ids)
        
        embeddings.append(self.small_constant_embedding(small_const_ids))
        
        # Continuous int features
        int_continuous = torch.zeros(batch_size, 4, device=device)
        if valid_int_mask.any():
            int_vals = int_value[valid_int_mask]
            int_continuous[valid_int_mask, 0] = torch.sign(int_vals)  # sign
            int_continuous[valid_int_mask, 1] = (int_vals == 0).float()  # zero
            int_continuous[valid_int_mask, 2] = (int_vals % 2).float()  # parity
            int_continuous[valid_int_mask, 3] = torch.tanh(int_vals / 10.0)  # scaled
        
        embeddings.append(self.int_continuous_proj(int_continuous))
        
        # String features (placeholder - would need actual string data)
        str_length_buckets = torch.zeros_like(node_type)
        str_features = torch.zeros(batch_size, 5, device=device)
        embeddings.append(self.str_length_bucket_embedding(str_length_buckets))
        embeddings.append(self.str_features_proj(str_features))
        
        # Boolean features
        bool_ids = torch.zeros_like(node_type)
        bool_mask = (value_type == 3).bool()  # value_type 3 = bool
        # Would extract actual boolean values from node features in full implementation
        embeddings.append(self.bool_embedding(bool_ids))
        
        # Structural features
        embeddings.append(self.depth_embedding(depth))
        embeddings.append(self.semantic_flag_embedding(is_semantic))
        
        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        # Project to final dimension
        output = self.output_proj(combined)
        
        return output


class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder using GCN layers.
    
    Architecture:
    - Node embedding layer
    - Multiple GCN layers
    - Global pooling to create graph-level representation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",  # "mean", "max", "add", "attention"
        node_embedding_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # Node embedding layer
        node_emb_kwargs = node_embedding_kwargs or {}
        self.node_embedding = NodeEmbedding(hidden_dim, **node_emb_kwargs)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Global pooling
        if pooling == "mean":
            self.global_pool = global_mean_pool
        elif pooling == "max":
            self.global_pool = global_max_pool
        elif pooling == "add":
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Encode a batch of graphs into latent representations.
        
        Args:
            data: PyTorch Geometric batch of graphs
            
        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if x.shape[0] == 0:
            # Empty batch
            batch_size = torch.max(batch) + 1 if batch.numel() > 0 else 1
            return torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Node embeddings
        x = self.node_embedding(x)
        
        # GCN layers with residual connections
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.layer_norms)):
            residual = x
            
            # GCN layer
            x = gcn(x, edge_index)
            
            # Layer normalization
            x = norm(x)
            
            # Activation
            x = F.gelu(x)
            
            # Dropout
            x = self.dropout_layer(x)
            
            # Residual connection
            x = x + residual
        
        # Global pooling
        graph_embeddings = self.global_pool(x, batch)
        
        return graph_embeddings


class ProgramGraphEncoder(nn.Module):
    """
    Complete encoder for program graphs that produces embeddings
    compatible with GrammarAwareGenerationHead.
    
    Includes additional processing for program-specific features.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        add_program_context: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.add_program_context = add_program_context
        
        # Core graph encoder
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            **kwargs
        )
        
        if add_program_context:
            # Additional context processing
            self.context_proj = CastedLinear(hidden_dim + 16, hidden_dim, bias=True)
    
    def forward(self, data: Batch, program_infos: Optional[List[Dict[str, Any]]] = None) -> torch.Tensor:
        """
        Encode program graphs with optional program context.
        
        Args:
            data: PyTorch Geometric batch of graphs
            program_infos: Optional program information for context
            
        Returns:
            Program embeddings [batch_size, hidden_dim]
        """
        # Get base graph embeddings
        graph_embeddings = self.graph_encoder(data)
        
        if not self.add_program_context or program_infos is None:
            return graph_embeddings
        
        # Add program context features
        batch_size = graph_embeddings.shape[0]
        context_features = torch.zeros(batch_size, 16, device=graph_embeddings.device)
        
        for i, info in enumerate(program_infos):
            if i >= batch_size:
                break
            
            # Add simple context features
            # Input/output type information
            input_types = info.get('input_types', [])
            output_types = info.get('output_types', [])
            
            context_features[i, 0] = len(input_types)
            context_features[i, 1] = len(output_types)
            
            # Type encoding (simplified)
            type_mapping = {'int': 1, 'str': 2, 'bool': 3, 'List[int]': 4, 'List[float]': 5}
            for j, input_type in enumerate(input_types[:5]):
                context_features[i, 2 + j] = type_mapping.get(input_type, 0)
            
            for j, output_type in enumerate(output_types[:5]):
                context_features[i, 7 + j] = type_mapping.get(output_type, 0)
            
            # Program complexity features
            program_code = info.get('program_code', '')
            context_features[i, 12] = len(program_code.split('\n'))  # Number of lines
            context_features[i, 13] = program_code.count('if')       # Number of conditions
            context_features[i, 14] = program_code.count('for') + program_code.count('while')  # Loops
            context_features[i, 15] = program_code.count('return')   # Returns
        
        # Combine graph embeddings with context
        combined = torch.cat([graph_embeddings, context_features], dim=-1)
        
        # Project back to hidden_dim
        output = self.context_proj(combined)
        
        return output


def test_graph_encoder():
    """Test function for the graph encoder."""
    print("Testing GraphEncoder...")
    
    hidden_dim = 64
    batch_size = 3
    num_nodes = 10
    
    # Create dummy data
    x = torch.randn(num_nodes, 6)  # 6 features as expected by NodeEmbedding
    edge_index = torch.randint(0, num_nodes, (2, 15))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size)
    
    # Pad batch to match num_nodes
    while len(batch) < num_nodes:
        batch = torch.cat([batch, torch.tensor([batch_size - 1])])
    batch = batch[:num_nodes]
    
    data = Batch(x=x, edge_index=edge_index, batch=batch)
    
    # Test encoder
    encoder = ProgramGraphEncoder(hidden_dim=hidden_dim)
    
    with torch.no_grad():
        output = encoder(data)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {hidden_dim})")
    
    assert output.shape == (batch_size, hidden_dim), f"Expected shape ({batch_size}, {hidden_dim}), got {output.shape}"
    print("✅ Test passed!")


if __name__ == "__main__":
    test_graph_encoder()