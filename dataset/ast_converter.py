"""
AST to Graph Converter

Converts Python AST nodes to PyTorch Geometric graph representations
for use with Graph Neural Networks.
"""

import ast
from typing import Dict, List, Tuple, Any, Optional, Set
import torch
from torch_geometric.data import Data

from dataset.ast_types import (
    ASTNodeType, OperatorType, EdgeType,
    PYTHON_AST_TO_ENUM, PYTHON_OPERATOR_TO_ENUM,
    get_node_category, is_semantic_node
)


class ASTGraph:
    """
    Represents an AST as a graph with nodes, edges, and features.
    
    Used as an intermediate representation before conversion to 
    PyTorch Geometric Data objects.
    """
    
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Tuple[int, int, EdgeType]] = []
        self.node_id_counter = 0
        self.variables: Dict[str, int] = {}  # variable name -> canonical ID
        self.variable_uses: Dict[str, List[int]] = {}  # variable -> list of node IDs using it
        
    def add_node(self, node_type: ASTNodeType, **features) -> int:
        """Add a node to the graph and return its ID."""
        node_id = self.node_id_counter
        self.node_id_counter += 1
        
        node_data = {
            'id': node_id,
            'type': node_type,
            'category': get_node_category(node_type),
            'is_semantic': is_semantic_node(node_type),
            **features
        }
        
        self.nodes.append(node_data)
        return node_id
    
    def add_edge(self, from_node: int, to_node: int, edge_type: EdgeType):
        """Add an edge between two nodes."""
        self.edges.append((from_node, to_node, edge_type))
    
    def register_variable(self, var_name: str) -> int:
        """Register a variable and return its canonical ID."""
        if var_name not in self.variables:
            self.variables[var_name] = len(self.variables)
        return self.variables[var_name]
    
    def add_variable_use(self, var_name: str, node_id: int):
        """Record that a node uses a variable."""
        if var_name not in self.variable_uses:
            self.variable_uses[var_name] = []
        self.variable_uses[var_name].append(node_id)


class ASTToGraphConverter:
    """
    Converts Python AST nodes to graph representations.
    
    Features extracted:
    - Node types and categories
    - Variable canonical IDs and usage patterns  
    - Operator types
    - Literal values (encoded appropriately)
    - Structural features (depth, position)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset converter state for new conversion."""
        self.graph = ASTGraph()
        self.depth_stack: List[int] = []
        self.current_depth = 0
    
    def convert(self, program_code: str) -> ASTGraph:
        """
        Convert program code to AST graph.
        
        Args:
            program_code: Python source code string
            
        Returns:
            ASTGraph object representing the program
        """
        self.reset()
        
        try:
            # Parse the Python code to AST
            tree = ast.parse(program_code)
            
            # Convert AST to graph
            self._visit_node(tree, parent_id=None)
            
            # Add variable connection edges
            self._add_variable_edges()
            
            return self.graph
            
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")
    
    def _visit_node(self, node: ast.AST, parent_id: Optional[int], edge_type: EdgeType = EdgeType.CHILD) -> int:
        """
        Visit an AST node and convert it to graph node(s).
        
        Args:
            node: AST node to visit
            parent_id: ID of parent graph node
            edge_type: Type of edge from parent to this node
            
        Returns:
            ID of the created graph node
        """
        self.current_depth = len(self.depth_stack)
        
        # Get node type
        node_type_name = node.__class__.__name__
        node_type = PYTHON_AST_TO_ENUM.get(node_type_name, ASTNodeType.MODULE)
        
        # Extract features based on node type
        features = self._extract_node_features(node)
        features['depth'] = self.current_depth
        features['ast_type'] = node_type_name
        
        # Create graph node
        node_id = self.graph.add_node(node_type, **features)
        
        # Add edge from parent
        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id, edge_type)
        
        # Visit children
        self.depth_stack.append(node_id)
        self._visit_children(node, node_id)
        self.depth_stack.pop()
        
        return node_id
    
    def _extract_node_features(self, node: ast.AST) -> Dict[str, Any]:
        """Extract features from an AST node."""
        features = {}
        
        if isinstance(node, ast.Name):
            # Variable name
            var_name = node.id
            var_id = self.graph.register_variable(var_name)
            features['variable_name'] = var_name
            features['variable_id'] = var_id
            features['value_type'] = 'variable'
            
            # Record variable usage
            self.graph.add_variable_use(var_name, len(self.graph.nodes))
            
        elif isinstance(node, ast.Constant):
            # Literal value
            value = node.value
            features.update(self._encode_literal(value))
            
        elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
            # Operator
            if hasattr(node, 'op'):
                op_name = node.op.__class__.__name__
                if op_name in PYTHON_OPERATOR_TO_ENUM:
                    features['operator'] = PYTHON_OPERATOR_TO_ENUM[op_name]
                    features['operator_name'] = op_name
            elif hasattr(node, 'ops') and node.ops:
                # Comparison with multiple operators
                op_name = node.ops[0].__class__.__name__
                if op_name in PYTHON_OPERATOR_TO_ENUM:
                    features['operator'] = PYTHON_OPERATOR_TO_ENUM[op_name]
                    features['operator_name'] = op_name
        
        elif isinstance(node, ast.FunctionDef):
            # Function definition
            features['function_name'] = node.name
            features['num_args'] = len(node.args.args)
            
        elif isinstance(node, ast.arg):
            # Function argument
            features['arg_name'] = node.arg
            
        return features
    
    def _encode_literal(self, value: Any) -> Dict[str, Any]:
        """
        Encode literal values following the strategy in ENCODING.md.
        
        Args:
            value: Literal value from AST
            
        Returns:
            Dictionary of encoded features
        """
        features = {}
        
        if isinstance(value, bool):
            # Boolean encoding - check bool before int since bool is subclass of int
            features['value_type'] = 'bool'
            features['bool_value'] = value
            
        elif isinstance(value, int):
            # Integer encoding with bucketing and special values
            features['value_type'] = 'int'
            features['int_value'] = value
            
            # Magnitude bucket
            if value == 0:
                bucket = 0
            else:
                import math
                bucket = min(max(math.floor(math.log10(abs(value))), -5), 5)
            features['magnitude_bucket'] = bucket
            
            # Special small constants vocabulary
            small_constants = [-1, 0, 1, 2, 3, 4, 5, 10, 100]
            features['is_small_constant'] = value in small_constants
            features['small_constant_id'] = small_constants.index(value) if value in small_constants else -1
            
            # Sign and parity
            features['sign'] = 1 if value > 0 else (-1 if value < 0 else 0)
            features['is_zero'] = value == 0
            features['parity'] = value % 2 if value != 0 else 0
            
        elif isinstance(value, str):
            # String encoding
            features['value_type'] = 'str'
            features['str_value'] = value
            features['str_length'] = len(value)
            
            # Length bucket
            if len(value) == 0:
                length_bucket = 0
            else:
                import math
                length_bucket = min(math.floor(math.log2(len(value))), 4)
            features['length_bucket'] = length_bucket
            
            # Character features (simplified)
            if value:
                features['first_char_ascii'] = ord(value[0]) if value else 0
                features['has_digits'] = any(c.isdigit() for c in value)
                features['has_alpha'] = any(c.isalpha() for c in value)
                features['is_alpha'] = value.isalpha()
                features['is_digit'] = value.isdigit()
            
        elif isinstance(value, float):
            # Float encoding
            features['value_type'] = 'float'
            features['float_value'] = value
            # Could add more sophisticated float encoding here
            
        elif value is None:
            # None encoding
            features['value_type'] = 'none'
            
        else:
            # Unknown literal type
            features['value_type'] = 'unknown'
            features['str_repr'] = str(value)
        
        return features
    
    def _visit_children(self, node: ast.AST, parent_id: int):
        """Visit all children of an AST node."""
        # Get all child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                # List of child nodes
                for i, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        child_id = self._visit_node(item, parent_id)
                        # Add sibling edges
                        if i > 0:
                            prev_sibling = parent_id + i  # Simplified - would need proper tracking
                            # self.graph.add_edge(prev_sibling, child_id, EdgeType.NEXT_SIBLING)
            elif isinstance(value, ast.AST):
                # Single child node
                self._visit_node(value, parent_id)
    
    def _add_variable_edges(self):
        """Add edges connecting variable definitions and uses."""
        for var_name, node_ids in self.graph.variable_uses.items():
            # Connect all uses of the same variable
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    # Add bidirectional edges between variable uses
                    self.graph.add_edge(node_ids[i], node_ids[j], EdgeType.DEF_USE)
                    self.graph.add_edge(node_ids[j], node_ids[i], EdgeType.USE_DEF)


def ast_to_pyg_data(ast_graph: ASTGraph) -> Data:
    """
    Convert ASTGraph to PyTorch Geometric Data object.
    
    Args:
        ast_graph: ASTGraph representation
        
    Returns:
        PyTorch Geometric Data object
    """
    if not ast_graph.nodes:
        # Empty graph
        return Data(x=torch.empty(0, 0), edge_index=torch.empty(2, 0, dtype=torch.long))
    
    num_nodes = len(ast_graph.nodes)
    
    # Create node features tensor
    # For now, use simple categorical encoding - will be enhanced later
    node_features = []
    for node in ast_graph.nodes:
        features = _encode_node_features(node)
        node_features.append(features)
    
    x = torch.stack(node_features) if node_features else torch.empty(0, 0)
    
    # Create edge index tensor
    if ast_graph.edges:
        edges = [(e[0], e[1]) for e in ast_graph.edges]  # Ignore edge types for now
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    
    # Create edge attributes (edge types)
    edge_attr = None
    if ast_graph.edges:
        edge_types = [e[2].value for e in ast_graph.edges]
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
    
    # Additional graph-level features
    graph_features = {
        'num_variables': len(ast_graph.variables),
        'max_depth': max((node.get('depth', 0) for node in ast_graph.nodes), default=0),
        'variable_names': list(ast_graph.variables.keys()),
        'variable_mapping': ast_graph.variables
    }
    
    return Data(
        x=x,
        edge_index=edge_index, 
        edge_attr=edge_attr,
        **graph_features
    )


def _encode_node_features(node_data: Dict[str, Any]) -> torch.Tensor:
    """
    Encode node features as a tensor.
    
    For now, this is a simple categorical encoding.
    Will be enhanced with proper embedding layers later.
    """
    features = []
    
    # Node type (categorical)
    node_type = node_data['type']
    features.append(float(node_type.value))
    
    # Depth
    features.append(float(node_data.get('depth', 0)))
    
    # Is semantic flag
    features.append(float(node_data.get('is_semantic', False)))
    
    # Variable ID (if applicable)
    features.append(float(node_data.get('variable_id', -1)))
    
    # Integer value (if applicable)
    features.append(float(node_data.get('int_value', 0)))
    
    # Value type encoding
    value_type = node_data.get('value_type', 'none')
    type_encodings = {'int': 1, 'str': 2, 'bool': 3, 'float': 4, 'none': 0, 'variable': 5}
    features.append(float(type_encodings.get(value_type, 0)))
    
    return torch.tensor(features, dtype=torch.float32)


# Convenience function
def program_to_graph(program_code: str) -> Data:
    """
    Convert program code directly to PyTorch Geometric Data.
    
    Args:
        program_code: Python source code string
        
    Returns:
        PyTorch Geometric Data object
    """
    converter = ASTToGraphConverter()
    ast_graph = converter.convert(program_code)
    return ast_to_pyg_data(ast_graph)