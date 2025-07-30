# HRM Program Synthesis Model

This document describes input and outputs for the neural program synthesis based on recent work on Hierarchical Reasoning Models (HRM).

We will try to encode examples and program descriptions into a fixed-size vector, and then use the vector to generate a AST for the program.

## Inputs

### Encoding examples

Set-based architectures treat examples as an unordered set using models like Deep Sets or Set Transformers. These are permutation-invariant, which is often desirable since example order shouldn't matter. You encode each example individually, then aggregate with operations like mean pooling or attention-based pooling.

### Encoding program descriptions

Fixed-vocabulary encoding of function signatures:
- Input types (Array[Int], String, etc.)
- Output types (Array[Int], String, etc.)

#### **Basic Types (4 primitives)**
```
int     # 32-bit integers
float   # 64-bit floating point
str     # UTF-8 strings
bool    # Boolean values
```

#### **Container Types (3 containers + generics)**
```
List[T]         # Ordered sequences: List[int], List[str]
Dict[K,V]       # Key-value mappings: Dict[str,int], Dict[int,List[str]]
Set[T]          # Unique collections: Set[int], Set[str]
```

## Outputs

## Output Structure: Graph-Based AST Generation

Instead of generating token sequences, HRM generates a **complete AST as a graph structure**:

### **Core Components**
1. **Node Existence Vector**: Binary mask determining which of the max_nodes slots are used
2. **Adjacency Matrix**: Defines parent-child relationships in the AST
3. **Node Feature Tensors**: Type, operation, variable, constant information for each node
4. **Node Embeddings**: Rich learned representations after GNN processing

### **Variable-Length Handling**
The graph approach naturally handles variable AST sizes:
- Simple functions use few nodes (e.g., 5 nodes for basic sorting)
- Complex functions use more nodes (e.g., 30+ nodes for recursive algorithms)
- No arbitrary depth limitations - tree depth determined by graph structure, not processing layers

### **GNN Processing**
Multiple Graph Neural Network layers iteratively refine all nodes simultaneously:
- **Layer 1**: Each node learns from immediate neighbors
- **Layer 2**: Information propagates to neighbors-of-neighbors
- **Layer 3-4**: Broader context and final refinement

**Critical Insight**: GNN layers are refinement iterations, NOT depth limits. A 4-layer GNN can process ASTs of arbitrary depth.

