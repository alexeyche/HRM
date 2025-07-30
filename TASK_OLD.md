THis is old version of the task description and deprecated. Please refer to `MODEL.md` for the latest version.

# HRM Program Synthesis System Design Summary

## Core Architecture Philosophy

We're building a **Graph Neural Network-based AST generation system** that leverages HRM's hierarchical reasoning capabilities. The key insight is that HRM's two-module structure maps perfectly to program synthesis:

- **H-module**: Makes high-level algorithmic decisions (strategy, patterns, complexity)
- **L-module**: Handles detailed implementation (syntax, variable names, specific constructs)

This avoids the limitations of fixed code patterns while maintaining an unlimited program search space through graph-based AST generation.

## Input Structure: Multi-Modal Specification Tensor

The input is a **fixed-size, multi-channel tensor** that HRM can see entirely at once (leveraging its strength over sequence models):

### **Channel 1: Signature Specification**
```
Fixed-vocabulary encoding of function signatures:
- Input types (Array[Int], String, etc.)
- Output types
```

### **Channel 2: Constraints/Requirements**
```
Logical predicates encoded as vectors:
- Preconditions (array.length >= 0)
- Postconditions (sorted(result), same_elements(input, result))
- Performance constraints (O(n log n) time complexity)
```

### **Channel 3: Metadata**
```
High-level guidance information:
- Algorithm class hints (divide_and_conquer, dynamic_programming)
- Time/space complexity targets
- Style preferences
```

### **Binary Attention Masks**
```
Masks indicating which parts of each channel contain valid information:
- signature_mask: [1,1,1,0,0,...] where signature data exists
- constraints_mask: [1,1,0,0,0,...] where constraints exist
- metadata_mask: [1,1,1,0,0,...] where metadata exists
```

**Key Advantage**: This structure scales to arbitrary specification complexity while maintaining fixed tensor dimensions.

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

## Hierarchical Target Structure

Following HRM's deep supervision approach, we provide targets at multiple reasoning levels:

### **Curriculum learning**
- Start with simpler problems and gradually increase difficulty, which helps provide more frequent positive signals early in training.

### **Multiple test cases per problem**
- Even if the final metric is binary, during training we can get partial credit by passing some subset of test cases, which provides intermediate learning signal. This is important for the model to learn from the examples.

### **Auxiliary losses**

- In some experiments they include things like program length penalties or intermediate supervision on program structure.

### **Smoothing mechanisms**

- Partial execution rewards - Credit for passing subsets of test cases
- Structural similarity losses - Compare generated AST structure to ground truth (if available)
- Intermediate execution states - Reward programs that produce correct intermediate values even if final output is wrong
- Soft program similarity - Use embedding-based similarity between generated and target programs
- Progressive test case difficulty - Start with easier cases and gradually add harder ones.

The key insight from HRM is that even with sparse rewards, the hierarchical structure helps a lot with credit assignment - H layer changes affect global program behavior while L layer changes are more local, making it easier to identify what went wrong.

## Multi-Component Feedback System

The system receives rich, immediate feedback from multiple sources:

### **1. Correctness Feedback**
```
Binary execution results:
- Syntax validity (can the AST be parsed?)
- Test case results (does generated code pass examples?)
- Constraint satisfaction (are requirements met?)
```

### **2. Structural Feedback**
```
AST quality metrics:
- Graph connectivity (is the AST well-formed?)
- Node type consistency (valid parent-child relationships?)
- Semantic coherence (do variable uses match definitions?)
```

### **3. Performance Feedback**
```
Efficiency analysis:
- Time complexity matching target
- Space usage evaluation
- Code quality metrics
```

### **4. Hierarchical Rewards**
```
Level-appropriate feedback:
- H-module: rewarded for good algorithmic choices
- L-module: rewarded for correct implementation details
- Combined: overall execution success
```

## Adaptive Computation Time (ACT) Integration

Following HRM's Q-learning approach for halt/continue decisions:

### **Q-Value System**
The model learns when to stop refining the generated AST:
- **Q_halt**: Expected reward for outputting current AST
- **Q_continue**: Expected reward for another refinement iteration

### **Decision Factors**
```
Factors influencing halt/continue:
- Test passage rate (all examples passing?)
- Constraint satisfaction level (requirements met?)
- Confidence in current solution
- Maximum iteration limit reached
```

### **Training Signal**
```
Binary reward structure:
- Reward = 1 if final code is correct and efficient
- Reward = 0 otherwise
- Q-learning updates based on actual outcomes
```

## Loss Function Architecture

Multi-component loss balancing different objectives:

### **Primary Components**
```
1. AST Generation Loss: Cross-entropy for node types, structure
2. Deep Supervision Loss: Intermediate target matching
3. Execution Correctness Loss: Binary success/failure
4. Constraint Compliance Loss: Requirement satisfaction
5. Q-Learning Loss: Adaptive computation decisions
```

### **Weighting Strategy**
```
Execution correctness gets highest weight (immediate feedback)
Deep supervision provides learning guidance
Q-learning enables efficiency optimization
```

## Training Data Format

Each training example contains:

### **Input Components**
```
- Multi-modal specification tensor (3 channels + masks)
- Target complexity and style preferences
- Example input-output pairs (encoded as patterns)
```

### **Target Components**
```
- Target AST structure (adjacency matrix + node features)
- Intermediate reasoning targets (segment-wise)
- Execution verification data (test cases)
```

### **Feedback Components**
```
- Immediate execution results
- Performance analysis
- Structural correctness metrics
```

## Key Advantages of This Design

### **1. Unlimited Program Space**
Graph-based AST generation avoids fixed pattern limitations while remaining computationally tractable.

### **2. Hierarchical Reasoning**
Natural mapping between HRM's two modules and program synthesis requirements (strategy vs. implementation).

### **3. Immediate Verification**
Generated ASTs can be immediately executed and tested, providing rich training signals.

### **4. Scalable Complexity**
Fixed input/output tensors handle varying specification and program complexity.

### **5. Proven Foundations**
Built on successful GNN approaches from recent program synthesis research (GraphCodeBERT, IPA-GNN).

## Implementation Priority

1. **Start with multi-modal input encoding** - establish the specification representation
2. **Implement basic graph AST generation** - core output structure
3. **Add GNN refinement layers** - iterative improvement mechanism
4. **Integrate execution feedback** - immediate correctness signals
5. **Implement ACT mechanism** - adaptive computation optimization

This design provides a concrete roadmap for building an HRM-based program synthesis system that leverages the model's hierarchical reasoning strengths while addressing the fundamental challenges of variable-length, structured code generation.
