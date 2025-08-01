# Dataset Compression Plan

## Current State Analysis

**Problem**: Severe dimensionality mismatch with limited training data
- **Samples**: 134 total samples
- **Input dimensions**: 512 (with ~450 zero padding)
- **Output dimensions**: 2650 (sparse adjacency matrix represented densely)
- **Sample/dimension ratio**: 134/(512+2650) = 0.042 (severely underdetermined)

## Compression Strategy

### Phase 1: Input Compression (512 → 80-120 dims)

#### 1.1 Remove Fixed Padding
```python
# Current: Fixed 512-dim vector with massive padding
spec_vector = np.zeros(512, dtype=np.float32)

# New: Dynamic length encoding
components = []  # Variable length, typically 80-120 dims
```

#### 1.2 Add Missing Type Information (Currently Missing!)
```python
# Type vocabulary for MODEL.md specification
TYPE_VOCAB = {
    # Basic types (4 primitives)
    'int': 1, 'float': 2, 'str': 3, 'bool': 4,
    
    # Container types (3 containers + generics)
    'List[int]': 5, 'List[float]': 6, 'List[str]': 7, 'List[bool]': 8,
    'Dict[str,int]': 9, 'Dict[str,str]': 10, 'Dict[int,str]': 11,
    'Set[int]': 12, 'Set[str]': 13, 'Set[float]': 14
}

def encode_types(type_specs: List[Dict]) -> List[int]:
    return [TYPE_VOCAB.get(spec.get('type', 'int'), 0) for spec in type_specs]
```

#### 1.3 Individual Example Encoding (No Aggregation)
```python
def encode_single_example(example: Dict) -> List[float]:
    """Encode one example as compact vector"""
    if isinstance(example["input"], list):
        # Variable-length input handling
        inputs = example["input"][:8] + [0] * max(0, 8 - len(example["input"]))
    else:
        inputs = [example["input"]] + [0] * 7
    
    output = [example["output"]]
    return inputs + output  # 9 dims per example
```

#### 1.4 Optional Description Encoding
```python
def simple_text_hash(text: str) -> List[int]:
    """Simple hash-based text encoding for descriptions"""
    # Use hash of description for basic semantic signal
    hash_val = abs(hash(text))
    return [
        (hash_val >> i) & 0xFF for i in range(0, 32, 8)  # 4 hash bytes
    ]
```

#### 1.5 Complete Compressed Input Encoding
```python
def encode_program_specification_compressed(spec: Dict[str, Any]) -> np.ndarray:
    components = []
    
    # Basic metadata (3 dims)
    components.extend([
        len(spec["inputs"]), 
        len(spec["outputs"]), 
        len(spec["examples"])
    ])
    
    # Type information (2-8 dims) - CURRENTLY MISSING!
    input_types = encode_types(spec["inputs"])
    output_types = encode_types(spec["outputs"])
    components.extend(input_types + output_types)
    
    # Description hash (4 dims)
    desc_hash = simple_text_hash(spec["description"])
    components.extend(desc_hash)
    
    # Individual examples (9 dims each, variable count)
    for example in spec["examples"]:
        components.extend(encode_single_example(example))
    
    return np.array(components, dtype=np.float32)
    # Total: ~15 header + 9*num_examples = ~42-195 dims
```

### Phase 2: Output Compression (2650 → 200-300 dims)

#### 2.1 Current Dense Representation Problems
```python
# Current: Massive waste with dense adjacency matrix
node_exists = np.zeros(50, dtype=bool)        # 50 dims
adjacency = np.zeros((50, 50), dtype=bool)    # 2500 dims (mostly False!)
node_types = np.zeros(50, dtype=np.int32)     # 50 dims
node_values = np.zeros(50, dtype=np.int32)    # 50 dims
# Total: 2650 dims for ~5-15 actual nodes
```

#### 2.2 Sparse Graph Representation
```python
def ast_to_sparse_representation(code: str) -> Dict[str, np.ndarray]:
    """Convert AST to efficient sparse representation"""
    tree = ast.parse(code)
    
    nodes = []
    edges = []
    node_counter = 0
    
    def traverse(node, parent_id=None):
        nonlocal node_counter
        current_id = node_counter
        node_counter += 1
        
        # Store node info compactly
        simplified = simplify_ast_node(node)
        nodes.append({
            'type': simplified['type'],
            'value': simplified.get('value', 0),
            'params': simplified.get('params', 0)
        })
        
        # Store edge (parent -> child)
        if parent_id is not None:
            edges.append((parent_id, current_id))
        
        # Traverse children
        for child in ast.iter_child_nodes(node):
            traverse(child, current_id)
    
    traverse(tree.body[0])  # Start from function definition
    
    # Convert to efficient arrays
    num_nodes = len(nodes)
    
    return {
        'num_nodes': num_nodes,                                    # 1 dim
        'node_types': np.array([n['type'] for n in nodes]),       # num_nodes dims
        'node_values': np.array([n['value'] for n in nodes]),     # num_nodes dims
        'node_params': np.array([n['params'] for n in nodes]),    # num_nodes dims
        'edge_list': np.array(edges) if edges else np.array([]).reshape(0, 2),  # num_edges × 2
    }
    # Total: 1 + 3*num_nodes + 2*num_edges = ~50-200 dims (vs 2650)
```

#### 2.3 Flattened Sparse Output
```python
def flatten_sparse_ast(sparse_ast: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten sparse AST to fixed-size tensor for training"""
    max_nodes = 30  # Reduced from 50
    max_edges = 25  # New: explicit edge limit
    
    components = []
    
    # Number of actual nodes/edges
    num_nodes = sparse_ast['num_nodes']
    num_edges = len(sparse_ast['edge_list'])
    components.extend([num_nodes, num_edges])  # 2 dims
    
    # Node features (padded to max_nodes)
    for key in ['node_types', 'node_values', 'node_params']:
        padded = np.pad(sparse_ast[key], (0, max_nodes - num_nodes))[:max_nodes]
        components.extend(padded)  # 3 × max_nodes = 90 dims
    
    # Edge list (padded to max_edges)
    edges_flat = sparse_ast['edge_list'].flatten()
    edges_padded = np.pad(edges_flat, (0, 2*max_edges - len(edges_flat)))[:2*max_edges]
    components.extend(edges_padded)  # 2 × max_edges = 50 dims
    
    return np.array(components, dtype=np.int32)
    # Total: 2 + 90 + 50 = 142 dims (vs 2650)
```

### Phase 3: Sample Generation (134 → 1000+ samples)

#### 3.1 Expanded Program Templates

##### Mathematical Operations (20 templates)
```python
MATH_TEMPLATES = {
    "gcd": "Calculate greatest common divisor",
    "lcm": "Calculate least common multiple", 
    "fibonacci": "Generate nth Fibonacci number",
    "prime_check": "Check if number is prime",
    "digit_sum": "Sum of digits in a number",
    "reverse_number": "Reverse digits of a number",
    "perfect_square": "Check if number is perfect square",
    "factorial_mod": "Factorial modulo p",
    # ... 12 more
}
```

##### Array/List Operations (15 templates)
```python
ARRAY_TEMPLATES = {
    "array_sum": "Sum all elements in array",
    "array_max": "Find maximum element",
    "array_sort": "Sort array in ascending order",
    "array_reverse": "Reverse array elements",
    "array_unique": "Remove duplicate elements",
    "array_filter_positive": "Keep only positive numbers",
    "array_map_double": "Double each element",
    "array_contains": "Check if element exists",
    # ... 7 more
}
```

##### String Operations (10 templates)
```python
STRING_TEMPLATES = {
    "string_length": "Get string length",
    "string_reverse": "Reverse string",
    "string_uppercase": "Convert to uppercase",
    "string_count_vowels": "Count vowel characters",
    "string_palindrome": "Check if string is palindrome",
    "string_concat": "Concatenate two strings",
    # ... 4 more
}
```

##### Conditional Logic (10 templates)
```python
CONDITIONAL_TEMPLATES = {
    "max_of_three": "Maximum of three numbers",
    "grade_classifier": "Classify grade (A/B/C/D/F)",
    "sign_function": "Return sign of number (-1/0/1)",
    "triangle_type": "Classify triangle (equilateral/isosceles/scalene)",
    "leap_year": "Check if year is leap year",
    # ... 5 more
}
```

#### 3.2 Data Augmentation Strategy

##### Parameter Variation
```python
def generate_augmented_examples(template_name: str, base_examples: List, count: int) -> List:
    """Generate more examples with parameter variation"""
    augmented = base_examples.copy()
    
    if template_name == "sum_up_to_n":
        # Vary range and include edge cases
        for _ in range(count - len(augmented)):
            n = np.random.choice([0, 1] + list(range(2, 50)) + [100, 200])
            output = sum(range(1, n + 1))
            augmented.append({"input": n, "output": output})
    
    # Similar for other templates...
    return augmented[:count]
```

##### Structural Variation (Same Logic, Different Implementation)
```python
IMPLEMENTATION_VARIANTS = {
    "sum_up_to_n": [
        "def program(n): return sum(range(1, n + 1))",           # Built-in sum
        "def program(n): return n * (n + 1) // 2",              # Formula
        "def program(n):\n    s = 0\n    for i in range(1, n+1):\n        s += i\n    return s"  # Loop
    ]
}
```

#### 3.3 Sample Distribution Target
```
- Mathematical: 300 samples (20 templates × 15 samples each)
- Array operations: 225 samples (15 templates × 15 samples each)  
- String operations: 150 samples (10 templates × 15 samples each)
- Conditional logic: 150 samples (10 templates × 15 samples each)
- Current simple math: 175 samples (enhanced existing 22 templates)
Total: 1000+ samples
```

## Implementation Timeline

### Week 1: Core Compression
- [ ] Implement input compression functions
- [ ] Implement sparse output representation  
- [ ] Modify dataset building pipeline
- [ ] Basic validation with small dataset

### Week 2: Template Expansion
- [ ] Add 20 mathematical operation templates
- [ ] Add 15 array operation templates
- [ ] Add 10 string operation templates
- [ ] Add 10 conditional logic templates

### Week 3: Augmentation & Validation
- [ ] Implement data augmentation strategies
- [ ] Generate full 1000+ sample dataset
- [ ] Validate with existing HRM model
- [ ] Benchmark training efficiency improvements

## Expected Results

### Compression Gains
- **Input**: 512 → 80-120 dims (4-6x compression)
- **Output**: 2650 → 142 dims (18x compression)  
- **Samples**: 134 → 1000+ (7.5x increase)

### Training Efficiency
- **Memory usage**: ~20x reduction from compression
- **Sample efficiency**: Better generalization with more diverse data
- **Convergence speed**: Faster training with lower dimensionality

### Validation Metrics
- [ ] Model can still process compressed representations
- [ ] Training loss decreases faster than baseline
- [ ] Generated ASTs are valid and executable
- [ ] Execution accuracy maintained or improved

## Risk Mitigation

### Potential Issues
1. **Information loss**: Compression might remove important signals
2. **Model compatibility**: Existing HRM might not handle new format
3. **Edge cases**: Complex programs might not fit sparse representation

### Mitigation Strategies
1. **Gradual rollout**: Test compression incrementally
2. **Backward compatibility**: Keep both dense and sparse formats initially
3. **Validation suite**: Comprehensive testing with known good programs
4. **Monitoring**: Track both compression ratio and model performance