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

#### Program example

Let's use yaml to describe the program.

```yaml
name: sum_up_to_n
description: "Sum up all numbers up to the input number N"
inputs:
  - type: int
    description: "The input number N"
outputs:
  - type: int
    description: "The sum of all numbers up to the input number N"
examples:
  - input: 10
    output: 55
  - input: 20
    output: 210
  - input: 30
    output: 465
```


## Outputs

TBD: see `./specs/GENERATION_HEAD.md`