# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a fork of HRM (Hierarchical Reasoning Model) adapted for **program synthesis using Graph Neural Networks and AST generation**. The core idea is leveraging HRM's hierarchical reasoning to generate code by:

- **H-module**: Making high-level algorithmic decisions (strategy, patterns, complexity)
- **L-module**: Handling detailed implementation (syntax, variables, constructs)

The system generates programs as graph-based AST structures rather than token sequences, using multi-modal specification tensors as input and immediate execution feedback for training.

## Common Development Commands

### Setup and Installation
```bash
# Install Python dependencies
uv pip install -e .

# run python script to test something
uv run python ...

# add depdencies
uv add ...

# Login to Weights & Biases for experiment tracking
wandb login
```

### Dataset Preparation

**Note**: The original HRM datasets (ARC, Sudoku, Maze) are included for reference, but this fork focuses on program synthesis tasks.

```bash
# Build ARC-1 dataset (960 examples) - for baseline comparison
python dataset/build_arc_dataset.py

# Build ARC-2 dataset (1120 examples) - for baseline comparison
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

# Build Sudoku dataset (1000 examples for quick demo) - for baseline comparison
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Build Maze dataset - for baseline comparison
python dataset/build_maze_dataset.py
```

**Program Synthesis Datasets** (to be implemented):
```bash
# Build program synthesis dataset with multi-modal specifications as input and correct AST-program as output
python dataset/build_program_synthesis_dataset.py --output-dir data/program-synthesis-100 --subsample-size 100

```

### Training Commands
```bash
# Single GPU training (laptop-friendly Sudoku demo)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Multi-GPU training (8 GPUs assumed)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py

# ARC-2 training
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

### Evaluation
```bash
# Evaluate trained model
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>

# Use arc_eval.ipynb notebook to inspect ARC results
```

## Architecture Overview

This fork adapts HRM (Hierarchical Reasoning Model) for **program synthesis via graph-based AST generation**. The core architecture leverages HRM's hierarchical structure for code generation tasks.

### Program Synthesis Architecture

See `MODEL.md` for the detailed design of the program synthesis system.

### Core Components

- **HRM Model** (`models/hrm/hrm_act_v1.py`): Adapted for program synthesis with:
  - **H-module**: High-level algorithmic decisions (strategy, patterns, complexity)
  - **L-module**: Detailed implementation (syntax, variables, constructs)
  - **ACT mechanism**: Q-learning for halt/continue decisions in AST refinement

- **Graph Neural Networks** (to be implemented): Multi-layer GNN processing for AST refinement
- **Multi-component Loss**: Balancing AST generation, execution correctness, and constraint compliance
- **Execution Feedback**: Immediate verification through code execution and testing

### Dataset Architecture

- **Program Synthesis Datasets** (to be implemented): Multi-modal specifications with target ASTs
- **AST Processing** (to be implemented): Graph structure handling and GNN integration
- **Execution Environment** (to be implemented): Code generation, compilation, and testing infrastructure

### Training Infrastructure

- **Pretraining** (`pretrain.py`): Adapted for program synthesis tasks with multi-component feedback
- **Evaluation** (`evaluate.py`): Extended with code execution and correctness metrics
- **Hierarchical Supervision**: Segment-wise targets for H-module and L-module decisions

### Key Design Principles for Program Synthesis

1. **Hierarchical Code Generation**: H-module for algorithmic strategy, L-module for implementation details
2. **Graph-based AST**: Unlimited program space without fixed patterns
3. **Multi-modal Input**: Rich specification encoding with signatures, constraints, and metadata
4. **Immediate Verification**: Generated code is executed and tested for correctness
5. **Adaptive Computation**: Dynamic refinement based on solution quality

### Implementation Roadmap

1. **Multi-modal input encoding**: Specification tensor representation
2. **Graph AST generation**: Core output structure with GNN processing
3. **Execution feedback**: Code generation, compilation, and testing
4. **Hierarchical supervision**: Segment-wise training targets
5. **ACT integration**: Adaptive computation for AST refinement

The adapted architecture maintains HRM's strengths (hierarchical reasoning, small-sample learning) while extending to structured code generation through graph neural networks and immediate execution feedback.


## Development strategy

1. Let's start with very simple tasks like sum, multiply, divide, and then move on to more complex tasks like sorting, searching, and graph algorithms.
- very simple programs like sum_up_to_n, multiply_up_to_n, divide_up_to_n, etc.
...



