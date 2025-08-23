# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a fork of HRM (Hierarchical Reasoning Model) adapted for **program synthesis using Graph Neural Networks and AST generation**. The core idea is leveraging HRM's hierarchical reasoning to generate code by:

- **H-module**: Making high-level algorithmic decisions (strategy, patterns, complexity)
- **L-module**: Handling detailed implementation (syntax, variables, constructs)

The system generates programs as graph-based AST structures rather than token sequences, using multi-modal specification tensors as input and immediate execution feedback for training.

## Important development guideline

We work agile. It means that when implemented new components we follow the following process:
1. Add a skeleton component to the codebase doing very minimal business logic.
2. Write tests for the component.
3. Think of next piece of logic that can be safely incorporated into the component.
4. Write new tests for the new logic.
5. Repeat 3-4 until the component is complete.

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

**Note**: Focusing on program synthesis tasks.

**Program Synthesis Datasets**:
```bash
# Build program synthesis dataset with multi-modal specifications as input and correct AST-program as output
uv run python -m dataset.build_program_dataset --out ./data/programs-200 --n 200 --seed 123
```

### Program Synthesis Architecture

See `./MODEL.md` for the detailed design of the program synthesis system.

### Core Components

- NLTK grammar is defined in `./dataset/grammar.py`
  - it contains subset of python syntax, enough to cover programs from dataset in `./dataset/programs.py`
  - it has random generation of parseable python programs, but it's used rather for testing of the grammar, see `./tests/test_grammar.py`

- Contrained generation head is to be implemented in `./model/generation_head.py`
  - specs is in `./specs/GENERATION_HEAD.md`
  - tests are in `./tests/test_generation_head.py`  (To be implemented)


### Training Commands

To be implemented later.

### Evaluation Commands

To be implemented later.

## Architecture Overview

To be implemented later.
