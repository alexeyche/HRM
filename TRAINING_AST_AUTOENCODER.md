# AST Autoencoder Training Guide

This guide covers how to train the AST autoencoder for program synthesis using the `train_ast_autoencoder.py` script.

## Quick Start

### 1. Generate a Program Dataset

First, create a program synthesis dataset:

```bash
# Generate a small test dataset (5 samples)
uv run python -m dataset.build_program_dataset --out ./test_dataset --n 5 --seed 42

# Generate a larger dataset for actual training
uv run python -m dataset.build_program_dataset --out ./data/programs-1000 --n 1000 --seed 123
```

### 2. Train the AST Autoencoder

Basic training with default settings:

```bash
uv run python train_ast_autoencoder.py --data_dir ./data/programs-1000
```

### 3. Monitor Training

The script will:
- Create an 80/20 train/validation split
- Train for the specified number of epochs
- Evaluate every epoch (or as specified by `--eval_interval`)
- Log metrics to WandB (if enabled)
- Save model checkpoints and artifacts

## Command Line Options

### Required Arguments
- `--data_dir`: Path to program dataset directory (must contain `graphs/` subdirectory)

### Model Configuration
- `--hidden_dim`: Hidden dimension for autoencoder (default: 128)
- `--encoder_layers`: Number of GCN layers in encoder (default: 3)
- `--max_decode_steps`: Maximum decoding steps (default: 100)

### Training Configuration
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 8)
- `--lr`: Learning rate (default: 1e-3)
- `--eval_interval`: Evaluate every N epochs (default: 1)
- `--log_interval`: Log training metrics every N steps (default: 10)

### Infrastructure
- `--device`: Device to use (auto/cpu/cuda/mps, default: auto)
- `--seed`: Random seed (default: 42)
- `--resume`: Path to checkpoint to resume from

### Experiment Tracking
- `--wandb_project`: WandB project name (default: "AST-Autoencoder")
- `--wandb_run`: WandB run name (default: auto-generated)
- `--no_wandb`: Disable WandB logging
- `--artifacts_dir`: Directory for training artifacts (default: "training_artifacts")

## Example Commands

### Basic Training
```bash
uv run python train_ast_autoencoder.py --data_dir ./data/programs-200
```

### Training with Custom Settings
```bash
uv run python train_ast_autoencoder.py \
    --data_dir ./data/programs-1000 \
    --epochs 50 \
    --batch_size 16 \
    --lr 5e-4 \
    --hidden_dim 256 \
    --eval_interval 5 \
    --log_interval 5 \
    --wandb_run "large_model_experiment"
```

### Training without WandB
```bash
uv run python train_ast_autoencoder.py \
    --data_dir ./data/programs-200 \
    --no_wandb \
    --epochs 20
```

### Resume Training from Checkpoint
```bash
uv run python train_ast_autoencoder.py \
    --data_dir ./data/programs-200 \
    --resume ./training_artifacts/run_20250824_143040/checkpoint_epoch_10.pt
```

### Offline WandB Mode
```bash
WANDB_MODE=offline uv run python train_ast_autoencoder.py --data_dir ./data/programs-200
```

## Output Artifacts

Training creates the following artifacts in the `training_artifacts/` directory:

- `config.json`: Training configuration and metadata
- `checkpoint_epoch_X.pt`: Model checkpoints (saved every 5 epochs)
- `best_model.pt`: Best model based on validation loss
- `final_model.pt`: Final model state after training
- `validation_epoch_X.json`: Detailed validation results (every 10 epochs)

## Evaluation Metrics

The training script tracks comprehensive metrics:

### Training Metrics
- **Loss**: Total reconstruction loss
- **Reconstruction Loss**: Component of loss from program reconstruction
- **Similarity Score**: Token-level similarity between original and reconstructed programs
- **Exact Match Rate**: Rate of perfect program reconstructions

### Validation Metrics
- **Compilation Rate**: Percentage of generated programs that compile successfully
- **One Example Rate**: Percentage of programs that satisfy at least one test example
- **All Examples Rate**: Percentage of programs that satisfy all test examples
- All training metrics computed on validation set

## Model Architecture

The AST autoencoder consists of:

1. **Graph Encoder**: Multi-layer GCN that encodes AST graphs into latent embeddings
2. **Generation Head**: Grammar-constrained decoder that generates syntactically valid programs
3. **Bridge Layer**: Transforms graph embeddings to decoder input format

Key features:
- Grammar-constrained generation ensures syntactic validity
- Multi-modal loss combining reconstruction and semantic similarity
- Copy mechanism for identifier reuse
- Smart production selection to avoid infinite recursion

## Step-wise Logging

The training script provides detailed step-wise logging to monitor training progress in real-time:

### Features
- **Real-time feedback**: See loss and similarity metrics every N training steps
- **Running averages**: Metrics are averaged over the last N steps to smooth out noise
- **WandB integration**: Step-wise metrics are automatically logged to WandB for detailed tracking
- **Configurable interval**: Use `--log_interval` to control frequency (default: every 10 steps)

### Examples
```bash
# Log every step (verbose)
uv run python train_ast_autoencoder.py --data_dir ./data/programs-200 --log_interval 1

# Log every 5 steps (balanced)
uv run python train_ast_autoencoder.py --data_dir ./data/programs-200 --log_interval 5

# Log every 50 steps (minimal)
uv run python train_ast_autoencoder.py --data_dir ./data/programs-200 --log_interval 50
```

### WandB Metrics
Step-wise logging adds these metrics to WandB:
- `train/step_loss`: Running average loss over recent steps
- `train/step_similarity`: Running average similarity over recent steps  
- `step`: Global step counter across all epochs
- `train/step_batch_idx`: Current batch within epoch

## Tips for Training

### For Small Datasets (< 100 samples)
```bash
uv run python train_ast_autoencoder.py \
    --data_dir ./small_dataset \
    --epochs 50 \
    --batch_size 4 \
    --hidden_dim 64 \
    --max_decode_steps 50
```

### For Large Datasets (> 1000 samples)
```bash
uv run python train_ast_autoencoder.py \
    --data_dir ./large_dataset \
    --epochs 20 \
    --batch_size 32 \
    --hidden_dim 256 \
    --eval_interval 2 \
    --lr 5e-4
```

### GPU Training
```bash
uv run python train_ast_autoencoder.py \
    --data_dir ./data/programs-5000 \
    --device cuda \
    --batch_size 64 \
    --epochs 100
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--batch_size` or `--hidden_dim`
2. **Training Too Slow**: Increase `--eval_interval` or reduce `--max_decode_steps`
3. **Low Compilation Rate**: Check dataset quality and increase `--max_decode_steps`
4. **WandB Issues**: Use `--no_wandb` or `WANDB_MODE=offline`

### Validation Metrics

- **Good Results**: Compilation rate > 0.8, One example rate > 0.3
- **Poor Results**: Compilation rate < 0.2, might need more training or better hyperparameters
- **Overfitting**: Training loss much lower than validation loss

## Next Steps

After training, you can:
1. Use the trained model for program synthesis
2. Fine-tune on specific program types
3. Integrate into larger systems
4. Analyze learned representations