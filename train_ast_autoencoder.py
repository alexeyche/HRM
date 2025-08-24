"""
Training script for AST autoencoder.

This script trains the AST autoencoder on program synthesis datasets with:
- 80/20 train/eval split
- Evaluation every epoch
- Compilation success and example satisfaction metrics
- WandB integration
- Organized training artifacts

Example usage:
    # Train on program dataset with default settings
    python train_ast_autoencoder.py --data_dir data/programs-200

    # Train with custom settings and WandB logging
    python train_ast_autoencoder.py --data_dir data/programs-200 --epochs 20 --batch_size 16 --lr 5e-4 --wandb_run experiment_1

    # Train without WandB logging
    python train_ast_autoencoder.py --data_dir data/programs-200 --no_wandb
"""

import argparse
import ast
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
# import torch.nn.functional as F  # Currently unused
from torch_geometric.data import Batch
import wandb
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

from dataset.build_program_dataset import load_sample
from models.ast_autoencoder import ASTAutoencoder, ASTAutoencoderTrainer
from dataset.programs import get_program_registry

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Rich console for better output
console = Console()


def clean_graph_for_batching(graph):
    """Clean graph by keeping only essential tensor attributes for batching."""
    from torch_geometric.data import Data
    clean_graph = Data()
    clean_graph.x = graph.x
    clean_graph.edge_index = graph.edge_index

    # Only add edge_attr if it exists and is a tensor
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        clean_graph.edge_attr = graph.edge_attr

    return clean_graph


def load_dataset_samples(dataset_path: str) -> Tuple[List[Any], List[Dict], List[int]]:
    """Load all samples from dataset directory."""
    dataset_dir = Path(dataset_path)
    graphs_dir = dataset_dir / "graphs"

    if not graphs_dir.exists():
        raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")

    # Get all .pt files
    pt_files = sorted(list(graphs_dir.glob("*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {graphs_dir}")

    graphs, infos, indices = [], [], []
    for pt_file in pt_files:
        try:
            graph, info, index = load_sample(str(pt_file))
            graphs.append(clean_graph_for_batching(graph))
            infos.append(info)
            indices.append(index)
        except Exception as e:
            log.warning(f"Failed to load {pt_file}: {e}")
            continue

    log.info(f"Loaded {len(graphs)} samples from {dataset_path}")
    return graphs, infos, indices


def split_data(graphs: List[Any], infos: List[Dict], indices: List[int],
               train_ratio: float = 0.8, seed: int = 42) -> Tuple[
    List[Any], List[Any], List[Dict], List[Dict], List[int], List[int]
]:
    """Split data into train and validation sets."""
    np.random.seed(seed)
    n_samples = len(graphs)

    # Create random permutation
    perm = np.random.permutation(n_samples)
    split_idx = int(train_ratio * n_samples)

    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_infos = [infos[i] for i in train_idx]
    val_infos = [infos[i] for i in val_idx]
    train_indices = [indices[i] for i in train_idx]
    val_indices = [indices[i] for i in val_idx]

    log.info(f"Split data: {len(train_graphs)} train, {len(val_graphs)} val")
    return train_graphs, val_graphs, train_infos, val_infos, train_indices, val_indices


def create_batches(graphs: List[Any], infos: List[Dict], batch_size: int) -> List[Tuple[Batch, List[Dict]]]:
    """Create batches from graphs and infos."""
    batches = []
    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i:i + batch_size]
        batch_infos = infos[i:i + batch_size]

        try:
            batched_graphs = Batch.from_data_list(batch_graphs)
            batches.append((batched_graphs, batch_infos))
        except Exception as e:
            log.warning(f"Failed to create batch starting at {i}: {e}")
            continue

    return batches


def train_one_epoch(model: ASTAutoencoder, trainer: ASTAutoencoderTrainer,
                   optimizer: torch.optim.Optimizer, train_batches: List[Tuple[Batch, List[Dict]]],
                   device: str, log_interval: int = 10, epoch: int = 1, 
                   use_wandb: bool = False, global_step: int = 0) -> Tuple[Dict[str, float], int]:
    """Train model for one epoch."""
    model.train()
    
    # Epoch-level metrics
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_similarity_score = 0.0
    total_exact_match_rate = 0.0
    num_batches = 0
    num_samples = 0
    
    # Step-wise metrics for running averages
    step_losses = []
    step_similarities = []
    current_step = global_step
    
    # Create progress bar for training batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Training"),
        BarColumn(complete_style="green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        train_task = progress.add_task(
            f"[cyan]Epoch {epoch}",
            total=len(train_batches)
        )

        for batch_idx, (batch_graphs, batch_infos) in enumerate(train_batches):
            batch_graphs = batch_graphs.to(device)
            batch_size = len(batch_infos)

            try:
                optimizer.zero_grad()

                # Forward pass through autoencoder
                result = model(batch_graphs, batch_infos, decode=True, max_steps=50, temperature=0.8)

                # Extract original program codes
                original_programs = [info.get('program_code', '') for info in batch_infos]
                reconstructed_programs = result['programs']
                latent = result['latent']

                # Compute reconstruction loss using trainer
                loss_dict = trainer.reconstruction_loss(original_programs, reconstructed_programs, latent)

                loss = loss_dict['total_loss']
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Accumulate epoch metrics
                total_loss += loss.item()
                total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
                total_similarity_score += loss_dict['similarity_score'].item()
                total_exact_match_rate += loss_dict['exact_match_rate'].item()
                num_batches += 1
                num_samples += batch_size
                
                # Track step-wise metrics
                step_losses.append(loss.item())
                step_similarities.append(loss_dict['similarity_score'].item())
                current_step += 1

                # Update progress bar with current metrics
                progress.update(
                    train_task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch} [dim]• Loss: {loss.item():.4f} • Sim: {loss_dict['similarity_score'].item():.3f}"
                )

                # Step-wise logging (reduced frequency when using progress bar)
                if (batch_idx + 1) % log_interval == 0:
                    # Calculate running averages for recent steps
                    recent_loss = sum(step_losses[-log_interval:]) / min(len(step_losses), log_interval)
                    recent_similarity = sum(step_similarities[-log_interval:]) / min(len(step_similarities), log_interval)
                    
                    # Log to WandB if enabled (console logging is replaced by progress bar)
                    if use_wandb:
                        wandb.log({
                            "step": current_step,
                            "epoch": epoch,
                            "train/step_loss": recent_loss,
                            "train/step_similarity": recent_similarity,
                            "train/step_batch_idx": batch_idx + 1,
                        })

            except Exception as e:
                console.print(f"[red]Training batch {batch_idx} failed: {e}")
                continue

    if num_batches == 0:
        return {}, current_step

    return {
        "train/loss": total_loss / num_batches,
        "train/reconstruction_loss": total_reconstruction_loss / num_batches,
        "train/similarity_score": total_similarity_score / num_batches,
        "train/exact_match_rate": total_exact_match_rate / num_batches,
        "train/samples": num_samples,
    }, current_step


def evaluate_model(model: ASTAutoencoder, trainer: ASTAutoencoderTrainer,
                  val_batches: List[Tuple[Batch, List[Dict]]], device: str) -> Dict[str, float]:
    """Evaluate model on validation set with comprehensive metrics."""
    model.eval()

    # Metrics tracking
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_similarity_score = 0.0
    total_exact_match_rate = 0.0
    num_batches = 0

    # Program evaluation metrics
    total_programs = 0
    compiled_programs = 0
    programs_with_one_example = 0
    programs_with_all_examples = 0

    registry = get_program_registry()

    with torch.no_grad():
        # Create progress bar for validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Validation"),
            BarColumn(complete_style="yellow"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            val_task = progress.add_task(
                "[yellow]Evaluating",
                total=len(val_batches)
            )
            
            for batch_idx, (batch_graphs, batch_infos) in enumerate(val_batches):
                batch_graphs = batch_graphs.to(device)

                try:
                    # Forward pass
                    result = model(batch_graphs, batch_infos, decode=True, max_steps=50, temperature=0.5)

                    # Extract programs and compute loss
                    original_programs = [info.get('program_code', '') for info in batch_infos]
                    reconstructed_programs = result['programs']
                    latent = result['latent']

                    # Compute reconstruction loss
                    loss_dict = trainer.reconstruction_loss(original_programs, reconstructed_programs, latent)

                    # Accumulate loss metrics
                    total_loss += loss_dict['total_loss'].item()
                    total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
                    total_similarity_score += loss_dict['similarity_score'].item()
                    total_exact_match_rate += loss_dict['exact_match_rate'].item()
                    num_batches += 1

                    # Evaluate each reconstructed program
                    for reconstructed_code, info in zip(reconstructed_programs, batch_infos):
                        total_programs += 1

                        # Check compilation
                        compilation_success = check_compilation(reconstructed_code)
                        if compilation_success:
                            compiled_programs += 1

                            # Check example satisfaction
                            if compilation_success:
                                spec_name = info.get('spec_name', '')
                                one_example, all_examples = check_example_satisfaction(
                                    reconstructed_code, spec_name, registry
                                )
                                if one_example:
                                    programs_with_one_example += 1
                                if all_examples:
                                    programs_with_all_examples += 1

                    # Update progress bar
                    progress.update(
                        val_task,
                        advance=1,
                        description=f"[yellow]Evaluating [dim]• Loss: {loss_dict['total_loss'].item():.4f} • Compiled: {compiled_programs}/{total_programs}"
                    )

                except Exception as e:
                    console.print(f"[red]Evaluation batch {batch_idx} failed: {e}")
                    continue

    if num_batches == 0:
        return {}

    # Calculate rates
    compilation_rate = compiled_programs / max(total_programs, 1)
    one_example_rate = programs_with_one_example / max(total_programs, 1)
    all_examples_rate = programs_with_all_examples / max(total_programs, 1)

    return {
        "val/loss": total_loss / num_batches,
        "val/reconstruction_loss": total_reconstruction_loss / num_batches,
        "val/similarity_score": total_similarity_score / num_batches,
        "val/exact_match_rate": total_exact_match_rate / num_batches,
        "val/compilation_rate": compilation_rate,
        "val/one_example_rate": one_example_rate,
        "val/all_examples_rate": all_examples_rate,
        "val/programs_evaluated": total_programs,
        "val/programs_compiled": compiled_programs,
    }


def check_compilation(code: str) -> bool:
    """Check if code compiles successfully."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def check_example_satisfaction(code: str, spec_name: str, registry) -> Tuple[bool, bool]:
    """Check if program satisfies examples. Returns (one_example_correct, all_examples_correct)."""
    spec = registry.get(spec_name)
    if spec is None:
        return False, False

    try:
        # Create execution environment
        local_vars = {}
        exec(code, {}, local_vars)

        # Find the program function
        program_func = None
        for name, obj in local_vars.items():
            if callable(obj) and name.startswith('program'):
                program_func = obj
                break

        if program_func is None:
            return False, False

        # Test examples
        correct_count = 0
        total_examples = len(spec.base_examples)

        for example in spec.base_examples:
            try:
                # Handle single vs multiple inputs
                if isinstance(example.input, list):
                    result = program_func(*example.input)
                else:
                    result = program_func(example.input)

                if result == example.output:
                    correct_count += 1
            except Exception:
                continue  # Example failed

        one_example_correct = correct_count > 0
        all_examples_correct = correct_count == total_examples

        return one_example_correct, all_examples_correct

    except Exception:
        return False, False


def main():
    parser = argparse.ArgumentParser(description="Train AST autoencoder for program synthesis")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to program dataset directory")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension for autoencoder")
    parser.add_argument("--encoder_layers", type=int, default=3,
                       help="Number of GCN layers in encoder")
    parser.add_argument("--max_decode_steps", type=int, default=100,
                       help="Maximum decoding steps")
    parser.add_argument("--device", type=str, default="mps",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training")
    parser.add_argument("--wandb_project", type=str, default="AST-Autoencoder",
                       help="WandB project name")
    parser.add_argument("--wandb_run", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--artifacts_dir", type=str, default="training_artifacts",
                       help="Directory for training artifacts")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--eval_interval", type=int, default=1,
                       help="Evaluate every N epochs")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log training metrics every N steps")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        warnings.warn("MPS not available, falling back to CPU")
        device = "cpu"

    log.info(f"Using device: {device}")

    # Load dataset
    log.info(f"Loading dataset from {args.data_dir}")
    graphs, infos, indices = load_dataset_samples(args.data_dir)

    if len(graphs) == 0:
        raise ValueError("No valid samples loaded from dataset")

    # Split data
    train_graphs, val_graphs, train_infos, val_infos, _, _ = split_data(
        graphs, infos, indices, seed=args.seed
    )

    # Create batches
    train_batches = create_batches(train_graphs, train_infos, args.batch_size)
    val_batches = create_batches(val_graphs, val_infos, args.batch_size)

    log.info(f"Created {len(train_batches)} train batches, {len(val_batches)} val batches")

    # Initialize model and trainer
    model = ASTAutoencoder(
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        max_decode_steps=args.max_decode_steps
    )

    # Move model to device
    model.to(device)

    # Initialize trainer
    trainer = ASTAutoencoderTrainer(model, device=torch.device(device))

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume:
        if Path(args.resume).exists():
            log.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            log.info(f"Resumed from epoch {checkpoint['epoch']}")
        else:
            log.warning(f"Checkpoint not found: {args.resume}")

    # Setup artifacts directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run or f"run_{timestamp}"
    artifacts_path = Path(args.artifacts_dir) / run_name
    artifacts_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Artifacts will be saved to: {artifacts_path}")

    # Initialize WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "encoder_layers": args.encoder_layers,
                "max_decode_steps": args.max_decode_steps,
                "eval_interval": args.eval_interval,
                "log_interval": args.log_interval,
                "device": device,
                "data_dir": args.data_dir,
                "train_samples": len(train_graphs),
                "val_samples": len(val_graphs),
                "seed": args.seed,
            }
        )
        log.info("WandB initialized")

    # Save config
    config = vars(args)
    config.update({
        "device": device,
        "train_samples": len(train_graphs),
        "val_samples": len(val_graphs),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": timestamp,
    })

    with open(artifacts_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Starting training...")

    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        log.info(f"Epoch {epoch}/{args.epochs}")

        # Training phase
        train_metrics, global_step = train_one_epoch(
            model, trainer, optimizer, train_batches, device, 
            log_interval=args.log_interval, epoch=epoch, 
            use_wandb=use_wandb, global_step=global_step
        )

        if train_metrics:
            console.print(f"[green]✓[/green] Epoch {epoch} - Train loss: {train_metrics['train/loss']:.4f}, "
                         f"Similarity: {train_metrics['train/similarity_score']:.4f}, "
                         f"Exact match: {train_metrics['train/exact_match_rate']:.4f}")
        else:
            console.print(f"[red]⚠[/red] No successful training batches in epoch {epoch}")

        # Comprehensive validation
        val_metrics = {}
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = evaluate_model(model, trainer, val_batches, device)

            if val_metrics:
                console.print(f"[blue]ⓘ[/blue] Epoch {epoch} - Val loss: {val_metrics['val/loss']:.4f}, "
                              f"Compilation: {val_metrics['val/compilation_rate']:.3f}, "
                              f"One example: {val_metrics['val/one_example_rate']:.3f}, "
                              f"All examples: {val_metrics['val/all_examples_rate']:.3f}")

                # Save best model
                if val_metrics['val/loss'] < best_val_loss:
                    best_val_loss = val_metrics['val/loss']
                    best_model_path = artifacts_path / "best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'val_metrics': val_metrics,
                        'config': config,
                    }, best_model_path)
                    console.print(f"[green]🎯[/green] New best model saved! Val loss: {best_val_loss:.4f}")
            else:
                console.print(f"[red]⚠[/red] No successful validation batches in epoch {epoch}")

        # Log to WandB
        if use_wandb:
            wandb_metrics = {"epoch": epoch}
            if train_metrics:
                wandb_metrics = {**wandb_metrics, **train_metrics}
            if val_metrics:
                wandb_metrics = {**wandb_metrics, **val_metrics}
            wandb.log(wandb_metrics)

        # Save checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:  # Save every 5 epochs and at the end
            checkpoint_path = artifacts_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config,
            }, checkpoint_path)
            log.info(f"Saved checkpoint: {checkpoint_path}")

        # Save detailed validation results
        if val_metrics and epoch % 10 == 0:  # Save detailed results every 10 epochs
            val_results_path = artifacts_path / f"validation_epoch_{epoch}.json"
            with open(val_results_path, "w") as f:
                json.dump(val_metrics, f, indent=2)
            log.info(f"Saved validation results: {val_results_path}")

    # Final model save
    final_model_path = artifacts_path / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_epoch': args.epochs,
    }, final_model_path)

    # Training summary
    console.print("\n[bold green]🎉 Training completed![/bold green]")
    console.print(f"[green]✓[/green] Best validation loss: [bold]{best_val_loss:.4f}[/bold]")
    console.print(f"[green]✓[/green] Final model: [dim]{final_model_path}[/dim]")
    console.print(f"[green]✓[/green] Best model: [dim]{artifacts_path / 'best_model.pt'}[/dim]")
    console.print(f"[green]✓[/green] Artifacts: [dim]{artifacts_path}[/dim]")

    # Log final summary
    if use_wandb:
        wandb.log({
            "training_completed": True,
            "best_val_loss": best_val_loss,
            "total_epochs": args.epochs,
            "total_steps": global_step,
        })
        wandb.finish()


if __name__ == "__main__":
    main()