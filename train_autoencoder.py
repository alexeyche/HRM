from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import logging
import warnings

import wandb

from dataset.ast import ASTSimplifier
from dataset.encoding import GraphEncoder, collate_to_pyg_batch
from dataset.grammar_actions import Action, ActionKind, actions_to_graph
from dataset.programs import get_program_registry
from models.autoencoder import ProgramAutoencoder
from models.generation import ProgramGenerator
from utils.functions import run_with_timeout

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def _load_samples(samples_jsonl: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with samples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _parse_action_kind(kind_str: str) -> ActionKind:
    # Accept values like "ActionKind.PROD_RETURN" or "PROD_RETURN" or just value "PROD_RETURN"
    if "." in kind_str:
        kind_str = kind_str.split(".")[-1]
    return ActionKind[kind_str]


def _actions_json_to_actions(actions_json: List[Dict[str, Any]]) -> List[Action]:
    actions: List[Action] = []
    for a in actions_json:
        kind = _parse_action_kind(a["kind"]) if isinstance(a["kind"], str) else ActionKind(a["kind"])  # type: ignore[arg-type]
        actions.append(Action(kind=kind, value=a.get("value")))
    return actions


def _build_graphs_from_codes(codes: List[str]) -> List[Any]:
    enc = GraphEncoder()
    graphs = [enc.encode(ASTSimplifier.ast_to_graph(code)) for code in codes]
    return graphs


def _prepare_batches(graphs: List[Any], batch_size: int) -> List[Any]:
    batches: List[Any] = []
    for i in range(0, len(graphs), batch_size):
        batches.append(collate_to_pyg_batch(graphs[i : i + batch_size]))
    return batches


def _prepare_action_batches(actions: List[List[Action]], batch_size: int) -> List[List[List[Action]]]:
    batches: List[List[List[Action]]] = []
    for i in range(0, len(actions), batch_size):
        batches.append(actions[i : i + batch_size])
    return batches


def _prepare_item_batches(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])
    return batches


def _prepare_teacher(ae: ProgramAutoencoder, actions_per_sample: List[List[Action]]) -> torch.Tensor:
    # map to ids
    seqs: List[List[int]] = []
    for acts in actions_per_sample:
        ids = [ae.action_to_id[a.kind] for a in acts]
        seqs.append(ids)
    return ae.prepare_teacher_tokens(seqs)


def _train_one_epoch(
    ae: ProgramAutoencoder,
    optimizer: torch.optim.Optimizer,
    train_batches: List[Any],
    train_actions: List[List[List[Action]]],
    device: str,
) -> float:
    ae.train()
    total_loss = 0.0
    count = 0
    for batch, acts in zip(train_batches, train_actions):
        batch = batch.to(device)
        action_ids = _prepare_teacher(ae, acts).to(device)
        out = ae.forward(batch, action_ids)
        logits = out["logits"]

        # labels: ignore BOS; also skip positions where all actions are invalid (would produce -inf logits only)
        labels = action_ids.clone()
        labels[labels == ae.num_actions] = -100

        valid_mask = ae.build_valid_action_masks(action_ids)  # [B, T, A]
        # Keep positions where label is not BOS and the labeled action is valid at that step
        # Gather per-position validity of the labeled action
        B, T = labels.shape
        label_idx = labels.clamp(min=0)  # replace -100 with 0 to allow gather
        per_pos_valid = valid_mask.gather(dim=-1, index=label_idx.unsqueeze(-1)).squeeze(-1)
        pos_mask = (labels != -100) & per_pos_valid

        if pos_mask.any():
            logits_sel = logits[pos_mask]
            labels_sel = labels[pos_mask]
            loss = F.cross_entropy(logits_sel, labels_sel.long())
        else:
            # Fallback: no valid positions in this batch; skip
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        count += 1
    return total_loss / max(1, count)




def _evaluate_and_write_outputs(val_items: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = get_program_registry()

    for idx, item in enumerate(val_items):
        program_name = item.get("program_name", f"prog_{idx}")
        actions_json = item["actions"]
        actions = _actions_json_to_actions(actions_json)

        # Reconstruct code from actions
        graph = actions_to_graph(actions)
        code = ASTSimplifier.ast_to_program(graph)

        # Execute on registry base examples
        spec = registry.get(program_name)
        if spec is None:
            # Skip if not found
            continue

        results: List[Dict[str, Any]] = []
        for ex in spec.base_examples:
            try:
                res = run_with_timeout(code, ex.input, timeout_s=0.5)
            except Exception as e:
                res = {"error": str(e)}
            results.append({"input": ex.input, "output": res})


        payload = {
            "code": code,
            "results": results,
        }
        # Write artifact
        out_path = out_dir / f"{program_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.inference_mode()
def _evaluate_model(
    ae: ProgramAutoencoder,
    val_batches: List[Any],
    val_actions: List[List[List[Action]]],
    device: str,
) -> Dict[str, float]:
    ae.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch, acts in zip(val_batches, val_actions):
        batch = batch.to(device)
        action_ids = _prepare_teacher(ae, acts).to(device)
        out = ae.forward(batch, action_ids)
        logits = out["logits"]

        labels = action_ids.clone()
        labels[labels == ae.num_actions] = -100

        valid_mask = ae.build_valid_action_masks(action_ids)  # [B, T, A]

        B, T = labels.shape
        label_idx = labels.clamp(min=0)
        per_pos_valid = valid_mask.gather(dim=-1, index=label_idx.unsqueeze(-1)).squeeze(-1)
        pos_mask = (labels != -100) & per_pos_valid

        if pos_mask.any():
            logits_sel = logits[pos_mask]
            labels_sel = labels[pos_mask]
            loss = F.cross_entropy(logits_sel, labels_sel.long())

            preds = logits.argmax(dim=-1)
            correct = (preds == labels) & pos_mask
            total_correct += int(correct.sum().item())
            total_count += int(pos_mask.sum().item())

            total_loss += float(loss.detach().cpu())

    steps = max(1, len(val_batches))
    avg_loss = total_loss / steps
    acc = (total_correct / max(1, total_count)) if total_count > 0 else 0.0
    return {"val/loss": avg_loss, "val/token_acc": acc}


@torch.inference_mode()
def _evaluate_generation_and_write_outputs(
    ae: ProgramAutoencoder,
    val_batches: List[Any],
    val_item_batches: List[List[Dict[str, Any]]],
    device: str,
    out_dir: Path,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = get_program_registry()

    total_examples = 0
    correct_examples = 0
    total_programs = 0
    passed_programs = 0

    # Build a generic ProgramGenerator configured with action/value vocab tokens
    ge = GraphEncoder()
    action_list = list(ActionKind)
    generator = ProgramGenerator(
        d_model=ae.token_emb.embedding_dim,
        num_heads=4,
        decoder_layers=2,
        action_list=action_list,
        op_tokens=list(ge.op_vocab._id_to_tok),
        fn_tokens=list(ge.function_vocab._id_to_tok),
        attr_tokens=list(ge.attribute_vocab._id_to_tok),
        small_int_tokens=list(ge.small_int_vocab._id_to_tok),
        max_var_vocab_size=ge.max_var_vocab_size,
    ).to(device)

    for batch, items in zip(val_batches, val_item_batches):
        batch = batch.to(device)
        # Encode graphs with AE to get context memory, then pass memory to generic generator
        _, ctx = ae._encode_graph(batch)
        gen_actions_batch = generator.generate(ctx, max_len=128)

        for idx_in_batch, (item, actions) in enumerate(zip(items, gen_actions_batch)):
            program_name = item.get("program_name", f"prog")

            # Build code from generated actions
            try:
                graph = actions_to_graph(actions)
                code = ASTSimplifier.ast_to_program(graph)
            except Exception as e:
                code = f"# generation_error: {e}\ndef program(n):\n    return 0"

            spec = registry.get(program_name)
            if spec is None:
                # Skip unknown program specs
                continue

            per_prog_total = 0
            per_prog_correct = 0
            results: List[Dict[str, Any]] = []
            for ex in spec.base_examples:
                per_prog_total += 1
                total_examples += 1
                try:
                    out = run_with_timeout(code, ex.input, timeout_s=0.5)
                    ok = (out == ex.output)
                    if ok:
                        correct_examples += 1
                        per_prog_correct += 1
                    results.append({"input": ex.input, "expected": ex.output, "output": out, "ok": ok})
                except Exception as e:
                    results.append({"input": ex.input, "expected": ex.output, "error": str(e), "ok": False})

            total_programs += 1
            if per_prog_total > 0 and per_prog_correct == per_prog_total:
                passed_programs += 1

            # Write artifact per program
            prog_dir = out_dir / program_name
            prog_dir.mkdir(parents=True, exist_ok=True)
            with (prog_dir / "gen.json").open("w", encoding="utf-8") as f:
                json.dump({"code": code, "results": results}, f, ensure_ascii=False, indent=2)

    example_pass_rate = (correct_examples / max(1, total_examples))
    program_full_pass_rate = (passed_programs / max(1, total_programs))
    return {"val/gen_example_pass_rate": example_pass_rate, "val/gen_program_full_pass_rate": program_full_pass_rate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train grammar-masked program autoencoder")
    parser.add_argument("--data_dir", type=str, default="data/programs-200", help="Directory containing samples.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--wandb_project", type=str, default="HRM-AE", help="wandb project name")
    parser.add_argument("--wandb_run", type=str, default=None, help="wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    samples_file = data_dir / "samples.jsonl"
    if not samples_file.exists():
        raise FileNotFoundError(f"{samples_file} not found. Generate with dataset/build_program_dataset.py")

    # Initialize wandb
    use_wandb = not args.no_wandb
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = args.wandb_project
    run_name = args.wandb_run or "run"
    if use_wandb:
        wandb.init(project=project_name, name=args.wandb_run, config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "data_dir": str(data_dir),
            "run_ts": run_ts,
        })
        # If user did not pass a run name, adopt wandb's generated name for foldering
        if args.wandb_run is None and wandb.run is not None:
            run_name = wandb.run.name or run_name

    # Create a single base output directory for this run
    base_out_dir = Path("outputs") / project_name / run_name / run_ts
    (base_out_dir / "eval").mkdir(parents=True, exist_ok=True)
    log.info(f"Run outputs will be stored under {base_out_dir}")

    items = _load_samples(samples_file)
    # Build graphs from code for each sample
    codes = [it["code"] for it in items]
    graphs = _build_graphs_from_codes(codes)

    # Build actions per sample
    all_actions = [_actions_json_to_actions(it["actions"]) for it in items]

    # Split 80/20
    N = len(items)
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    split = int(0.8 * N)
    train_idx, val_idx = idxs[:split], idxs[split:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_actions_list = [all_actions[i] for i in train_idx]
    val_actions_list = [all_actions[i] for i in val_idx]

    train_batches = _prepare_batches(train_graphs, args.batch_size)
    val_batches = _prepare_batches(val_graphs, args.batch_size)
    train_actions = _prepare_action_batches(train_actions_list, args.batch_size)
    val_actions = _prepare_action_batches(val_actions_list, args.batch_size)

    # Model
    ae = ProgramAutoencoder(d_model=128, num_gnn_layers=2, decoder_layers=2, num_heads=4)
    # Select device
    requested = args.device
    device = "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif requested == "mps" and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = "mps"
    elif requested != "cpu":
        warnings.warn(f"Requested device '{requested}' not available. Falling back to CPU.")
        device = "cpu"

    ae.to(device)

    optimizer = torch.optim.Adam(
        [p for p in ae.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Train + Eval per epoch
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(ae, optimizer, train_batches, train_actions, device)
        log.info(f"epoch {epoch} train_loss={train_loss:.4f}")

        # Validation
        metrics = _evaluate_model(ae, val_batches, val_actions, device)
        log.info(f"epoch {epoch} val_loss={metrics['val/loss']:.4f} token_acc={metrics['val/token_acc']:.4f}")

        if use_wandb:
            wandb.log({"train/loss": train_loss, **metrics, "epoch": epoch})

        # Generation-based evaluation and artifacts once per epoch
        out_dir = base_out_dir / "eval" / f"epoch_{epoch:03d}"
        val_items = [items[i] for i in val_idx]
        val_item_batches = _prepare_item_batches(val_items, args.batch_size)
        log.info(f"Evaluating generation on {len(val_items)} programs")
        gen_metrics = _evaluate_generation_and_write_outputs(ae, val_batches, val_item_batches, device, out_dir)
        log.info(
            f"epoch {epoch} gen_example_pass_rate={gen_metrics['val/gen_example_pass_rate']:.4f} "
            f"gen_program_full_pass_rate={gen_metrics['val/gen_program_full_pass_rate']:.4f}"
        )
        if use_wandb:
            wandb.log({**gen_metrics, "epoch": epoch})


if __name__ == "__main__":
    main()


