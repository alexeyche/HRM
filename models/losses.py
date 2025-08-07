from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float32), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class ProgramSynthesisLossHead(nn.Module):
    """Loss head for program synthesis with AST generation"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        # Forward through program synthesis model
        new_carry, outputs = self.model(**model_kwargs)

        # Extract AST targets from batch
        batch = model_kwargs.get('batch', {})
        ast_targets = batch.get('ast_targets', {})

        # Compute program synthesis losses
        targets = {'ast_targets': ast_targets}
        losses = self.model.compute_loss(outputs, targets)

        # Extract individual losses
        total_loss = losses.get('total_loss', torch.tensor(0.0))

        # Compute metrics
        with torch.no_grad():
            # Count valid examples (those with existing nodes)
            valid_examples = ast_targets.get('node_exists', torch.ones(1, 1, dtype=torch.bool)).any(dim=-1)
            count = valid_examples.sum()

            # Node existence accuracy
            node_exist_acc = 0.0
            if 'node_exists' in outputs and 'node_exists' in ast_targets:
                node_pred = torch.sigmoid(outputs['node_exists']) > 0.5
                node_exist_acc = (node_pred == ast_targets['node_exists']).float().mean()

            # Node type accuracy (only for existing nodes)
            node_type_acc = 0.0
            if 'node_types' in outputs and 'node_types' in ast_targets:
                mask = ast_targets['node_exists']
                if mask.any():
                    node_type_pred = outputs['node_types'].argmax(dim=-1)
                    correct = (node_type_pred[mask] == ast_targets['node_types'][mask])
                    node_type_acc = correct.float().mean()

            # Metrics dict
            metrics = {
                'count': count.float(),
                'node_exist_accuracy': (node_exist_acc * count).float(),
                'node_type_accuracy': (node_type_acc * count).float(),
            }

            # Add individual losses to metrics
            for loss_name, loss_value in losses.items():
                if loss_name != 'total_loss' and isinstance(loss_value, torch.Tensor):
                    metrics[loss_name] = loss_value.detach() * count

        # Q-learning metrics if available
        if 'q_halt_logits' in outputs and new_carry.halted is not None:
            # Simple Q-value metrics
            with torch.no_grad():
                valid_q = new_carry.halted & (count > 0)
                if valid_q.any():
                    metrics['q_halt_logits'] = outputs['q_halt_logits'][valid_q].mean().detach() * valid_q.sum()
                    metrics['steps'] = new_carry.steps[valid_q].float().sum()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # All finished when all sequences are halted (for ACT compatibility)
        all_finished = new_carry.halted.all() if hasattr(new_carry, 'halted') else torch.tensor(True)

        return new_carry, total_loss, metrics, detached_outputs, all_finished
