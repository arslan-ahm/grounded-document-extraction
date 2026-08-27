"""One training loop for both heads.

There is a single loop, and ``model.head`` is the only thing that differs between
the method and its reference baseline. Separate scripts per arm are how an
incidental difference in schedule, warm-up or checkpoint selection gets reported
as a difference in method, so the arrangement here is deliberate: the arms cannot
diverge in anything but the head, because there is nowhere for them to diverge.

Checkpoint selection is on **validation loss**, not on a validation metric. The
two heads' natural metrics are not comparable -- span exactness versus character
accuracy -- so selecting on a metric would apply a different rule to each arm.
Validation loss is each head's own objective, evaluated on held-out data, and is
the only selection criterion that is the same *rule* for both.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gdx.config import Config
from gdx.data.dataset import DocumentDataset
from gdx.models.model import GDXModel
from gdx.utils.logging import JsonlLogger, get_logger

LOG = get_logger(__name__)


@dataclass
class TrainResult:
    """Outcome of one training run."""

    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    train_seconds: float = 0.0
    n_params: int = 0
    steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_seconds": self.train_seconds,
            "n_params": self.n_params,
            "steps": self.steps,
            "n_epochs": len(self.history),
        }


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Linear warm-up then cosine decay to 10% of ``base_lr``.

    The floor at 10% rather than 0 matters at this budget: with six epochs the
    last epoch would otherwise run at an effectively zero learning rate and
    contribute nothing but wall-clock.
    """
    if total <= 0:
        return base_lr
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate_loss(model: GDXModel, data: DocumentDataset, batch_size: int) -> float:
    """Mean loss over ``data``, weighted by batch size.

    Weighting matters because the last batch is usually short; an unweighted mean
    over batches would let it count as much as a full one.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        for batch in data.batches(batch_size, shuffle=False):
            loss, _ = model.loss(batch)
            total += float(loss.detach()) * len(batch)
            count += len(batch)
    finally:
        model.train(was_training)
    return total / count if count else float("nan")


def train(
    model: GDXModel,
    train_data: DocumentDataset,
    val_data: DocumentDataset,
    cfg: Config,
    run_dir: Path | None = None,
) -> TrainResult:
    """Train ``model`` and return its history, restoring the best checkpoint.

    Args:
        model: The model, modified in place. On return it holds the parameters
            from the epoch with the lowest validation loss.
        train_data: Training split.
        val_data: Validation split.
        cfg: Full configuration; ``cfg.optim`` drives the schedule.
        run_dir: If given, ``history.jsonl`` is appended there as training
            proceeds, so an interrupted run still leaves a readable partial log.

    Returns:
        A :class:`TrainResult`.
    """
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    n_batches = max(1, math.ceil(len(train_data) / cfg.optim.batch_size))
    total_steps = n_batches * cfg.optim.epochs
    warmup = int(cfg.optim.warmup_frac * total_steps)
    logger = JsonlLogger(run_dir / "history.jsonl") if run_dir is not None else None

    result = TrainResult(n_params=sum(p.numel() for p in model.parameters()))
    best_state: dict[str, torch.Tensor] | None = None
    step = 0
    started = time.perf_counter()

    for epoch in range(cfg.optim.epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch in train_data.batches(
            cfg.optim.batch_size, shuffle=True, seed=cfg.run.seed * 1000 + epoch
        ):
            lr = cosine_lr(step, total_steps, warmup, cfg.optim.lr)
            for group in optim.param_groups:
                group["lr"] = lr
            loss, _ = model.loss(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optim.step()
            epoch_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
            step += 1

        val_loss = evaluate_loss(model, val_data, cfg.optim.batch_size)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, seen),
            "val_loss": val_loss,
            "lr": lr,
            "seconds": time.perf_counter() - started,
        }
        result.history.append(record)
        if logger is not None:
            logger.log(**record)
        LOG.info(
            "epoch %d/%d train %.4f val %.4f (%.1fs)",
            epoch + 1,
            cfg.optim.epochs,
            record["train_loss"],
            val_loss,
            record["seconds"],
        )
        if val_loss == val_loss and val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    result.train_seconds = time.perf_counter() - started
    result.steps = step
    return result


def save_checkpoint(model: GDXModel, path: Path) -> Path:
    """Write model weights plus the config needed to rebuild the module.

    Checkpoints are excluded from git (they are large and derived), but they are
    written because the inference-time ablations -- verification on/off,
    arithmetic on/off, abstention on/off -- must reuse the *same weights*.
    Retraining for each would confound the ablation with initialisation noise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "head": model.cfg.head, "config": vars(model.cfg)},
        path,
    )
    return path


def load_checkpoint(model: GDXModel, path: Path) -> GDXModel:
    """Load weights into ``model``, checking the head matches.

    Raises:
        ValueError: If the checkpoint was trained with a different head, which
            would otherwise fail as a confusing shape error much later.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("head") != model.cfg.head:
        raise ValueError(
            f"checkpoint head {payload.get('head')!r} != model head {model.cfg.head!r}"
        )
    model.load_state_dict(payload["state_dict"])
    return model
