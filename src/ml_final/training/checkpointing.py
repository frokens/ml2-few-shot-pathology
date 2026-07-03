"""Checkpoint and EMA utilities."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class CheckpointState:
    """Training state persisted for resume."""

    epoch: int
    best_macro_f1: float
    best_balanced_accuracy: float
    history: list[dict[str, Any]]


class ModelEma:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.cpu() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], *, device: torch.device) -> None:
        self.shadow = {name: tensor.to(device) for name, tensor in state.items()}

    @contextmanager
    def apply_to(self, model: nn.Module):
        """Temporarily swap trainable parameters to their EMA values."""

        backup: dict[str, torch.Tensor] = {}
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, shadow in self.shadow.items():
                parameter = parameters.get(name)
                if parameter is None:
                    continue
                backup[name] = parameter.detach().clone()
                parameter.copy_(shadow.to(parameter.device, dtype=parameter.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, tensor in backup.items():
                    parameter = parameters.get(name)
                    if parameter is not None:
                        parameter.copy_(tensor)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    state: CheckpointState,
    config: dict[str, Any],
    ema: ModelEma | None = None,
    model_scope: str = "full",
    save_optimizer: bool = True,
    save_rng: bool = True,
) -> None:
    """Save a training checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": state.epoch,
        "best_macro_f1": state.best_macro_f1,
        "best_balanced_accuracy": state.best_balanced_accuracy,
        "history": state.history,
        "model_scope": model_scope,
        "model": build_model_state_dict(model, scope=model_scope),
        "optimizer": optimizer.state_dict() if save_optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "ema": ema.state_dict() if ema is not None else None,
        "rng_state": build_rng_state() if save_rng else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_model_state_dict(model: nn.Module, *, scope: str) -> dict[str, torch.Tensor]:
    """Return either full model state or only trainable parameter tensors."""

    if scope == "full":
        return model.state_dict()
    if scope != "trainable":
        raise ValueError(f"unsupported checkpoint model_scope: {scope}")
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = model.state_dict()
    return {
        name: tensor
        for name, tensor in state.items()
        if name in trainable_names
    }


def build_rng_state() -> dict[str, Any]:
    """Return Python, NumPy, PyTorch, and CUDA RNG state."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ModelEma | None = None,
    device: torch.device,
    model_strict: bool | None = None,
) -> CheckpointState:
    """Load model and optimizer state for resume."""

    payload = torch.load(path, map_location=device, weights_only=False)
    strict = model_strict
    if strict is None:
        strict = str(payload.get("model_scope", "full")) == "full"
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"], device=device)
    rng_state = payload.get("rng_state") or {}
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    return CheckpointState(
        epoch=int(payload.get("epoch", 0)),
        best_macro_f1=float(payload.get("best_macro_f1", -1.0)),
        best_balanced_accuracy=float(payload.get("best_balanced_accuracy", -1.0)),
        history=list(payload.get("history", [])),
    )
