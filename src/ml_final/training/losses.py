"""Loss and batch mixing helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def cross_entropy_with_optional_smoothing(
    label_smoothing: float = 0.0,
    *,
    weight: torch.Tensor | None = None,
) -> nn.Module:
    """Return standard cross entropy with optional label smoothing."""

    return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(label_smoothing))


def maybe_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    probability: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float] | None]:
    """Apply batch-level MixUp when enabled."""

    if alpha <= 0 or probability <= 0:
        return images, None
    if torch.rand((), device=images.device).item() > probability:
        return images, None
    lam = torch.distributions.Beta(alpha, alpha).sample().to(images.device)
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, (labels, labels[perm], float(lam.item()))


def mixed_cross_entropy(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mixup_target: tuple[torch.Tensor, torch.Tensor, float] | None,
) -> torch.Tensor:
    """Compute CE for plain or MixUp batches."""

    if mixup_target is None:
        return criterion(logits, labels)
    y_a, y_b, lam = mixup_target
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy for probabilistic labels with optional per-sample weights."""

    log_probs = F.log_softmax(logits, dim=1)
    losses = -(soft_targets * log_probs).sum(dim=1)
    if sample_weight is not None:
        losses = losses * sample_weight
        denom = torch.clamp(sample_weight.sum(), min=1e-12)
        return losses.sum() / denom
    return losses.mean()


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float,
    label_smoothing: float = 0.0,
    class_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy with a focal factor for hard-label batches."""

    weight = class_weight.to(device=logits.device, dtype=logits.dtype) if class_weight is not None else None
    row_weight = sample_weight.to(device=logits.device, dtype=logits.dtype) if sample_weight is not None else None
    ce = F.cross_entropy(
        logits,
        labels,
        reduction="none",
        label_smoothing=float(label_smoothing),
        weight=weight,
    )
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, labels.reshape(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)
    losses = ce * torch.pow(1.0 - pt, float(gamma))
    if row_weight is not None:
        losses = losses * row_weight
    if weight is not None:
        denom_weight = weight.index_select(0, labels)
        if row_weight is not None:
            denom_weight = denom_weight * row_weight
        denom = torch.clamp(denom_weight.sum(), min=1e-12)
        return losses.sum() / denom
    if row_weight is not None:
        return losses.sum() / torch.clamp(row_weight.sum(), min=1e-12)
    return losses.mean()


def hard_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    class_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard-label CE with optional class and per-row weights."""

    weight = class_weight.to(device=logits.device, dtype=logits.dtype) if class_weight is not None else None
    row_weight = sample_weight.to(device=logits.device, dtype=logits.dtype) if sample_weight is not None else None
    losses = F.cross_entropy(
        logits,
        labels,
        label_smoothing=float(label_smoothing),
        weight=weight,
        reduction="none",
    )
    if row_weight is not None:
        losses = losses * row_weight
    if weight is not None:
        denom_weight = weight.index_select(0, labels)
        if row_weight is not None:
            denom_weight = denom_weight * row_weight
        return losses.sum() / torch.clamp(denom_weight.sum(), min=1e-12)
    if row_weight is not None:
        return losses.sum() / torch.clamp(row_weight.sum(), min=1e-12)
    return losses.mean()


def mixed_hard_soft_focal_loss(
    logits: torch.Tensor,
    hard_labels: torch.Tensor,
    soft_labels: torch.Tensor,
    sample_weight: torch.Tensor,
    is_pseudo: torch.Tensor,
    *,
    gamma: float,
    label_smoothing: float = 0.0,
    class_weight: torch.Tensor | None = None,
    pseudo_loss_scale: float = 1.0,
    true_soft_loss_scale: float = 0.0,
) -> torch.Tensor:
    """Compute focal CE for true rows and weighted focal soft CE for pseudo rows."""

    true_mask = ~is_pseudo.bool()
    pseudo_mask = is_pseudo.bool()
    losses = []
    if true_mask.any():
        true_logits = logits[true_mask]
        losses.append(
            focal_cross_entropy(
                true_logits,
                hard_labels[true_mask],
                gamma=gamma,
                label_smoothing=label_smoothing,
                class_weight=class_weight,
                sample_weight=sample_weight[true_mask],
            )
        )
        if true_soft_loss_scale > 0:
            losses.append(
                float(true_soft_loss_scale)
                * soft_cross_entropy(
                    true_logits,
                    soft_labels[true_mask],
                    sample_weight[true_mask].to(device=logits.device, dtype=logits.dtype),
                )
            )
    if pseudo_mask.any():
        log_probs = F.log_softmax(logits[pseudo_mask], dim=1)
        probs = log_probs.exp()
        soft = soft_labels[pseudo_mask]
        pseudo_losses = -(soft * log_probs).sum(dim=1)
        pt = (soft * probs).sum(dim=1).clamp(min=1e-6, max=1.0)
        weights = sample_weight[pseudo_mask].to(device=logits.device, dtype=logits.dtype)
        if class_weight is not None:
            resolved_weight = class_weight.to(device=logits.device, dtype=logits.dtype)
            weights = weights * (soft * resolved_weight.reshape(1, -1)).sum(dim=1)
        weights = weights * torch.pow(1.0 - pt, float(gamma))
        pseudo_loss = (pseudo_losses * weights).sum() / torch.clamp(weights.sum(), min=1e-12)
        losses.append(float(pseudo_loss_scale) * pseudo_loss)
    if not losses:
        return logits.sum() * 0.0
    return sum(losses)


def mixed_hard_soft_loss(
    logits: torch.Tensor,
    hard_labels: torch.Tensor,
    soft_labels: torch.Tensor,
    sample_weight: torch.Tensor,
    is_pseudo: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    class_weight: torch.Tensor | None = None,
    pseudo_loss_scale: float = 1.0,
    true_soft_loss_scale: float = 0.0,
) -> torch.Tensor:
    """Compute CE for true rows and weighted soft CE for pseudo rows."""

    true_mask = ~is_pseudo.bool()
    pseudo_mask = is_pseudo.bool()
    losses = []
    if true_mask.any():
        true_logits = logits[true_mask]
        losses.append(
            hard_cross_entropy(
                true_logits,
                hard_labels[true_mask],
                label_smoothing=float(label_smoothing),
                class_weight=class_weight,
                sample_weight=sample_weight[true_mask],
            )
        )
        if true_soft_loss_scale > 0:
            losses.append(
                float(true_soft_loss_scale)
                * soft_cross_entropy(
                    true_logits,
                    soft_labels[true_mask],
                    sample_weight[true_mask].to(device=logits.device, dtype=logits.dtype),
                )
            )
    if pseudo_mask.any():
        pseudo_sample_weight = sample_weight[pseudo_mask]
        if class_weight is not None:
            class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)
            pseudo_sample_weight = pseudo_sample_weight * (
                soft_labels[pseudo_mask] * class_weight.reshape(1, -1)
            ).sum(dim=1)
        pseudo_loss = soft_cross_entropy(
            logits[pseudo_mask],
            soft_labels[pseudo_mask],
            pseudo_sample_weight,
        )
        losses.append(float(pseudo_loss_scale) * pseudo_loss)
    if not losses:
        return logits.sum() * 0.0
    return sum(losses)
