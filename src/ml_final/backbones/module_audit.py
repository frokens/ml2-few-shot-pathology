"""Module audit and LoRA target selection.

PEFT chooses LoRA targets by module names.  timm ViTs often use fused attention
layers such as ``attn.qkv`` instead of Hugging Face names like ``q_proj`` and
``v_proj``.  Scheme 02 therefore audits the concrete model before training and
persists the selected target modules in the run directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml_final.backbones.factory import count_parameters
from ml_final.utils.paths import project_relative


@dataclass(frozen=True)
class ModuleAudit:
    """Audit output for one model."""

    all_modules: list[dict[str, str]]
    linear_modules: list[dict[str, Any]]
    selected_lora_targets: list[str]
    parameter_counts: dict[str, int | float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_modules": self.all_modules,
            "linear_modules": self.linear_modules,
            "selected_lora_targets": self.selected_lora_targets,
            "parameter_counts": self.parameter_counts,
        }


def audit_modules(
    model: nn.Module,
    *,
    target_policy: str = "attention_qkv",
    explicit_targets: list[str] | None = None,
) -> ModuleAudit:
    """Audit module names and choose LoRA target modules."""

    all_modules = [
        {"name": name or "<root>", "type": module.__class__.__name__}
        for name, module in model.named_modules()
    ]
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                }
            )
    selected = explicit_targets or select_lora_targets(
        [item["name"] for item in linear_modules],
        policy=target_policy,
    )
    return ModuleAudit(
        all_modules=all_modules,
        linear_modules=linear_modules,
        selected_lora_targets=selected,
        parameter_counts=count_parameters(model),
    )


def select_lora_targets(linear_names: list[str], *, policy: str) -> list[str]:
    """Select LoRA target module names from audited Linear modules."""

    if policy == "attention_qkv":
        patterns = [".attn.qkv", ".attention.query", ".attention.key", ".attention.value"]
    elif policy == "attention_qkv_proj":
        patterns = [
            ".attn.qkv",
            ".attn.proj",
            ".attention.query",
            ".attention.key",
            ".attention.value",
            ".attention.output.dense",
        ]
    elif policy == "attention_mlp":
        patterns = [".attn.qkv", ".attn.proj", ".mlp.fc1", ".mlp.fc2"]
    elif policy == "all_linear_except_head":
        return [
            name
            for name in linear_names
            if "classifier" not in name and not name.endswith("head") and ".head." not in name
        ]
    elif policy == "none":
        return []
    else:
        raise ValueError(f"unsupported LoRA target policy: {policy}")

    targets = []
    for name in linear_names:
        if any(name.endswith(pattern) or pattern.strip(".") in name for pattern in patterns):
            if "classifier" not in name:
                targets.append(name)
    return sorted(set(targets))


def write_module_audit(audit: ModuleAudit, out_dir: Path, *, model_key: str) -> dict[str, str]:
    """Write module audit text and JSON artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    modules_txt = out_dir / f"{model_key}_modules.txt"
    linear_txt = out_dir / f"{model_key}_linear_modules.txt"
    targets_txt = out_dir / f"{model_key}_lora_targets.txt"
    audit_json = out_dir / f"{model_key}_module_audit.json"

    modules_txt.write_text(
        "\n".join(f"{item['name']}\t{item['type']}" for item in audit.all_modules) + "\n",
        encoding="utf-8",
    )
    linear_txt.write_text(
        "\n".join(
            f"{item['name']}\t{item['type']}\t{item['in_features']}->{item['out_features']}"
            for item in audit.linear_modules
        )
        + "\n",
        encoding="utf-8",
    )
    targets_txt.write_text("\n".join(audit.selected_lora_targets) + "\n", encoding="utf-8")
    audit_json.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n")
    return {
        "modules": project_relative(modules_txt),
        "linear_modules": project_relative(linear_txt),
        "targets": project_relative(targets_txt),
        "json": project_relative(audit_json),
    }


def assert_lora_targets_valid(model: nn.Module, targets: list[str]) -> None:
    """Fail early when selected targets do not correspond to Linear modules."""

    modules = dict(model.named_modules())
    missing = [name for name in targets if name not in modules]
    non_linear = [name for name in targets if name in modules and not isinstance(modules[name], nn.Linear)]
    if missing or non_linear:
        raise ValueError(
            "invalid LoRA target modules: "
            f"missing={missing}, non_linear={non_linear}. "
            "Run module audit and update target policy/config."
        )

