"""Generate and optionally execute Hugging Face download commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def build_hf_download_command(
    repo_id: str,
    cache_dir: str,
    local_dir: str,
    endpoint: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> list[str]:
    """Build a `hf download` command-line.

    Args:
        repo_id: e.g. "MahmoodLab/UNI2-h".
        cache_dir: HF cache directory.
        local_dir: Destination directory for symlinked / copied files.
        endpoint: HF_ENDPOINT value (https://huggingface.co or https://hf-mirror.com).
        allow_patterns: File patterns to include.
        ignore_patterns: File patterns to exclude.

    Returns:
        A list of shell command tokens.
    """
    cmd = ["hf", "download", repo_id]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    cmd += ["--local-dir", local_dir]

    if allow_patterns:
        for pat in allow_patterns:
            cmd += ["--include", pat]

    if ignore_patterns:
        for pat in ignore_patterns:
            cmd += ["--exclude", pat]

    return cmd


def resolve_ignore_patterns(
    model_key: str,
    allow_patterns: list[str] | None = None,
) -> list[str]:
    """Determine which patterns to ignore for a given model.

    Safety rules:
    - For Virchow2, never download both model.safetensors AND pytorch_model.bin.
      Prefer safetensors by default; exclude pytorch_model.bin.
    """
    ignore: list[str] = []

    if model_key == "virchow2":
        # Virchow2 must prefer safetensors — exclude all .bin weights
        ignore.append("pytorch_model.bin")
        ignore.append("*.bin")
        # If safetensors not explicitly in allow_patterns, warn
        if allow_patterns and "*.safetensors" not in allow_patterns:
            logger.warning(
                "Virchow2: safetensors not in allow_patterns. "
                "Will still exclude pytorch_model.bin and *.bin for safety."
            )

    return ignore


def resolve_endpoint(source: str) -> str:
    """Map a source name to the HF_ENDPOINT URL."""
    mapping = {
        "official": "https://huggingface.co",
        "hf-mirror": "https://hf-mirror.com",
    }
    return mapping.get(source, "https://huggingface.co")


def dry_run_command(env_vars: dict[str, str], cmd: list[str]) -> str:
    """Render a shell command string for display (dry-run)."""
    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
    cmd_str = " ".join(cmd)
    return f"{env_str} {cmd_str}"
