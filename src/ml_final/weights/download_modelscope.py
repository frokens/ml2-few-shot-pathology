"""Generate ModelScope download commands."""

from __future__ import annotations


def build_modelscope_command(
    repo_id: str,
    local_dir: str,
    allow_patterns: list[str] | None = None,
) -> str:
    """Build a ModelScope download command string.

    Uses the CLI: `modelscope download --model <repo_id> --local_dir <path>`.

    If allow_patterns is provided, a note is appended — ModelScope CLI
    does not natively support glob includes, so post-download filtering
    may be needed.

    Returns:
        A shell command string for display or execution.
    """
    parts = ["modelscope", "download", "--model", repo_id, "--local_dir", local_dir]
    cmd = " ".join(parts)

    if allow_patterns:
        cmd += (
            "\n# Note: ModelScope CLI does not support --include patterns. "
            "The following patterns are desired:\n"
        )
        for pat in allow_patterns:
            cmd += f"#   {pat}\n"

    return cmd


def build_modelscope_sdk_snippet(
    repo_id: str,
    local_dir: str,
) -> str:
    """Return a Python SDK snippet for ModelScope download."""
    return (
        f"from modelscope.hub.snapshot_download import snapshot_download\n"
        f'snapshot_download("{repo_id}", local_dir="{local_dir}")'
    )
