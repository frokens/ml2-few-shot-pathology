"""SHA256 checksum generation and verification for model weights."""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger


def sha256_file(path: str | Path) -> str:
    """Compute SHA256 hex digest for a single file."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_checksums(
    directory: str | Path,
    output_path: str | Path,
    glob_pattern: str = "*",
) -> dict[str, str]:
    """Generate SHA256SUMS for all files in a directory.

    Args:
        directory: Directory to scan.
        output_path: Path to write SHA256SUMS file.
        glob_pattern: File pattern to match (default '*').

    Returns:
        Dict mapping relative filename to SHA256 hex digest.
    """
    directory = Path(directory)
    output_path = Path(output_path)
    checksums: dict[str, str] = {}

    files = sorted(directory.rglob(glob_pattern))
    files = [f for f in files if f.is_file()]

    lines = []
    for fpath in files:
        digest = sha256_file(fpath)
        relative = fpath.relative_to(directory)
        checksums[str(relative)] = digest
        lines.append(f"{digest}  {relative}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {len(lines)} checksums to {output_path}")

    return checksums


def verify_checksums(
    directory: str | Path,
    checksum_path: str | Path,
) -> tuple[bool, list[str]]:
    """Verify files against a SHA256SUMS file.

    Returns:
        (ok: bool, errors: list[str]) — ok is True if all files match.
    """
    directory = Path(directory)
    checksum_path = Path(checksum_path)
    errors: list[str] = []

    if not checksum_path.exists():
        return False, [f"Checksum file not found: {checksum_path}"]

    expected: dict[str, str] = {}
    for line in checksum_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        digest, _, relpath = line.partition("  ")
        expected[relpath.strip()] = digest

    for relpath, expected_digest in expected.items():
        fpath = directory / relpath
        if not fpath.exists():
            errors.append(f"Missing file: {relpath}")
            continue
        actual_digest = sha256_file(fpath)
        if actual_digest != expected_digest:
            errors.append(
                f"Checksum mismatch: {relpath} "
                f"(expected {expected_digest[:12]}..., got {actual_digest[:12]}...)"
            )

    # Check for unexpected files? Not required for verification, just report.

    ok = len(errors) == 0
    return ok, errors
