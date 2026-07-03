"""Safety guards for data-source isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


FORBIDDEN_REFERENCE_PATH_MARKERS = (
    "external_reference/",
    "external_reference/train.zip",
)


def assert_no_forbidden_reference_paths(payload: Any, *, context: str) -> None:
    """Reject configs that try to feed the external reference set into training."""

    for location, value in iter_string_values(payload):
        normalized = value.replace("\\", "/")
        if any(marker in normalized for marker in FORBIDDEN_REFERENCE_PATH_MARKERS):
            raise ValueError(
                f"{context} cannot use external reference data path `{value}` at `{location}`. "
                "The external reference set is only allowed in reference alignment."
            )


def iter_string_values(payload: Any, *, prefix: str = "$"):
    """Yield string-like values from nested config payloads."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from iter_string_values(value, prefix=f"{prefix}.{key}")
    elif isinstance(payload, (list, tuple)):
        for idx, value in enumerate(payload):
            yield from iter_string_values(value, prefix=f"{prefix}[{idx}]")
    elif isinstance(payload, (str, Path)):
        yield prefix, str(payload)
