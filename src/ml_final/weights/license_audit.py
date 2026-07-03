"""License audit report generation for model weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


LICENSE_TERMS: dict[str, str] = {
    "CC-BY-NC-ND-4.0": (
        "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0. "
        "Permits sharing with attribution; no commercial use; no derivatives. "
        "Suitable for academic/non-commercial research."
    ),
    "CC-BY-NC-SA-4.0": (
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0. "
        "Permits sharing and adaptation with attribution; no commercial use; "
        "derivatives must use the same license."
    ),
    "check_model_card_terms": (
        "Refer to the model card on the original hub for exact terms. "
        "This model may have custom or additional restrictions beyond "
        "standard open-source licenses."
    ),
}


def generate_license_audit(
    requested: dict[str, Any],
    lock: dict[str, Any] | None = None,
    output_path: str | Path = "artifacts/model_registry/license_audit.md",
) -> str:
    """Generate a license audit Markdown report.

    Args:
        requested: Parsed models.requested.yaml data.
        lock: Optional parsed models.lock.yaml data.
        output_path: Where to write the report.

    Returns:
        The report content as a string.
    """
    lines: list[str] = [
        "# Model License Audit",
        "",
        f"Generated: AUTO",
        "",
        "## Summary",
        "",
        "| Model | License | Gated | Usage Allowed | Terms Accepted |",
        "|-------|---------|-------|---------------|----------------|",
    ]

    for key, entry in requested.get("models", {}).items():
        license_id = entry.get("license", "unknown")
        gated = "Yes" if entry.get("gated") else "No"
        usage = entry.get("usage", "unspecified")
        accepted = (
            "Yes"
            if (lock and key in lock.get("models", {}) and lock["models"][key].get("terms_accepted"))
            else "Pending"
        )
        lines.append(
            f"| {key} | {license_id} | {gated} | {usage} | {accepted} |"
        )

    lines.extend([
        "",
        "## License Details",
        "",
    ])

    for key, entry in requested.get("models", {}).items():
        license_id = entry.get("license", "unknown")
        lines.append(f"### {key}")
        lines.append(f"- **License:** {license_id}")
        lines.append(f"- **Hub:** {entry.get('hub', 'unknown')}")
        lines.append(f"- **Repo:** {entry.get('repo_id', 'unknown')}")
        lines.append(f"- **Gated:** {'Yes' if entry.get('gated') else 'No'}")
        lines.append(f"- **Allowed usage:** {entry.get('usage', 'unspecified')}")

        explanation = LICENSE_TERMS.get(license_id, "See model card for details.")
        lines.append(f"- **Terms:** {explanation}")

        if lock and key in lock.get("models", {}):
            lk = lock["models"][key]
            lines.append(f"- **Downloaded:** {lk.get('downloaded_at', 'unknown')}")
            lines.append(f"- **Source:** {lk.get('source_endpoint', 'unknown')}")
            lines.append(f"- **Local path:** {lk.get('local_path', 'unknown')}")

        lines.append("")

    lines.extend([
        "## Important Notice",
        "",
        "- All required and optional foundation models are **gated** on Hugging Face.",
        "- You MUST accept the model terms on the Hugging Face website before downloading.",
        "- A Hugging Face token with read access to gated repos is required.",
        "- These models are for **academic research and course project use only**.",
        "- Commercial use is NOT permitted under these licenses.",
        "- Do not redistribute model weights. Share only the registry lockfile.",
        "",
        "## Model-Specific Notes",
        "",
        "### UNI2-h (MahmoodLab/UNI2-h)",
        "- CC-BY-NC-ND-4.0: No derivatives, no commercial use.",
        "- Feature extraction and optional LoRA is permitted under research terms.",
        "",
        "### Virchow2 (paige-ai/Virchow2)",
        "- Check model card on Hugging Face for exact terms.",
        "- Paige AI model: may have additional institutional restrictions.",
        "- Feature extraction and optional LoRA only where license permits.",
        "",
        "### CONCH (MahmoodLab/CONCH)",
        "- CC-BY-NC-ND-4.0: No derivatives, no commercial use.",
        "- Image encoder features and prompt/text encoder probing are permitted under research terms.",
        "- CONCH has its own GitHub repository with additional documentation.",
        "",
        "### H-optimus-0 (bioptimus/H-optimus-0)",
        "- Apache-2.0 model and code license.",
        "- Publicly listed gated repository; accept the contact-sharing conditions before download.",
        '- Official model card loader: `timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)`.',
        "- Project configs set `num_classes=0` where a feature vector backbone is required.",
    ])

    report = "\n".join(lines) + "\n"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info(f"License audit written to {output_path}")

    return report
