"""Tests for model registry YAML parsing and validation."""

import tempfile
from pathlib import Path

import pytest

from ml_final.weights.registry import (
    VALID_SOURCES,
    generate_lock_template,
    load_requested,
)


VALID_MODELS_YAML = """
models:
  uni2_h:
    hub: huggingface
    repo_id: MahmoodLab/UNI2-h
    required: true
    gated: true
    license: CC-BY-NC-ND-4.0
    usage: feature_extraction_and_optional_lora
    allow_patterns:
      - "*.safetensors"
      - "*.bin"
  virchow2:
    hub: huggingface
    repo_id: paige-ai/Virchow2
    required: true
    gated: true
    license: check_model_card_terms
    usage: feature_extraction_and_optional_lora
    allow_patterns:
      - "*.safetensors"
  conch:
    hub: huggingface
    repo_id: MahmoodLab/CONCH
    required: true
    gated: true
    license: CC-BY-NC-ND-4.0
    usage: image_encoder_features_and_prompt_encoder_probe
    allow_patterns:
      - "*.safetensors"
  h_optimus_0:
    hub: huggingface
    repo_id: bioptimus/H-optimus-0
    required: true
    gated: true
    license: Apache-2.0
    usage: feature_extraction_and_optional_lora
    official_loader:
      library: timm
      model_name: hf-hub:bioptimus/H-optimus-0
      input_size: 224
      feature_dim: 1536
      model_kwargs:
        num_classes: 0
        init_values: 1.0e-5
        dynamic_img_size: false
      mean: [0.707223, 0.578729, 0.703617]
      std: [0.211883, 0.230117, 0.177517]
    allow_patterns:
      - "*.safetensors"
      - "*.bin"
"""

MISSING_FIELD_YAML = """
models:
  uni2_h:
    hub: huggingface
    repo_id: MahmoodLab/UNI2-h
    required: true
    gated: true
    # missing license and usage
"""

NO_MODELS_KEY_YAML = """
not_models:
  something: else
"""


class TestRegistryLoadRequested:
    """Test loading models.requested.yaml."""

    def test_loads_valid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(VALID_MODELS_YAML)
            f.flush()
            path = f.name

        try:
            doc = load_requested(path)
            assert "models" in doc
            assert "uni2_h" in doc["models"]
            assert "virchow2" in doc["models"]
            assert "conch" in doc["models"]
            assert "h_optimus_0" in doc["models"]
            assert doc["models"]["uni2_h"]["repo_id"] == "MahmoodLab/UNI2-h"
            assert doc["models"]["virchow2"]["license"] == "check_model_card_terms"
            assert doc["models"]["h_optimus_0"]["required"] is True
        finally:
            Path(path).unlink()

    def test_raises_on_missing_required_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(MISSING_FIELD_YAML)
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="missing required fields"):
                load_requested(path)
        finally:
            Path(path).unlink()

    def test_raises_on_no_models_key(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(NO_MODELS_KEY_YAML)
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="models"):
                load_requested(path)
        finally:
            Path(path).unlink()

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_requested("/nonexistent/path/models.requested.yaml")

    def test_required_models_present(self):
        """Acceptance: registry contains the four required backbone weights."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(VALID_MODELS_YAML)
            f.flush()
            path = f.name

        try:
            doc = load_requested(path)
            model_keys = set(doc["models"].keys())
            assert model_keys == {"uni2_h", "virchow2", "conch", "h_optimus_0"}
            required_keys = {k for k, v in doc["models"].items() if v["required"]}
            assert required_keys == {"uni2_h", "virchow2", "conch", "h_optimus_0"}
        finally:
            Path(path).unlink()

    def test_h_optimus_0_official_loader_metadata(self):
        """H-optimus-0 follows the official timm model-card loading recipe."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(VALID_MODELS_YAML)
            f.flush()
            path = f.name

        try:
            doc = load_requested(path)
            loader = doc["models"]["h_optimus_0"]["official_loader"]
            assert loader["model_name"] == "hf-hub:bioptimus/H-optimus-0"
            assert loader["input_size"] == 224
            assert loader["feature_dim"] == 1536
            assert loader["model_kwargs"] == {
                "num_classes": 0,
                "init_values": 1.0e-5,
                "dynamic_img_size": False,
            }
            assert loader["mean"] == [0.707223, 0.578729, 0.703617]
            assert loader["std"] == [0.211883, 0.230117, 0.177517]
        finally:
            Path(path).unlink()


class TestGenerateLockTemplate:
    """Test lockfile template generation."""

    def test_generates_lock_for_all_models(self):
        requested = {
            "models": {
                "uni2_h": {
                    "hub": "huggingface",
                    "repo_id": "MahmoodLab/UNI2-h",
                    "required": True,
                    "gated": True,
                    "license": "CC-BY-NC-ND-4.0",
                    "usage": "features",
                },
                "conch": {
                    "hub": "huggingface",
                    "repo_id": "MahmoodLab/CONCH",
                    "required": True,
                    "gated": True,
                    "license": "CC-BY-NC-ND-4.0",
                    "usage": "features",
                },
            }
        }
        lock = generate_lock_template(requested, source="official", store="/tmp/test_store")
        assert "models" in lock
        assert "uni2_h" in lock["models"]
        assert "conch" in lock["models"]
        assert lock["models"]["uni2_h"]["repo_id"] == "MahmoodLab/UNI2-h"
        assert "revision" in lock["models"]["uni2_h"]
        assert "local_path" in lock["models"]["uni2_h"]

    def test_source_maps_to_endpoint(self):
        requested = {
            "models": {
                "uni2_h": {
                    "hub": "huggingface",
                    "repo_id": "MahmoodLab/UNI2-h",
                    "required": True,
                    "gated": True,
                    "license": "CC-BY-NC-ND-4.0",
                    "usage": "features",
                },
            }
        }
        lock_official = generate_lock_template(
            requested, source="official", store="/data/store"
        )
        assert "huggingface.co" in lock_official["models"]["uni2_h"]["source_endpoint"]

        lock_mirror = generate_lock_template(
            requested, source="hf-mirror", store="/data/store"
        )
        assert "hf-mirror.com" in lock_mirror["models"]["uni2_h"]["source_endpoint"]

    def test_valid_sources_defined(self):
        """Ensure the VALID_SOURCES constant is correct."""
        assert "official" in VALID_SOURCES
        assert "hf-mirror" in VALID_SOURCES
        assert "modelscope" in VALID_SOURCES
        assert "offline" in VALID_SOURCES
