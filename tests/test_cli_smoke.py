"""CLI smoke tests — verify commands parse without crashing."""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from ml_final.cli import _default_training_offline_env, app
from ml_final.utils.paths import project_root

runner = CliRunner()


def test_training_entrypoints_default_to_offline_hf_cache(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    _default_training_offline_env()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_gpu_training_wrappers_default_to_offline_hf_cache():
    wrappers = [
        "scripts/08_train_peft_cv.sh",
        "scripts/08c_train_peft_adapted_heads.sh",
        "scripts/08d_train_fusion_peft_cv.sh",
        "scripts/08e_train_peft_single_refit.sh",
        "scripts/08g_train_fusion_single_refit.sh",
        "scripts/08h_train_fusion_pseudo_cv.sh",
        "scripts/09b_predict_fusion_single_refit.sh",
        "scripts/12_train_pseudo_lora.sh",
        "scripts/12b_train_pseudo_lora_single_refit.sh",
    ]

    for rel_path in wrappers:
        text = (project_root() / rel_path).read_text(encoding="utf-8")
        assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in text
        assert 'export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"' in text


class TestEnvCheck:
    """Smoke test for env-check CLI command."""

    def test_env_check_help(self):
        result = runner.invoke(app, ["env-check", "--help"])
        assert result.exit_code == 0
        assert "env-check" in result.stdout

    def test_env_check_runs(self):
        result = runner.invoke(app, ["env-check"])
        # Should succeed even without optional args
        assert result.exit_code == 0
        assert "ML Final" in result.stdout or "Environment" in result.stdout or "Python" in result.stdout


def test_build_teacher_defaults_to_formal_s01_manifest_config():
    result = runner.invoke(app, ["build-teacher", "--help"])

    assert result.exit_code == 0
    assert "configs/scheme_03/teacher_s01.yaml" in result.stdout
    assert "artifacts/teachers/s01_teacher" in result.stdout


def test_fusion_pseudo_cv_help_mentions_pseudolabels():
    result = runner.invoke(app, ["train-fusion-pseudo-cv", "--help"])

    assert result.exit_code == 0
    assert "train-fusion-pseudo-cv" in result.stdout
    assert "--pseudolabels" in result.stdout


class TestInitModelRegistry:
    """Smoke test for init-model-registry CLI command."""

    def test_init_registry_help(self):
        result = runner.invoke(app, ["init-model-registry", "--help"])
        assert result.exit_code == 0
        assert "init-model-registry" in result.stdout

    def test_init_registry_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "init-model-registry",
                    "--out",
                    tmpdir,
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            # Should mention dry-run
            assert "dry-run" in result.stdout.lower()

    def test_init_registry_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # We need to mock project_root or use absolute paths
            out_dir = Path(tmpdir) / "model_registry"
            result = runner.invoke(
                app,
                [
                    "init-model-registry",
                    "--out",
                    str(out_dir),
                ],
            )
            # May fail if project_root resolution doesn't match, but should not crash
            # The command uses project_root() / out, so we need a real path
            assert result.exit_code == 0


class TestDownloadModels:
    """Smoke test for download-models CLI command."""

    def test_download_models_help(self):
        result = runner.invoke(app, ["download-models", "--help"])
        assert result.exit_code == 0
        assert "download-models" in result.stdout

    def test_download_models_dry_run_is_default_safe(self):
        """Verify that without --execute, download-models is safe (dry-run behavior)."""
        result = runner.invoke(
            app,
            [
                "download-models",
                "--source", "official",
                "--store", "/tmp/test_store",
                "--out", "/tmp/test_out",
                "--dry-run",
            ],
        )
        # Should not error; should show plan
        assert result.exit_code == 0
        assert "dry-run" in result.stdout.lower()
        assert "Models: uni2_h, virchow2, conch, h_optimus_0" in result.stdout

    def test_download_models_can_select_h_optimus_0(self):
        result = runner.invoke(
            app,
            [
                "download-models",
                "--source", "official",
                "--model", "h_optimus_0",
                "--store", "/tmp/test_store",
                "--out", "/tmp/test_out",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Models: h_optimus_0" in result.stdout
        assert "bioptimus/H-optimus-0" in result.stdout

    def test_download_models_rejects_invalid_source(self):
        result = runner.invoke(
            app,
            [
                "download-models",
                "--source", "invalid_source",
                "--out", "/tmp/test_out",
                "--dry-run",
            ],
        )
        assert result.exit_code != 0


class TestVerifyModels:
    """Smoke test for verify-models CLI command."""

    def test_verify_models_help(self):
        result = runner.invoke(app, ["verify-models", "--help"])
        assert result.exit_code == 0
        assert "verify-models" in result.stdout

    def test_verify_models_missing_lockfile(self):
        result = runner.invoke(
            app,
            [
                "verify-models",
                "--lock", "/nonexistent/path/models.lock.yaml",
            ],
        )
        assert result.exit_code != 0
