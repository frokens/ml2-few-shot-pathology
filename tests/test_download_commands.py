"""Tests for download command generation — dry-run safety and Virchow2 exclusion."""

from ml_final.weights.download_hf import (
    build_hf_download_command,
    resolve_endpoint,
    resolve_ignore_patterns,
)
from ml_final.weights.download_modelscope import (
    build_modelscope_command,
    build_modelscope_sdk_snippet,
)


class TestVirchow2Exclusion:
    """Virchow2 must prefer safetensors and exclude pytorch_model.bin by default."""

    def test_virchow2_excludes_pytorch_model_bin(self):
        ignore = resolve_ignore_patterns("virchow2")
        assert "pytorch_model.bin" in ignore, (
            "Virchow2 should exclude pytorch_model.bin by default"
        )

    def test_other_models_no_exclusion(self):
        ignore_uni = resolve_ignore_patterns("uni2_h")
        assert ignore_uni == [], "UNI2-h should not have automatic exclusions"

        ignore_conch = resolve_ignore_patterns("conch")
        assert ignore_conch == [], "CONCH should not have automatic exclusions"

        ignore_h_optimus = resolve_ignore_patterns("h_optimus_0")
        assert ignore_h_optimus == [], "H-optimus-0 should keep its official bin weight available"

    def test_virchow2_command_includes_exclude_flag(self):
        cmd = build_hf_download_command(
            repo_id="paige-ai/Virchow2",
            cache_dir="/tmp/cache",
            local_dir="/tmp/store/hf/paige-ai__Virchow2/main",
            endpoint="https://huggingface.co",
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["pytorch_model.bin"],
        )
        assert "paige-ai/Virchow2" in cmd
        assert "--exclude" in cmd
        assert "pytorch_model.bin" in cmd
        # We should NOT exclude safetensors
        assert "--include" in cmd
        assert "*.safetensors" in cmd


class TestBuildHFCommand:
    """Test HF download command construction."""

    def test_basic_command_structure(self):
        cmd = build_hf_download_command(
            repo_id="MahmoodLab/UNI2-h",
            cache_dir="/tmp/hf_cache",
            local_dir="/tmp/local_store/MahmoodLab__UNI2-h/main",
            endpoint="https://huggingface.co",
        )
        assert cmd[0] == "hf"
        assert cmd[1] == "download"
        assert "MahmoodLab/UNI2-h" in cmd
        assert "--cache-dir" in cmd
        assert "/tmp/hf_cache" in cmd
        assert "--local-dir" in cmd

    def test_command_with_allow_patterns(self):
        cmd = build_hf_download_command(
            repo_id="MahmoodLab/CONCH",
            cache_dir="/tmp/hf_cache",
            local_dir="/tmp/store",
            endpoint="https://huggingface.co",
            allow_patterns=["*.json", "*.safetensors"],
            ignore_patterns=["pytorch_model.bin"],
        )
        # Check include patterns
        includes = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--include"]
        assert "*.json" in includes
        assert "*.safetensors" in includes
        # Check exclude patterns
        excludes = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--exclude"]
        assert "pytorch_model.bin" in excludes

    def test_command_includes_necessary_files(self):
        """Generated commands should include config files, not just weights."""
        cmd = build_hf_download_command(
            repo_id="MahmoodLab/UNI2-h",
            cache_dir="/tmp/hf_cache",
            local_dir="/tmp/store",
            endpoint="https://huggingface.co",
            allow_patterns=["*.json", "*.py", "*.txt", "*.md", "*.safetensors", "*.bin"],
        )
        includes = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--include"]
        for needed in ["*.json", "*.safetensors"]:
            assert needed in includes, f"Expected --include {needed} in command"


class TestEndpointResolution:
    """Test source-to-endpoint mapping."""

    def test_official_endpoint(self):
        assert resolve_endpoint("official") == "https://huggingface.co"

    def test_hf_mirror_endpoint(self):
        assert resolve_endpoint("hf-mirror") == "https://hf-mirror.com"

    def test_unknown_source_defaults(self):
        assert resolve_endpoint("unknown") == "https://huggingface.co"


class TestModelScopeCommands:
    """Test ModelScope download command generation."""

    def test_modelscope_cli_command(self):
        cmd = build_modelscope_command(
            repo_id="some-org/some-model",
            local_dir="/tmp/modelscope_store/some_model",
            allow_patterns=["*.safetensors"],
        )
        assert "modelscope download" in cmd
        assert "some-org/some-model" in cmd

    def test_modelscope_sdk_snippet(self):
        snippet = build_modelscope_sdk_snippet(
            repo_id="some-org/some-model",
            local_dir="/tmp/modelscope_store",
        )
        assert "snapshot_download" in snippet
        assert "some-org/some-model" in snippet
