"""Tests for offline verification behavior."""

import tempfile
from pathlib import Path

from ml_final.weights.checksum import (
    generate_checksums,
    sha256_file,
    verify_checksums,
)


class TestSHA256:
    """Test SHA256 checksum computation."""

    def test_sha256_file_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()
            path = f.name

        try:
            digest1 = sha256_file(path)
            digest2 = sha256_file(path)
            assert digest1 == digest2
            assert len(digest1) == 64  # SHA256 hex is 64 chars
        finally:
            Path(path).unlink()

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f1:
            f1.write("content a")
            f1.flush()
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:
            f2.write("content b")
            f2.flush()
            path2 = f2.name

        try:
            assert sha256_file(path1) != sha256_file(path2)
        finally:
            Path(path1).unlink()
            Path(path2).unlink()


class TestOfflineVerify:
    """Test offline verification — must fail clearly when files are absent."""

    def test_verify_fails_when_directory_missing(self):
        """Offline verify should clearly fail when model directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a checksum file that references non-existent files
            sums_path = Path(tmpdir) / "SHA256SUMS"
            sums_path.write_text(
                "abc123def456   nonexistent_file.safetensors\n"
            )
            ok, errors = verify_checksums(tmpdir, sums_path)
            assert not ok
            assert len(errors) > 0
            assert any("Missing" in e for e in errors)

    def test_verify_succeeds_when_files_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_text("hello world")

            # Generate checksums
            sums_path = Path(tmpdir) / "SHA256SUMS"
            generate_checksums(tmpdir, sums_path)

            # Verify (should pass — files just created)
            ok, errors = verify_checksums(tmpdir, sums_path)
            assert ok, f"Verification failed: {errors}"

    def test_verify_fails_when_file_modified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file and generate checksums
            test_file = Path(tmpdir) / "model.bin"
            test_file.write_text("original content")
            sums_path = Path(tmpdir) / "SHA256SUMS"
            generate_checksums(tmpdir, sums_path)

            # Modify the file
            test_file.write_text("modified content!!!")

            # Verify should now fail
            ok, errors = verify_checksums(tmpdir, sums_path)
            assert not ok
            assert any("mismatch" in e.lower() for e in errors)

    def test_verify_fails_when_checksum_file_missing(self):
        ok, errors = verify_checksums("/tmp", "/nonexistent/SHA256SUMS")
        assert not ok
        assert any("not found" in e for e in errors)
