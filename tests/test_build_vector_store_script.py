"""Tests for build_vector_store.py helper behavior."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from scripts import build_vector_store


class TestBuildVectorStoreScript(unittest.TestCase):
    """Tests for vector build progress and GPU preflight helpers."""

    def test_format_progress_line_includes_percent_rate_and_eta(self) -> None:
        """Progress lines should include useful progress details."""

        line = build_vector_store.format_progress_line(
            processed_count=25,
            total_count=100,
            chunks_added=25,
            elapsed_seconds=60,
        )

        self.assertIn("[vector-build] 25/100 chunks added (25.00%)", line)
        self.assertIn("elapsed 00:01:00", line)
        self.assertIn("rate 25.0 chunks/min", line)
        self.assertIn("ETA 00:03:00", line)

    def test_format_progress_line_handles_zero_total_safely(self) -> None:
        """Progress formatting should not divide by zero."""

        line = build_vector_store.format_progress_line(
            processed_count=0,
            total_count=0,
            chunks_added=0,
            elapsed_seconds=0,
        )

        self.assertIn("0/0 chunks added (0.00%)", line)
        self.assertIn("ETA unknown", line)

    def test_parse_args_accepts_progress_every(self) -> None:
        """CLI parsing should accept the progress interval option."""

        args = build_vector_store.parse_args(["--progress-every", "10"])

        self.assertEqual(args.progress_every, 10)

    def test_parse_args_accepts_batch_size(self) -> None:
        """CLI parsing should accept the Chroma batch size option."""

        args = build_vector_store.parse_args(["--batch-size", "25"])

        self.assertEqual(args.batch_size, 25)

    def test_parse_args_accepts_shard_count(self) -> None:
        """CLI parsing should accept the Chroma shard count option."""

        args = build_vector_store.parse_args(["--shard-count", "10"])

        self.assertEqual(args.shard_count, 10)

    def test_parse_args_accepts_reset_chroma_dir(self) -> None:
        """CLI parsing should accept the Chroma directory reset option."""

        args = build_vector_store.parse_args(["--reset-chroma-dir"])

        self.assertTrue(args.reset_chroma_dir)

    def test_parse_args_accepts_post_build_health_check_flags(self) -> None:
        """CLI parsing should support enabling and disabling health checks."""

        enabled_args = build_vector_store.parse_args(["--post-build-health-check"])
        disabled_args = build_vector_store.parse_args(["--skip-post-build-health-check"])

        self.assertTrue(enabled_args.post_build_health_check)
        self.assertFalse(disabled_args.post_build_health_check)

    def test_parse_args_accepts_post_build_settle_seconds(self) -> None:
        """CLI parsing should accept the post-build settle delay."""

        args = build_vector_store.parse_args(["--post-build-settle-seconds", "1.5"])

        self.assertEqual(args.post_build_settle_seconds, 1.5)

    def test_print_vector_build_start_includes_shard_count(self) -> None:
        """Build start output should include the configured shard count."""

        output = io.StringIO()
        with redirect_stdout(output):
            build_vector_store.print_vector_build_start(
                selected_count=100,
                progress_every=25,
                db_path=Path("metadata.sqlite"),
                batch_size=50,
                shard_count=10,
                post_build_health_check=True,
                reset_chroma_dir=False,
            )

        self.assertIn("Shard count: 10", output.getvalue())

    def test_health_check_subprocess_helper_returns_success(self) -> None:
        """Post-build health check should return True on exit code zero."""

        calls: list[list[str]] = []

        def runner(command: list[str], timeout: int) -> tuple[bool, str]:
            calls.append(command)
            self.assertEqual(timeout, 120)
            return True, "health ok"

        output = io.StringIO()
        with redirect_stdout(output):
            result = build_vector_store.run_post_build_health_check(
                query="Who is Albert Einstein?",
                entity="Albert Einstein",
                entity_type="person",
                command_runner=runner,
            )

        self.assertTrue(result)
        self.assertIn("health ok", output.getvalue())
        self.assertIn("--query", calls[0])
        self.assertIn("--entity", calls[0])
        self.assertIn("--shard-count", calls[0])

    def test_health_check_subprocess_helper_returns_failure(self) -> None:
        """Post-build health check should return False on nonzero exit."""

        def runner(command: list[str], timeout: int) -> tuple[bool, str]:
            del command, timeout
            return False, "query failed"

        output = io.StringIO()
        with redirect_stdout(output):
            result = build_vector_store.run_post_build_health_check(
                query="Who is Albert Einstein?",
                command_runner=runner,
            )

        self.assertFalse(result)
        self.assertIn("query failed", output.getvalue())

    def test_reset_chroma_dir_deletes_only_chroma_directory(self) -> None:
        """reset_chroma_directory should remove Chroma data but leave SQLite data."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chroma_dir = root / "chroma_db"
            sqlite_dir = root / "data" / "sqlite"
            chroma_dir.mkdir()
            sqlite_dir.mkdir(parents=True)
            (chroma_dir / "index.bin").write_text("fake", encoding="utf-8")
            sqlite_file = sqlite_dir / "rag_metadata.sqlite"
            sqlite_file.write_text("sqlite", encoding="utf-8")

            removed = build_vector_store.reset_chroma_directory(chroma_dir)

            self.assertTrue(removed)
            self.assertFalse(chroma_dir.exists())
            self.assertTrue(sqlite_file.exists())

    def test_gpu_preflight_handles_nvidia_smi_success(self) -> None:
        """GPU preflight should pass when NVIDIA and Ollama checks pass."""

        def runner(command: list[str], timeout: int) -> tuple[bool, str]:
            del timeout
            if command == ["nvidia-smi"]:
                return True, "NVIDIA-SMI 555.85    Driver Version: 555.85"
            if command == ["ollama", "ps"]:
                return True, "NAME ID PROCESSOR UNTIL\nnomic-embed-text abc GPU"
            return False, "unexpected command"

        output = io.StringIO()
        with redirect_stdout(output):
            result = build_vector_store.run_gpu_preflight(command_runner=runner)

        self.assertTrue(result)
        self.assertIn("NVIDIA GPU detected", output.getvalue())
        self.assertIn("Ollama is running", output.getvalue())

    def test_gpu_preflight_handles_nvidia_failure_without_require_gpu(self) -> None:
        """GPU preflight should warn but not fail hard unless required."""

        def runner(command: list[str], timeout: int) -> tuple[bool, str]:
            del timeout
            if command == ["nvidia-smi"]:
                return False, "not found"
            if command == ["ollama", "ps"]:
                return True, "NAME ID PROCESSOR UNTIL"
            return False, "unexpected command"

        output = io.StringIO()
        with redirect_stdout(output):
            result = build_vector_store.run_gpu_preflight(command_runner=runner)

        self.assertFalse(result)
        self.assertIn("NVIDIA GPU was not detected", output.getvalue())
        self.assertIn("GPU/Ollama warning", output.getvalue())

    def test_gpu_preflight_handles_ollama_ps_success(self) -> None:
        """Ollama process status output should be printed when available."""

        def runner(command: list[str], timeout: int) -> tuple[bool, str]:
            del timeout
            if command == ["nvidia-smi"]:
                return True, "NVIDIA-SMI\nollama.exe"
            if command == ["ollama", "ps"]:
                return True, "llama3.2:3b running"
            return False, "unexpected command"

        output = io.StringIO()
        with redirect_stdout(output):
            build_vector_store.run_gpu_preflight(command_runner=runner)

        self.assertIn("llama3.2:3b running", output.getvalue())
        self.assertIn("Ollama appears in nvidia-smi output", output.getvalue())

    def test_gpu_preflight_require_gpu_failure_path(self) -> None:
        """--require-gpu mode should report failure when preflight fails."""

        def runner(command: list[str], timeout: int) -> tuple[bool, str]:
            del command, timeout
            return False, "missing"

        output = io.StringIO()
        with redirect_stdout(output):
            result = build_vector_store.run_gpu_preflight(
                require_gpu=True,
                command_runner=runner,
            )

        self.assertFalse(result)
        self.assertIn("--require-gpu was set", output.getvalue())


if __name__ == "__main__":
    unittest.main()
