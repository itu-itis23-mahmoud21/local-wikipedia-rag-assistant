"""Tests for build_vector_store.py helper behavior."""

from contextlib import redirect_stdout
import io
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
