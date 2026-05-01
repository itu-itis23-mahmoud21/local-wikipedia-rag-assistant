"""Tests for document chunking utilities."""

from pathlib import Path
import tempfile
import unittest

from src.chunker import TextChunk, chunk_file, chunk_text, estimate_tokens


class TestChunker(unittest.TestCase):
    """Tests for deterministic Wikipedia document chunking."""

    def test_blank_text_returns_empty_list(self) -> None:
        """Blank input should not create chunks."""

        self.assertEqual(chunk_text(" \n\n\t "), [])

    def test_short_text_returns_one_chunk(self) -> None:
        """Text shorter than chunk_size should produce one stripped chunk."""

        chunks = chunk_text(
            "  Albert Einstein was a physicist.  ",
            chunk_size=100,
            overlap=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], TextChunk)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].text, "Albert Einstein was a physicist.")

    def test_chunk_indexes_are_sequential(self) -> None:
        """Chunk indexes should start at zero and increase by one."""

        text = "\n\n".join(f"Paragraph {index} " + ("x" * 20) for index in range(8))

        chunks = chunk_text(text, chunk_size=70, overlap=10)

        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(len(chunks))),
        )

    def test_char_count_and_token_estimate_are_correct(self) -> None:
        """Chunk metadata should match chunk text."""

        chunks = chunk_text("abcd" * 5, chunk_size=100, overlap=10)

        self.assertEqual(chunks[0].char_count, len(chunks[0].text))
        self.assertEqual(chunks[0].token_estimate, estimate_tokens(chunks[0].text))
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abc"), 1)

    def test_long_text_creates_multiple_chunks(self) -> None:
        """Long paragraph text should create multiple chunks."""

        text = "0123456789" * 30

        chunks = chunk_text(text, chunk_size=80, overlap=20)

        self.assertGreater(len(chunks), 1)

    def test_overlap_causes_shared_text_between_consecutive_chunks(self) -> None:
        """Neighboring chunks should share overlap text when practical."""

        first = "A" * 60
        second = "B" * 60
        text = f"{first}\n\n{second}"

        chunks = chunk_text(text, chunk_size=80, overlap=12)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn(chunks[0].text[-12:], chunks[1].text)

    def test_long_paragraph_is_split_safely(self) -> None:
        """Paragraphs longer than chunk_size should be split without looping."""

        text = "abcdefghijklmnopqrstuvwxyz" * 20

        chunks = chunk_text(text, chunk_size=50, overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(chunk.char_count <= 60 for chunk in chunks))

    def test_invalid_chunk_size_raises_value_error(self) -> None:
        """chunk_size must be positive."""

        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=0, overlap=0)

    def test_invalid_overlap_raises_value_error(self) -> None:
        """overlap must be non-negative and smaller than chunk_size."""

        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=-1)

        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)

    def test_chunk_file_reads_utf8_and_chunks_correctly(self) -> None:
        """chunk_file should read UTF-8 text and return chunks."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "document.txt"
            path.write_text("Sagrada Fam\u00edlia is in Barcelona.", encoding="utf-8")

            chunks = chunk_file(path, chunk_size=100, overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Sagrada Fam\u00edlia", chunks[0].text)

    def test_chunking_is_deterministic_for_same_input(self) -> None:
        """Same input and settings should return the same chunks."""

        text = "\n\n".join(f"Paragraph {index} " + ("x" * 30) for index in range(10))

        first_run = chunk_text(text, chunk_size=90, overlap=15)
        second_run = chunk_text(text, chunk_size=90, overlap=15)

        self.assertEqual(first_run, second_run)

    def test_no_chunk_text_is_empty_or_whitespace_only(self) -> None:
        """Generated chunks should always contain non-whitespace text."""

        text = "\n\n".join(["Alpha", "   ", "Beta" * 40, "\n", "Gamma"])

        chunks = chunk_text(text, chunk_size=40, overlap=5)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.text.strip() for chunk in chunks))
