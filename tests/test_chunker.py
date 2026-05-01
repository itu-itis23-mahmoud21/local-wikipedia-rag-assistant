"""Tests for document chunking utilities."""

from pathlib import Path
import re
import tempfile
import unittest

from src.chunker import TextChunk, chunk_file, chunk_text, estimate_tokens


class TestChunker(unittest.TestCase):
    """Tests for deterministic Wikipedia document chunking."""

    @staticmethod
    def _normalized_words(text: str) -> set[str]:
        """Return lowercase word tokens stripped of punctuation for assertions."""

        return {
            TestChunker._normalize_token(word)
            for word in text.split()
            if TestChunker._normalize_token(word)
        }

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Normalize a token for boundary-focused tests."""

        return re.sub(r"^\W+|\W+$", "", token).casefold()

    @staticmethod
    def _first_word(text: str) -> str:
        """Return the first normalized word in a chunk."""

        return TestChunker._normalize_token(text.split()[0])

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
        self.assertTrue(all(chunk.char_count <= 62 for chunk in chunks))

    def test_long_paragraph_sentences_start_with_complete_words(self) -> None:
        """Sentence-aware splitting should avoid mid-word chunk starts."""

        text = (
            "Theory explains electricity in detail with careful historical context. "
            "Especially notable experiments followed across several laboratories. "
            "Researchers documented observations and shared complete reports. "
            "Final sentence closes the paragraph with a clean boundary."
        )

        chunks = chunk_text(text, chunk_size=85, overlap=20)
        source_words = self._normalized_words(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn(self._first_word(chunk.text), source_words)

    def test_normal_sentences_are_grouped_on_sentence_boundaries(self) -> None:
        """Long normal paragraphs should prefer complete sentence chunks."""

        text = (
            "Alpha researchers wrote the first complete sentence. "
            "Beta engineers added another complete sentence for context. "
            "Gamma historians described the final complete sentence clearly."
        )

        chunks = chunk_text(text, chunk_size=70, overlap=15)
        source_words = self._normalized_words(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn(self._first_word(chunk.text), source_words)

    def test_very_long_sentence_without_punctuation_splits_by_words(self) -> None:
        """A long sentence without punctuation should still split on words."""

        text = " ".join(f"word{index}" for index in range(40))

        chunks = chunk_text(text, chunk_size=45, overlap=12)
        source_words = self._normalized_words(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn(self._first_word(chunk.text), source_words)

    def test_extremely_long_word_uses_character_fallback_safely(self) -> None:
        """A single abnormal word should still be chunked without looping."""

        chunks = chunk_text("x" * 130, chunk_size=40, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(chunk.char_count <= 47 for chunk in chunks))

    def test_overlap_tail_starts_with_complete_word_for_normal_text(self) -> None:
        """Overlap should prefer whole trailing words instead of word suffixes."""

        text = (
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. "
            "Mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega. "
            "Another sentence keeps the paragraph long enough for chunking."
        )

        chunks = chunk_text(text, chunk_size=75, overlap=18)
        source_words = self._normalized_words(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks[1:]:
            self.assertIn(self._first_word(chunk.text), source_words)

    def test_chunks_do_not_exceed_size_plus_overlap_unexpectedly(self) -> None:
        """Overlap may grow chunks slightly, but should stay bounded."""

        chunk_size = 70
        overlap = 15
        text = (
            "First sentence has enough words to participate in chunking. "
            "Second sentence provides additional source material. "
            "Third sentence adds more words for another chunk. "
            "Fourth sentence closes the example cleanly."
        )

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        self.assertTrue(chunks)
        self.assertTrue(
            all(chunk.char_count <= chunk_size + overlap + 2 for chunk in chunks)
        )

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
