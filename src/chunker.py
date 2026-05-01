"""Document chunking utilities for processed Wikipedia text.

The chunker keeps the strategy intentionally simple for a university RAG
project: normalize text, prefer paragraph boundaries, split oversized
paragraphs by sentence and word boundaries where possible, and add overlap
between neighboring chunks where practical.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

from src import config


_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    """One retrieval-ready text chunk and its lightweight metadata."""

    chunk_index: int
    text: str
    char_count: int
    token_estimate: int
    start_char: int | None = None
    end_char: int | None = None


def estimate_tokens(text: str) -> int:
    """Estimate token count with a simple character-based approximation."""

    if not text:
        return 0
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE_CHARS,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Split text into deterministic chunks suitable for later retrieval."""

    _validate_chunk_settings(chunk_size, overlap)

    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []

    paragraphs = _split_paragraphs(normalized_text)
    base_chunks = _build_base_chunks(paragraphs, chunk_size)
    chunk_texts = [chunk for chunk in _add_overlap(base_chunks, overlap) if chunk.strip()]

    return [
        TextChunk(
            chunk_index=index,
            text=chunk,
            char_count=len(chunk),
            token_estimate=estimate_tokens(chunk),
        )
        for index, chunk in enumerate(chunk_texts)
    ]


def chunk_file(
    path: str | Path,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE_CHARS,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Read a UTF-8 text file and split it into chunks."""

    text = Path(path).read_text(encoding="utf-8")
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    """Validate chunk sizing settings."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def _normalize_text(text: str) -> str:
    """Normalize line endings, whitespace, and excessive blank lines."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    output_lines: list[str] = []
    previous_blank = False

    for line in normalized.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            if not previous_blank:
                output_lines.append("")
            previous_blank = True
            continue

        output_lines.append(clean_line)
        previous_blank = False

    return "\n".join(output_lines).strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split normalized text into non-empty paragraphs."""

    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def _build_base_chunks(
    paragraphs: list[str],
    chunk_size: int,
) -> list[str]:
    """Build chunks by paragraph, splitting oversized paragraphs safely."""

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_length = 0
            chunks.extend(_split_long_paragraph(paragraph, chunk_size))
            continue

        separator_length = 2 if current_parts else 0
        next_length = current_length + separator_length + len(paragraph)

        if current_parts and next_length > chunk_size:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = [paragraph]
            current_length = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_length = next_length

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _split_long_paragraph(
    paragraph: str,
    chunk_size: int,
) -> list[str]:
    """Split a long paragraph by sentences, then words, then characters."""

    sentences = _split_sentences(paragraph)
    return _pack_text_units(
        sentences,
        chunk_size=chunk_size,
        separator=" ",
        overflow_splitter=_split_long_sentence,
    )


def _split_sentences(text: str) -> list[str]:
    """Split text on simple sentence boundaries while keeping punctuation."""

    clean_text = text.strip()
    if not clean_text:
        return []
    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_PATTERN.split(clean_text)
        if sentence.strip()
    ]


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Split a sentence by whole words, with character fallback for long words."""

    words = sentence.split()
    if not words:
        return []

    return _pack_text_units(
        words,
        chunk_size=chunk_size,
        separator=" ",
        overflow_splitter=_split_long_word,
    )


def _split_long_word(word: str, chunk_size: int) -> list[str]:
    """Split an abnormal long word by characters as a final fallback."""

    clean_word = word.strip()
    if not clean_word:
        return []
    return [
        clean_word[start : start + chunk_size]
        for start in range(0, len(clean_word), chunk_size)
        if clean_word[start : start + chunk_size]
    ]


def _pack_text_units(
    units: list[str],
    chunk_size: int,
    separator: str,
    overflow_splitter: Callable[[str, int], list[str]],
) -> list[str]:
    """Pack sentences or words into chunks without exceeding chunk_size."""

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for unit in units:
        clean_unit = unit.strip()
        if not clean_unit:
            continue

        if len(clean_unit) > chunk_size:
            if current_parts:
                chunks.append(separator.join(current_parts).strip())
                current_parts = []
                current_length = 0
            chunks.extend(overflow_splitter(clean_unit, chunk_size))
            continue

        separator_length = len(separator) if current_parts else 0
        next_length = current_length + separator_length + len(clean_unit)

        if current_parts and next_length > chunk_size:
            chunks.append(separator.join(current_parts).strip())
            current_parts = [clean_unit]
            current_length = len(clean_unit)
        else:
            current_parts.append(clean_unit)
            current_length = next_length

    if current_parts:
        chunks.append(separator.join(current_parts).strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap text between neighboring chunks where useful."""

    if overlap == 0 or not chunks:
        return chunks

    overlapped_chunks = [chunks[0]]
    for index in range(1, len(chunks)):
        previous_tail = _overlap_tail(chunks[index - 1], overlap)
        current_chunk = chunks[index]
        if previous_tail and not current_chunk.startswith(previous_tail):
            current_chunk = f"{previous_tail}\n\n{current_chunk}".strip()
        overlapped_chunks.append(current_chunk)

    return overlapped_chunks


def _overlap_tail(text: str, overlap: int) -> str:
    """Return a clean overlap tail, preferring complete trailing words."""

    if overlap <= 0:
        return ""

    clean_text = text.strip()
    if not clean_text:
        return ""
    if len(clean_text) <= overlap:
        return clean_text

    words = clean_text.split()
    selected_words: list[str] = []
    selected_length = 0

    for word in reversed(words):
        next_length = len(word) if not selected_words else selected_length + 1 + len(word)
        if next_length > overlap:
            break
        selected_words.insert(0, word)
        selected_length = next_length

    if selected_words:
        return " ".join(selected_words)

    # Character fallback is only for abnormal text where no whole word fits.
    tail = clean_text[-overlap:].strip()
    first_space = tail.find(" ")
    if first_space != -1 and first_space + 1 < len(tail):
        clean_tail = tail[first_space + 1 :].strip()
        if clean_tail:
            return clean_tail
    return tail
