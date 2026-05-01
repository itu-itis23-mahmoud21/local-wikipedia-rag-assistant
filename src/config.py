"""Configuration placeholders for the local RAG system.

This module will centralize project paths, Ollama model names, Chroma collection
settings, SQLite paths, and retrieval defaults.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SQLITE_DIR = DATA_DIR / "sqlite"
SQLITE_DB_PATH = SQLITE_DIR / "rag_metadata.sqlite"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

OLLAMA_GENERATION_MODEL = "llama3.2:3b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_COLLECTION_NAME = "wikipedia_entities"

# Backwards-compatible aliases for earlier skeleton placeholders.
CHROMA_DIR = CHROMA_DB_DIR
SQLITE_PATH = SQLITE_DB_PATH
