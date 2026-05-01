"""Configuration placeholders for the local RAG system.

This module will centralize project paths, Ollama model names, Chroma collection
settings, SQLite paths, and retrieval defaults.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
SQLITE_PATH = DATA_DIR / "metadata.sqlite"

OLLAMA_GENERATION_MODEL = "llama3.2:3b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_COLLECTION_NAME = "wikipedia_entities"
