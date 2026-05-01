# Product PRD: Local Wikipedia RAG Assistant

## Product Goal

Build a local ChatGPT-style assistant that answers questions about famous people
and famous places using Wikipedia-derived context, local embeddings, a local
vector database, SQLite metadata, and a local Ollama LLM.

## Target User

The target user is a student, evaluator, or local desktop user who wants to ask
grounded questions about a curated set of famous people and places without
sending prompts, context, or generated responses to external LLM APIs.

## Core Requirements

- Ingest Wikipedia pages locally.
- Include at least 20 famous people and 20 famous places; this project will use
  50 people and 50 places.
- Split documents into chunks.
- Generate embeddings locally.
- Store embeddings in a vector database.
- Use SQLite for local metadata/database storage.
- Retrieve relevant chunks based on user questions.
- Decide whether a query is about a person, a place, or both.
- Generate grounded answers using a local LLM.
- Return `I don't know` when the answer is not in the retrieved context.
- Provide a simple chat-style UI with Streamlit.
- Include documentation, ingestion scripts, vector store creation logic, and a
  demo video link placeholder.

## High-Level Architecture

1. Entity configuration defines the selected people and places.
2. Wikipedia ingestion downloads or reads page content for each entity.
3. Chunking logic splits long pages into smaller passages.
4. SQLite stores entity and chunk metadata.
5. Ollama generates local embeddings with `nomic-embed-text`.
6. Chroma stores all vectors in one collection with metadata filters.
7. Query routing classifies each question as person, place, or mixed.
8. Retrieval returns relevant chunks from Chroma.
9. Ollama generates grounded answers with `llama3.2:3b`.
10. Streamlit presents a chat-style local UI.

## Acceptance Criteria

Coming soon. Acceptance criteria will be finalized after the ingestion,
retrieval, generation, and UI behavior are implemented.
