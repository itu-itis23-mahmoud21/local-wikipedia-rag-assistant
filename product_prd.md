# Product PRD: Local Wikipedia RAG Assistant

## Product Name

Local Wikipedia RAG Assistant

## Problem Statement

Students need to demonstrate a complete retrieval-augmented generation workflow
without relying on external LLM APIs. The system must be understandable,
reproducible, and runnable on a local laptop while answering questions from a
controlled Wikipedia-based knowledge base.

## Goal

Build a local ChatGPT-style assistant that answers questions about famous people
and famous places using Wikipedia-derived context, local embeddings, a local
vector database, SQLite metadata, and a local Ollama LLM.

## Target Users

- Course instructors evaluating HW3
- Students learning RAG system architecture
- Local desktop users who want a small offline-first knowledge assistant after
  ingestion is complete

## Core Requirements

- Ingest Wikipedia pages for configured entities.
- Include 50 famous people and 50 famous places.
- Store entity, document, chunk, and ingestion metadata in SQLite.
- Split processed articles into retrieval chunks.
- Generate embeddings locally with Ollama `nomic-embed-text`.
- Store all vectors in one Chroma collection with metadata.
- Route queries as `person`, `place`, `both`, or `unknown`.
- Retrieve relevant chunks from Chroma.
- Generate grounded answers with Ollama `llama3.2:3b`.
- Return `I don't know.` when the answer is not available in retrieved context.
- Provide a Streamlit chat UI with source/context display.
- Provide setup and reset scripts.

## Non-Goals

- No external LLM APIs.
- No production authentication or multi-user account system.
- No cloud deployment in the homework prototype.
- No advanced answer evaluation framework.
- No automatic refresh of Wikipedia pages after ingestion.

## User Flow

1. User installs Python dependencies.
2. User installs and starts Ollama.
3. User pulls `llama3.2:3b` and `nomic-embed-text`.
4. User runs `scripts/setup_all.py`.
5. The system ingests Wikipedia pages, chunks text, embeds chunks, and builds
   Chroma.
6. User starts Streamlit with `streamlit run app.py`.
7. User asks a question.
8. The app routes the query, retrieves context, generates a grounded answer,
   and displays sources.

## Architecture

```text
Entity config
  -> Wikipedia client
  -> raw and processed text files
  -> SQLite metadata
  -> chunker
  -> Ollama embeddings
  -> Chroma vector store
  -> query router
  -> retriever
  -> Ollama answer generator
  -> Streamlit UI
```

## Data Model Overview

SQLite stores:

- `entities`: entity name, type, source URL, timestamps
- `documents`: Wikipedia document metadata and local file paths
- `chunks`: chunk text, chunk index, character count, token estimate, vector ID
- `ingestion_runs`: run status, counts, timestamps, notes

Chroma stores:

- vector ID such as `chunk-123`
- chunk text as document content
- local embedding vector
- metadata including `chunk_id`, `document_id`, `entity_id`, `entity`,
  `entity_type`, `source_url`, `chunk_index`, and embedding model

## Retrieval and Generation Behavior

- Query routing first checks exact configured entity mentions.
- If a query mentions people, retrieval can filter to person chunks.
- If a query mentions places, retrieval can filter to place chunks.
- Comparison or mixed queries search across all chunks.
- Retrieved chunks are formatted into a context block.
- The local generation prompt instructs the model to answer using only that
  context and to say `I don't know.` when the context is insufficient.

## Local-Only Constraints

- Generation uses local Ollama `llama3.2:3b`.
- Embeddings use local Ollama `nomic-embed-text`.
- SQLite and Chroma are local files.
- Wikipedia network access is used only during ingestion.
- No prompts or answers are sent to external LLM services.

## Failure Behavior

- Missing or empty Wikipedia pages are recorded as failed document metadata.
- Missing processed files are skipped or reported by chunking scripts.
- Missing Chroma/Ollama setup is shown as a friendly UI error.
- Blank queries are rejected.
- Empty retrieved context returns `I don't know.`

## Acceptance Criteria

- Repository contains source code, scripts, tests, README, PRD, recommendation,
  and demo checklist.
- `python -m unittest discover -v` passes.
- `python -m compileall -q app.py src scripts tests` passes.
- `python scripts/setup_all.py --limit 5 --reset-db --reset-chunks --reset-collection`
  can build a small local dataset when Ollama is running.
- `streamlit run app.py` opens a local chat UI.
- The app displays answers, route information, retrieved sources, and retrieved
  context.
- The app handles missing local system pieces without raw stack traces.

## Future Improvements

- Add a formal retrieval and answer quality evaluation set.
- Add better entity disambiguation.
- Add incremental ingestion and update detection.
- Add richer citation formatting.
- Add configurable chunking and retrieval settings in the UI.
- Add packaging for a one-command local desktop demo.
