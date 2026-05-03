# Product Requirements Document

## Product Name

Local Wikipedia RAG Assistant

## Purpose

Build a local ChatGPT-style retrieval-augmented generation assistant for HW3 in
AI Aided Computer Engineering. The assistant must answer questions about a fixed
set of famous people and famous places using locally stored Wikipedia content,
local embeddings, a local vector database, and a local language model.

## Target Users

- Course instructors evaluating the homework submission
- Students demonstrating a complete local RAG pipeline
- Local users who want a small Wikipedia-based assistant after setup is complete

## Product Goals

- Run fully on a user's laptop after Wikipedia ingestion.
- Avoid external LLM and embedding APIs.
- Provide grounded answers from retrieved Wikipedia context.
- Show retrieved sources and context so answers can be inspected.
- Keep the codebase understandable for a university project.

## Scope

The configured knowledge base contains:

- 25 famous people
- 25 famous places
- 50 total Wikipedia entities

The system should support direct questions, generic person/place questions,
location-based place questions, and comparison questions.

## Core Requirements

- Maintain a static entity configuration for people and places.
- Ingest Wikipedia page text using the MediaWiki API.
- Save raw and processed article text locally.
- Remove low-value Wikipedia footer/reference sections from processed text.
- Store metadata in SQLite.
- Split processed articles into deterministic text chunks.
- Generate embeddings locally with Ollama `nomic-embed-text`.
- Store vectors in persistent sharded Chroma collections.
- Route queries as `person`, `place`, `both`, or `unknown`.
- Retrieve chunks with metadata filters for exact entity matches.
- Include intro chunks for exact entity and comparison queries.
- Generate answers locally with Ollama `llama3.2:3b`.
- Return `I don't know.` when retrieved context does not support an answer.
- Provide a Streamlit chat UI with source display, context display, copy
  controls, chat export, and local system status.
- Provide setup and reset scripts for repeatable local demos.

## Non-Goals

- No external LLM APIs.
- No cloud-hosted production deployment.
- No multi-user authentication system.
- No paid APIs.
- No large-scale general Wikipedia search beyond the configured entities.
- No advanced automatic answer grading system.

## User Flow

1. User installs Python dependencies.
2. User installs Ollama and pulls the required local models.
3. User runs the setup script to ingest, chunk, embed, and build the vector
   store.
4. User starts the Streamlit app.
5. User asks a question in the chat input.
6. The app routes the query, retrieves relevant context, generates a grounded
   answer, and displays sources.
7. User can inspect retrieved sources/context, copy them, export chat history,
   or reset generated local artifacts.

## Architecture

```text
Entity configuration
  -> Wikipedia ingestion
  -> raw local files
  -> processed local files
  -> SQLite metadata
  -> chunking
  -> Ollama embeddings
  -> sharded Chroma vector store
  -> query router
  -> retriever
  -> Ollama answer generator
  -> Streamlit chat UI
```

## Data Storage

SQLite stores:

- `entities`: configured entity names, entity type, source URL, timestamps
- `documents`: article title, source URL, local paths, status, errors
- `chunks`: chunk text, chunk index, character count, token estimate, vector ID
- `ingestion_runs`: run status, counts, timestamps, notes

Local files store:

- raw Wikipedia extracts under `data/raw/`
- cleaned processed text under `data/processed/`

Chroma stores:

- vector IDs such as `chunk-123`
- chunk text
- local embedding vectors
- metadata including entity, entity type, source URL, document ID, chunk ID,
  chunk index, and embedding model

## Retrieval Behavior

- Exact configured entity names should filter retrieval to those entities.
- Supported aliases, such as last names, should resolve to canonical entities
  when unambiguous.
- Generic person questions should search person chunks.
- Generic place questions should search place chunks.
- Location-based place questions should use configured place-location hints.
- Comparison questions should retrieve balanced context for each compared
  entity.
- Low-information chunks should be filtered when better semantic chunks are
  available.
- Retrieved overlap should be cleaned before display and generation.

## Generation Behavior

- The prompt must instruct the model to use only retrieved context.
- The model must not use outside knowledge.
- The model must not copy internal source metadata into the final answer.
- Unsupported questions must produce a concise `I don't know.` response.
- Direct questions should receive natural direct answers.
- Comparison questions should keep facts separated by entity and avoid moving
  records, numbers, roles, or dates from one entity to another.

## Local-Only Constraints

- Ollama handles generation locally with `llama3.2:3b`.
- Ollama handles embeddings locally with `nomic-embed-text`.
- SQLite and Chroma data remain on the local machine.
- Wikipedia is contacted only during ingestion.
- Prompts, retrieved context, embeddings, and generated answers must not be sent
  to external LLM services.

## Scripts

- `scripts/ingest_wikipedia.py`: fetch and store Wikipedia pages.
- `scripts/chunk_documents.py`: chunk processed documents and store chunk
  metadata.
- `scripts/build_vector_store.py`: embed chunks and build sharded Chroma
  collections.
- `scripts/setup_all.py`: run the full preparation pipeline.
- `scripts/reset_system.py`: safely remove generated local data.

## UI Requirements

- Use Streamlit with a wide layout.
- Show title, subtitle, and local stack badges.
- Show model, storage, dataset, and vector-store status in the sidebar.
- Show configured people and places in collapsible sidebar sections.
- Use a chat-style message layout.
- Prevent overlapping generation requests.
- Provide stop-generation behavior.
- Provide copy buttons for retrieved sources and context.
- Provide TXT chat export options.
- Show friendly errors when Ollama, Chroma, or local data are not ready.

## Acceptance Criteria

- `python -m unittest discover -v` passes.
- `python -m compileall -q app.py src scripts tests` passes.
- The setup script can build a small demo dataset with `--limit`.
- The app can answer direct person and place questions.
- The app can answer configured location questions such as places in Turkey,
  Egypt, Paris, and New York.
- The app can handle comparison questions without mixing facts between entities.
- Unsupported questions return `I don't know.` instead of unsupported claims.
- Generated local artifacts are ignored by Git.

## Future Improvements

- Add a formal evaluation set for retrieval and generation quality.
- Add incremental Wikipedia refresh support.
- Add configurable retrieval settings in the UI.
- Add an admin panel for ingestion and vector-store rebuilds.
- Add packaging with Docker or a local installer.
