# Local Wikipedia RAG Assistant

Local Wikipedia RAG Assistant is a ChatGPT-style retrieval-augmented generation
project for answering questions about famous people and famous places using
locally ingested Wikipedia data.

This repository is being built for HW3 in the AI Aided Computer Engineering
course.

## Overview

The final system will:

- ingest Wikipedia pages for famous people and famous places;
- split pages into retrievable chunks;
- generate embeddings locally;
- store chunks and metadata locally;
- retrieve relevant context for a user question;
- route questions as person, place, or mixed queries;
- generate grounded answers with a local LLM;
- answer `I don't know` when the retrieved context does not contain the answer.

## Chosen Stack

- Python
- Streamlit for the chat UI
- Ollama for local generation and embeddings
- `llama3.2:3b` as the local generation model
- `nomic-embed-text` as the local embedding model
- Chroma as the vector database
- SQLite for local metadata storage
- One Chroma collection with metadata fields such as `entity`, `entity_type`,
  `source`, `source_url`, and `chunk_index`

## Local-Only Note

The project is designed to run fully locally on a laptop. No external LLM APIs
are allowed or required for generation or embeddings.

## Run Instructions

Coming soon. The Streamlit UI and ingestion pipeline are placeholders in this
initial skeleton.

## Demo Video

Coming soon.
