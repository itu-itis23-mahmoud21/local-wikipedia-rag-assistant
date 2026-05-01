# Local Wikipedia RAG Assistant

## Overview

Local Wikipedia RAG Assistant is a localhost-only retrieval-augmented generation
assistant that answers questions about famous people and famous places using
locally ingested Wikipedia data.

The system downloads Wikipedia article text, stores local metadata in SQLite,
chunks the documents, embeds chunks with a local Ollama embedding model, stores
vectors in Chroma, retrieves relevant chunks for a user question, and generates
grounded answers with a local Ollama language model.

## Course Context

This repository is HW3 for the AI Aided Computer Engineering course. The goal is
to build a simplified local ChatGPT-style RAG system without external LLM APIs.

## Features

- 50 configured famous people and 50 configured famous places
- Wikipedia ingestion through the MediaWiki API
- Local SQLite metadata database
- Deterministic document chunking
- Local embeddings with Ollama `nomic-embed-text`
- Persistent Chroma vector store
- Query routing for `person`, `place`, `both`, and `unknown`
- Local answer generation with Ollama `llama3.2:3b`
- Streamlit chat UI
- Retrieved source and context display
- Setup and reset scripts

## Architecture

```text
Wikipedia API
  -> local files
  -> SQLite metadata
  -> chunking
  -> Ollama embeddings
  -> Chroma vector store
  -> retriever
  -> Ollama LLM
  -> Streamlit UI
```

## Tech Stack

- Python
- Streamlit
- Ollama
- `llama3.2:3b`
- `nomic-embed-text`
- Chroma
- SQLite
- requests
- unittest

## Local-Only Guarantee

- No external LLM API is used.
- Ollama runs both generation and embeddings locally.
- Wikipedia is used only for ingestion and source data collection.
- Retrieved context, prompts, embeddings, generated answers, SQLite metadata,
  and Chroma vectors stay on the local machine.

## Prerequisites

- Python 3.10+ or 3.11+
- Git
- Ollama installed and running
- Enough disk space for local article text, SQLite metadata, and Chroma vectors

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Pull the local Ollama models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Build Local Data

Recommended full build:

```powershell
python scripts/setup_all.py --reset-db --reset-chunks --reset-collection
```

Quick test build with a small subset:

```powershell
python scripts/setup_all.py --limit 5 --reset-db --reset-chunks --reset-collection
```

The full build fetches Wikipedia pages, chunks documents, generates local
embeddings, and writes vectors to Chroma. It requires Ollama to be running.

## Manual Pipeline Commands

Run these only if you want to execute each stage separately:

```powershell
python scripts/ingest_wikipedia.py
python scripts/chunk_documents.py
python scripts/build_vector_store.py
```

## Run the App

```powershell
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal.

## Reset Generated Artifacts

This removes generated local data and vector-store files:

```powershell
python scripts/reset_system.py --yes
```

Generated artifacts are ignored by Git through `.gitignore`.

## Example Questions

People:

- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Mohamed Salah famous?
- Compare Lionel Messi and Cristiano Ronaldo.

Places:

- Where is the Eiffel Tower located?
- Which famous place is located in Turkey?
- What was the Colosseum used for?
- Where is Mount Everest?

Mixed and failure cases:

- Which person is associated with electricity?
- Compare Albert Einstein and Nikola Tesla.
- Who is the president of Mars?
- Tell me about a random unknown person John Doe.

For questions not supported by retrieved context, the assistant is instructed to
answer: `I don't know.`

## Testing

Run unit tests:

```powershell
python -m unittest discover -v
```

Compile-check Python files:

```powershell
python -m compileall -q app.py src scripts tests
```

## Repository Structure

```text
.
|-- app.py
|-- README.md
|-- product_prd.md
|-- recommendation.md
|-- DEMO_CHECKLIST.md
|-- requirements.txt
|-- src/
|   |-- config.py
|   |-- entities.py
|   |-- database.py
|   |-- wiki_client.py
|   |-- chunker.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- query_router.py
|   |-- retriever.py
|   `-- generator.py
|-- scripts/
|   |-- ingest_wikipedia.py
|   |-- chunk_documents.py
|   |-- build_vector_store.py
|   |-- setup_all.py
|   `-- reset_system.py
`-- tests/
```

## Troubleshooting

Ollama is not running:

- Start Ollama.
- Confirm it responds with `ollama list`.

Models are not pulled:

- Run `ollama pull llama3.2:3b`.
- Run `ollama pull nomic-embed-text`.

Empty vector store:

- Run `python scripts/setup_all.py --reset-db --reset-chunks --reset-collection`.
- For a small check, run the same command with `--limit 5`.

Wikipedia request failure:

- Check internet access during ingestion.
- Rerun ingestion with `python scripts/ingest_wikipedia.py --force`.

Streamlit cannot start:

- Confirm dependencies are installed with `pip install -r requirements.txt`.
- Confirm the virtual environment is activated.

Chroma/data reset:

- Run `python scripts/reset_system.py --yes`.
- Rebuild with `python scripts/setup_all.py --reset-db --reset-chunks --reset-collection`.

## Demo Video

Demo video link: TODO - add Loom or unlisted YouTube link before submission.
