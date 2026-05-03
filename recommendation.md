# Production Deployment Recommendation

## Current Prototype

The Local Wikipedia RAG Assistant is a local-first homework prototype. It uses
Python, Streamlit, SQLite, sharded Chroma collections, and Ollama to answer
questions about 25 famous people and 25 famous places from locally ingested
Wikipedia data. No external LLM API is used.

## Recommended Production Direction

For a real production deployment, keep the privacy-first design but move the
heavy work behind a private backend:

```text
Scheduled Wikipedia ingestion
  -> raw/processed document storage
  -> metadata database
  -> chunking + embedding workers
  -> vector database
  -> retrieval API
  -> private/local LLM service
  -> web UI
```

Streamlit is appropriate for the class demo. A production product would likely
use a standard web frontend with a backend API for authentication, monitoring,
and operational controls.

## Data and Retrieval

- Keep raw and processed Wikipedia text for traceability.
- Store source URL, entity type, document status, errors, chunk IDs, and vector
  IDs in a metadata database.
- Use scheduled refresh jobs instead of manual ingestion scripts.
- Continue using metadata filters for entity name and entity type.
- Keep sharding or another partitioning strategy for larger vector stores to
  avoid single-index reliability issues.

## Models

- Keep embeddings and generation inside the private environment.
- Ollama is good for local demos and small deployments.
- For production, benchmark stronger generation and embedding models if better
  hardware is available.
- Track model names, embedding dimensions, build dates, and rebuild vectors when
  the embedding model changes.

## Operations

- Add logging for ingestion, chunking, embedding, retrieval, and generation.
- Monitor vector counts, failed documents, retrieval latency, generation latency,
  and `I don't know.` rates.
- Add backups for SQLite metadata and vector-store data.
- Add health checks that reopen the vector store and run a real query after
  rebuilds.

## Security and Privacy

- Do not send prompts, retrieved context, embeddings, or answers to external LLM
  APIs without explicit approval.
- Keep `data/`, `chroma_db/`, local databases, logs, and secrets out of Git.
- Add access control before any multi-user deployment.

## Future Improvements

- Add automated retrieval and answer-quality evaluation.
- Improve query routing with a lightweight classifier.
- Add an admin screen for ingestion and rebuild status.
- Package the app with Docker or a local installer.
- Expand beyond people and places once the pipeline is stable.
