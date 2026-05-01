# Production Deployment Recommendation

## Current Local Prototype Summary

The current Local Wikipedia RAG Assistant is a laptop-first prototype. It uses
Python, Streamlit, SQLite, Chroma, and Ollama to ingest Wikipedia pages, build a
local vector store, retrieve relevant chunks, and generate grounded answers
without external LLM APIs.

## Production Deployment Options

1. Local desktop package
   - Best privacy posture.
   - Requires user-managed Ollama models and local storage.

2. Private server deployment
   - Centralizes ingestion, vector search, and model serving.
   - Easier to monitor and update.

3. Hybrid deployment
   - Local UI with remote private infrastructure.
   - Useful when laptops cannot run local models efficiently.

## Recommended Production Architecture

For a production version, use a private server or managed internal environment:

```text
Scheduled ingestion workers
  -> document object storage
  -> metadata database
  -> chunking and embedding workers
  -> managed vector database
  -> retrieval API
  -> local/private LLM service
  -> web UI
```

This separates data preparation, retrieval, generation, and presentation while
preserving a clear privacy boundary.

## Data Ingestion Recommendations

- Use scheduled ingestion jobs instead of manual scripts.
- Store raw and processed document versions.
- Track source URL, fetch timestamp, page revision ID, status, and errors.
- Add retry logic and rate limiting for Wikipedia requests.
- Support incremental updates instead of rebuilding everything.

## Model and Embedding Recommendations

- Keep embeddings and generation in a private environment.
- Benchmark several embedding models for retrieval quality.
- Use a stronger generation model for production if hardware allows.
- Keep a model registry with version, embedding dimension, and build date.
- Rebuild vectors when the embedding model changes.

## Vector Database Recommendations

- Chroma is suitable for the homework prototype and small local deployments.
- For production, consider a managed vector database or a hardened self-hosted
  service if concurrency, backups, and observability are required.
- Keep metadata filters for entity type, source, document ID, and model.
- Store vector IDs back in the metadata database for traceability.

## API and Backend Recommendation

- Move core RAG logic behind a FastAPI backend for production.
- Keep ingestion, chunking, vector build, retrieval, and generation as separate
  services or jobs.
- Add request validation, structured errors, and timeout handling.
- Expose endpoints for status, query, sources, and rebuild operations.

## UI Recommendation

- Streamlit is appropriate for a course demo and local prototype.
- For production, use a standard web frontend if authentication, roles,
  analytics, and design control are needed.
- Keep source display and retrieved context visible for trust and debugging.

## Monitoring and Observability

- Log ingestion success/failure counts.
- Track retrieval latency, generation latency, and answer failure rates.
- Track empty retrieval results and `I don't know.` responses.
- Add dashboards for vector count, document count, and last ingestion time.
- Store anonymized query metrics only when privacy requirements allow it.

## Security and Privacy

- Do not send prompts, retrieved context, embeddings, or generated answers to
  external LLM APIs unless explicitly approved.
- Keep local data directories out of Git.
- Add access control for any multi-user deployment.
- Validate uploaded or ingested content if future versions support user data.
- Back up metadata and vector stores according to data retention policies.

## Scalability

- Parallelize ingestion and embedding jobs.
- Batch embeddings where the model server supports it.
- Use queues for large rebuilds.
- Separate hot query serving from offline data preparation.
- Add caching for repeated queries and common retrieval results.

## Limitations

- The prototype covers a fixed set of 100 entities.
- Wikipedia extraction can fail or change over time.
- Rule-based query routing is simple and may misclassify ambiguous questions.
- Local model quality depends on available hardware and model size.
- The system currently lacks automated answer quality evaluation.

## Future Improvements

- Add automated retrieval and generation evaluation.
- Improve query routing with a lightweight classifier.
- Add citation snippets with chunk-level source references.
- Add admin UI for ingestion and vector-store rebuilds.
- Add Docker or local installer packaging.
- Add support for additional entity categories.
