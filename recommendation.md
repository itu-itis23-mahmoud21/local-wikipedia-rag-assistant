# Recommendation

## Production Deployment Recommendation

Coming soon. The final recommendation will describe how the local RAG assistant
could be packaged, distributed, and operated in a production-like environment.

## Privacy and Local Deployment Note

The recommended deployment model is local-first. Wikipedia data, embeddings,
retrieved chunks, user questions, and generated answers should remain on the
user's machine. Ollama should provide both embeddings and generation locally, so
no external LLM API is required.

## Future Improvements

Coming soon. Candidate areas include better entity expansion, richer metadata,
evaluation datasets, answer citation formatting, improved query routing,
incremental re-ingestion, and more robust hallucination checks.
