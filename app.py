"""Streamlit chat UI for the Local Wikipedia RAG Assistant."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from src import config
from src.database import MetadataDB
from src.generator import GeneratedAnswer, OllamaAnswerGenerator
from src.vector_store import ChromaVectorStore


APP_TITLE = "Local Wikipedia RAG Assistant"
APP_CAPTION = (
    "Ask questions about famous people and famous places using locally ingested "
    "Wikipedia data."
)


def initialize_session_state(session_state: MutableMapping[str, Any] | None = None) -> None:
    """Initialize Streamlit session state keys used by the chat UI."""

    state = session_state if session_state is not None else st.session_state
    if "messages" not in state:
        state["messages"] = []


def get_system_status(
    db_factory=MetadataDB,
    vector_store_factory=ChromaVectorStore,
) -> dict:
    """Collect local database and vector-store status for the sidebar."""

    status = {
        "generation_model": config.OLLAMA_GENERATION_MODEL,
        "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
        "collection_name": config.CHROMA_COLLECTION_NAME,
        "sqlite_db_path": str(config.SQLITE_DB_PATH),
        "chroma_db_path": str(config.CHROMA_DB_DIR),
        "counts": {},
        "vector_count": None,
        "database_error": None,
        "vector_error": None,
    }

    try:
        db = db_factory()
        db.init_schema()
        status["sqlite_db_path"] = str(db.db_path)
        status["counts"] = db.get_summary_counts()
    except Exception as exc:
        status["database_error"] = _friendly_runtime_message(exc)

    try:
        vector_store = vector_store_factory()
        status["vector_count"] = vector_store.count()
    except Exception as exc:
        status["vector_error"] = _friendly_runtime_message(exc)

    return status


def create_generator() -> OllamaAnswerGenerator:
    """Create the default local answer generator."""

    return OllamaAnswerGenerator()


def normalize_source(source: dict) -> dict:
    """Normalize source metadata for display and tests."""

    return {
        "rank": source.get("rank", ""),
        "entity": source.get("entity") or "",
        "entity_type": source.get("entity_type") or "",
        "source_url": source.get("source_url") or "",
        "chunk_id": source.get("chunk_id", ""),
        "distance": source.get("distance"),
        "preview": source.get("preview") or "",
    }


def answer_to_chat_message(answer: GeneratedAnswer) -> dict:
    """Convert a GeneratedAnswer into a session-state chat message."""

    return {
        "role": "assistant",
        "content": answer.answer,
        "route": answer.route,
        "sources": answer.sources,
        "context": answer.context,
        "model": answer.model,
        "error": False,
    }


def error_to_chat_message(exc: Exception) -> dict:
    """Convert an expected runtime exception into a friendly assistant message."""

    return {
        "role": "assistant",
        "content": _friendly_runtime_message(exc),
        "route": None,
        "sources": [],
        "context": "",
        "model": config.OLLAMA_GENERATION_MODEL,
        "error": True,
    }


def handle_user_query(
    prompt: str,
    generator: OllamaAnswerGenerator | None = None,
    top_k: int | None = None,
) -> dict:
    """Run the local RAG answer flow and return an assistant chat message."""

    try:
        active_generator = generator or create_generator()
        answer = active_generator.answer_query(prompt, top_k=top_k)
        return answer_to_chat_message(answer)
    except Exception as exc:
        return error_to_chat_message(exc)


def render_sidebar(status: dict) -> None:
    """Render the status sidebar."""

    st.sidebar.header("Local System")
    st.sidebar.write(f"Generation model: `{status['generation_model']}`")
    st.sidebar.write(f"Embedding model: `{status['embedding_model']}`")
    st.sidebar.write(f"Chroma collection: `{status['collection_name']}`")
    st.sidebar.write(f"SQLite database: `{status['sqlite_db_path']}`")
    st.sidebar.write(f"Chroma database: `{status['chroma_db_path']}`")

    st.sidebar.subheader("Metadata Counts")
    if status["database_error"]:
        st.sidebar.warning(status["database_error"])
    else:
        counts = status.get("counts", {})
        st.sidebar.write(f"Entities: `{counts.get('entities', 0)}`")
        st.sidebar.write(f"People: `{counts.get('people', 0)}`")
        st.sidebar.write(f"Places: `{counts.get('places', 0)}`")
        st.sidebar.write(f"Documents: `{counts.get('documents', 0)}`")
        st.sidebar.write(f"Chunks: `{counts.get('chunks', 0)}`")
        st.sidebar.write(
            f"Successful documents: `{counts.get('successful_documents', 0)}`"
        )
        st.sidebar.write(f"Failed documents: `{counts.get('failed_documents', 0)}`")

    st.sidebar.subheader("Vector Store")
    if status["vector_error"]:
        st.sidebar.warning(status["vector_error"])
    else:
        st.sidebar.write(f"Stored vectors: `{status.get('vector_count', 0)}`")
        if not status.get("vector_count"):
            st.sidebar.info("No vectors found yet. Build the vector store first.")

    if st.sidebar.button("Refresh status"):
        st.rerun()

    if st.sidebar.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()


def render_chat_history() -> None:
    """Render messages stored in session state."""

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant":
                if message.get("route"):
                    st.caption(f"Route: {message['route']} | Model: {message['model']}")
                render_sources(message.get("sources", []))
                render_context(message.get("context", ""))


def render_sources(sources: list[dict]) -> None:
    """Render retrieved source metadata for an assistant answer."""

    if not sources:
        return

    with st.expander("Retrieved sources"):
        for source in sources:
            normalized = normalize_source(source)
            st.markdown(
                f"**Source {normalized['rank']}** - "
                f"{normalized['entity']} ({normalized['entity_type']})"
            )
            st.write(f"URL: {normalized['source_url'] or 'Not available'}")
            st.write(f"Chunk ID: `{normalized['chunk_id']}`")
            st.write(f"Distance: `{normalized['distance']}`")
            st.caption(normalized["preview"])


def render_context(context: str) -> None:
    """Render the retrieved context used by the generator."""

    if not context:
        return

    with st.expander("Retrieved context used"):
        st.text(context)


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title=APP_TITLE, page_icon=":mag:")
    initialize_session_state()

    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    render_sidebar(get_system_status())
    render_chat_history()

    prompt = st.chat_input("Ask about a famous person or place")
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt}
    st.session_state["messages"].append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching local Wikipedia context..."):
            assistant_message = handle_user_query(prompt)
        st.markdown(assistant_message["content"])
        if assistant_message.get("route"):
            st.caption(
                f"Route: {assistant_message['route']} | "
                f"Model: {assistant_message['model']}"
            )
        render_sources(assistant_message.get("sources", []))
        render_context(assistant_message.get("context", ""))

    st.session_state["messages"].append(assistant_message)


def _friendly_runtime_message(exc: Exception) -> str:
    """Return a clear message for expected local runtime failures."""

    detail = str(exc).strip() or exc.__class__.__name__
    lower_detail = detail.casefold()

    if "ollama" in lower_detail:
        return (
            "I couldn't reach the local Ollama model. Make sure Ollama is "
            "running and the required models are pulled. "
            f"Details: {detail}"
        )

    if "chroma" in lower_detail or "vector" in lower_detail or "chromadb" in lower_detail:
        return (
            "The local vector store is not ready. Run ingestion, chunking, and "
            "vector-store build steps first. "
            f"Details: {detail}"
        )

    if "sqlite" in lower_detail or "database" in lower_detail:
        return (
            "The local metadata database is not ready. Run the setup or "
            "ingestion steps first. "
            f"Details: {detail}"
        )

    return (
        "I couldn't answer because the local RAG system is not ready. "
        f"Details: {detail}"
    )


if __name__ == "__main__":
    main()
