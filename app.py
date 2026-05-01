"""Streamlit chat UI for the Local Wikipedia RAG Assistant."""

from __future__ import annotations

from collections.abc import MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Any
from uuid import uuid4

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
STOPPED_MESSAGE = "Generation stopped by user."
GENERATION_STATUS_TEXT = "Searching local Wikipedia context and generating answer..."
GENERATION_POLL_SECONDS = 0.5


def initialize_session_state(session_state: MutableMapping[str, Any] | None = None) -> None:
    """Initialize Streamlit session state keys used by the chat UI."""

    state = session_state if session_state is not None else st.session_state
    if "messages" not in state:
        state["messages"] = []
    if "active_request_id" not in state:
        state["active_request_id"] = None
    if "active_future_request_id" not in state:
        state["active_future_request_id"] = None
    if "active_future" not in state:
        state["active_future"] = None
    if "active_prompt" not in state:
        state["active_prompt"] = None
    if "stop_requested" not in state:
        state["stop_requested"] = False
    if "is_generating" not in state:
        state["is_generating"] = False


@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    """Return a shared single-worker executor for background generation."""

    return ThreadPoolExecutor(max_workers=1)


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


def build_stopped_message() -> dict:
    """Build the assistant message shown when a request is stopped."""

    return {
        "role": "assistant",
        "content": STOPPED_MESSAGE,
        "route": None,
        "sources": [],
        "context": "",
        "model": config.OLLAMA_GENERATION_MODEL,
        "error": False,
        "stopped": True,
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


def is_generation_active(session_state: MutableMapping[str, Any] | None = None) -> bool:
    """Return whether a generation request is currently active."""

    state = _get_state(session_state)
    return bool(state.get("is_generating") and state.get("active_future") is not None)


def start_generation(
    prompt: str,
    session_state: MutableMapping[str, Any] | None = None,
    executor: ThreadPoolExecutor | None = None,
    query_handler=handle_user_query,
) -> bool:
    """Start a background generation request if no request is active."""

    state = _get_state(session_state)
    initialize_session_state(state)

    clean_prompt = prompt.strip()
    if not clean_prompt or is_generation_active(state):
        return False

    request_id = uuid4().hex
    active_executor = executor or get_executor()
    future = active_executor.submit(query_handler, clean_prompt)

    state["messages"].append({"role": "user", "content": clean_prompt})
    state["active_request_id"] = request_id
    state["active_future_request_id"] = request_id
    state["active_future"] = future
    state["active_prompt"] = clean_prompt
    state["stop_requested"] = False
    state["is_generating"] = True
    return True


def stop_generation(session_state: MutableMapping[str, Any] | None = None) -> bool:
    """Mark the active generation request as stopped and unblock the UI."""

    state = _get_state(session_state)
    initialize_session_state(state)

    if not is_generation_active(state):
        return False

    future = state.get("active_future")
    if future is not None and hasattr(future, "cancel"):
        try:
            future.cancel()
        except Exception:
            pass

    state["stop_requested"] = True
    state["messages"].append(build_stopped_message())
    _clear_generation_state(state, stop_requested=True)
    return True


def finish_generation_if_ready(
    session_state: MutableMapping[str, Any] | None = None,
) -> bool:
    """Append a completed background answer if the active request is still valid."""

    state = _get_state(session_state)
    initialize_session_state(state)
    future = state.get("active_future")

    if not is_generation_active(state) or future is None or not future.done():
        return False

    active_request_id = state.get("active_request_id")
    future_request_id = state.get("active_future_request_id")
    stop_requested = bool(state.get("stop_requested"))

    if stop_requested or active_request_id != future_request_id:
        _clear_generation_state(state, stop_requested=False)
        return True

    try:
        assistant_message = future.result()
    except Exception as exc:
        assistant_message = error_to_chat_message(exc)

    state["messages"].append(assistant_message)
    _clear_generation_state(state, stop_requested=False)
    return True


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
        _clear_generation_state(st.session_state, stop_requested=False)
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


def render_generation_controls() -> None:
    """Render generation status and stop control while a request is active."""

    if not is_generation_active():
        return

    st.info(GENERATION_STATUS_TEXT)
    if st.button("Stop generation"):
        stop_generation()
        st.rerun()

    time.sleep(GENERATION_POLL_SECONDS)
    st.rerun()


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title=APP_TITLE, page_icon=":mag:")
    initialize_session_state()

    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    render_sidebar(get_system_status())
    if finish_generation_if_ready():
        st.rerun()

    render_chat_history()
    generation_active = is_generation_active()
    if generation_active:
        render_generation_controls()

    prompt = st.chat_input(
        "Ask about a famous person or place",
        disabled=generation_active,
    )
    if not prompt:
        return

    if not start_generation(prompt):
        st.warning("Please wait for the current answer to finish or stop it first.")
        return
    st.rerun()


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


def _get_state(
    session_state: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Return explicit state or Streamlit session state."""

    return session_state if session_state is not None else st.session_state


def _clear_generation_state(
    session_state: MutableMapping[str, Any],
    stop_requested: bool,
) -> None:
    """Clear active generation fields while preserving chat history."""

    session_state["active_request_id"] = None
    session_state["active_future_request_id"] = None
    session_state["active_future"] = None
    session_state["active_prompt"] = None
    session_state["is_generating"] = False
    session_state["stop_requested"] = stop_requested


if __name__ == "__main__":
    main()
