"""Streamlit chat UI for the Local Wikipedia RAG Assistant."""

from __future__ import annotations

from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
import html
import json
import random
from typing import Any
from uuid import uuid4

import streamlit as st

from src import config
from src.database import MetadataDB
from src.entities import get_people, get_places
from src.generator import GeneratedAnswer, OllamaAnswerGenerator
from src.vector_store import ChromaVectorStore


SessionStateMapping = MutableMapping[Any, Any]

APP_TITLE = "Local Wikipedia RAG Assistant"
APP_CAPTION = (
    "Ask questions about famous people and famous places using locally ingested "
    "Wikipedia data."
)
STOPPED_MESSAGE = "Generation stopped by user."
GENERATION_STATUS_TEXT = "Searching local Wikipedia context and generating answer..."
GENERATION_POLL_SECONDS = 0.5
EXPORT_FILENAME = "local_wikipedia_rag_chat_export.txt"
BUSY_COMPOSER_TEXT = "Assistant is generating. Stop the current response to ask something else."
CHAT_INPUT_KEY = "main_chat_input"
RANDOM_PROMPT_BUTTON_KEY = "random_prompt_button"


def initialize_session_state(session_state: SessionStateMapping | None = None) -> None:
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
    if "last_random_prompt" not in state:
        state["last_random_prompt"] = None


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


def format_status_metric_value(value: object) -> str:
    """Format sidebar metric values consistently."""

    if value is None:
        return "0"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    if isinstance(value, str):
        try:
            return f"{int(value):,}"
        except ValueError:
            return value
    return str(value)


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


def format_sources_for_copy(sources: list[dict]) -> str:
    """Format retrieved source metadata as copyable plain text."""

    if not sources:
        return "No retrieved sources available."

    sections: list[str] = []
    for source in sources:
        normalized = normalize_source(source)
        sections.append(
            "\n".join(
                [
                    f"Source {normalized['rank']}",
                    f"Entity: {normalized['entity'] or 'Not available'}",
                    f"Type: {normalized['entity_type'] or 'Not available'}",
                    f"URL: {normalized['source_url'] or 'Not available'}",
                    f"Chunk ID: {normalized['chunk_id']}",
                    f"Distance: {normalized['distance']}",
                    f"Preview: {normalized['preview']}",
                ]
            ).strip()
        )

    return "\n\n".join(sections).strip()


def format_context_for_copy(context: str) -> str:
    """Format retrieved context as copyable plain text."""

    return str(context or "").strip()


def format_entity_list_for_sidebar(entity_names: list[str]) -> str:
    """Format configured entity names for the sidebar expanders."""

    if not entity_names:
        return "_No configured entities available._"
    return "\n".join(
        f"{index}. {entity_name}" for index, entity_name in enumerate(entity_names, start=1)
    )


def build_random_prompt(
    entity_type: str | None = None,
    entity_name: str | None = None,
) -> str:
    """Build a random starter question for one configured person or place."""

    selected_type = entity_type or random.choice(("person", "place"))
    if selected_type == "person":
        selected_name = entity_name or random.choice(get_people())
        return f"Who is {selected_name}?"
    if selected_type == "place":
        selected_name = entity_name or random.choice(get_places())
        return f"Where is {selected_name} located?"
    raise ValueError("entity_type must be 'person' or 'place'")


def set_random_prompt_in_chat_input(
    session_state: SessionStateMapping | None = None,
    prompt: str | None = None,
) -> str:
    """Set the main chat input text to a random starter prompt."""

    state = _get_state(session_state)
    initialize_session_state(state)
    selected_prompt = prompt or build_random_prompt()
    state[CHAT_INPUT_KEY] = selected_prompt
    state["last_random_prompt"] = selected_prompt
    return selected_prompt


def build_chat_export_text(
    messages: list[dict],
    include_user_questions: bool = True,
    include_answers: bool = True,
    include_sources: bool = False,
    include_context: bool = False,
    include_metadata: bool = False,
) -> str:
    """Build a readable TXT export of the chat history."""

    del include_answers  # Answers are mandatory and always included.

    header = "Local Wikipedia RAG Assistant Chat Export"
    if not messages:
        return f"{header}\n\nNo chat messages are available."

    sections: list[str] = []
    for message in messages:
        role = message.get("role")

        if role == "user":
            if not include_user_questions:
                continue
            section_parts = [
                "User question:",
                str(message.get("content") or "").strip(),
            ]
        elif role == "assistant":
            section_parts = [
                "Assistant answer:",
                str(message.get("content") or "").strip(),
            ]

            if include_sources and message.get("sources"):
                section_parts.extend(
                    [
                        "",
                        "Retrieved sources:",
                        format_sources_for_copy(message.get("sources", [])),
                    ]
                )

            if include_context and message.get("context"):
                section_parts.extend(
                    [
                        "",
                        "Retrieved context used:",
                        format_context_for_copy(message.get("context", "")),
                    ]
                )
        else:
            continue

        if include_metadata:
            section_parts.extend(["", "Metadata:", _format_message_metadata(message)])

        sections.append("\n".join(section_parts).strip())

    if not sections:
        return f"{header}\n\nNo chat messages are available."

    separator = "\n\n" + ("-" * 60) + "\n\n"
    return f"{header}\n\n{separator.join(sections)}".strip()


def get_export_filename() -> str:
    """Return the default TXT filename for chat export."""

    return EXPORT_FILENAME


def format_assistant_caption(message: dict) -> str:
    """Build the compact assistant metadata caption."""

    parts: list[str] = []
    if message.get("route"):
        parts.append(f"Route: {message['route']}")
    if message.get("model"):
        parts.append(f"Model: {message['model']}")
    return " · ".join(parts)


def _format_message_metadata(message: dict) -> str:
    """Format optional message metadata for TXT export."""

    metadata_lines = [f"Role: {message.get('role', '')}"]
    if "route" in message:
        metadata_lines.append(f"Route: {message.get('route')}")
    if "model" in message:
        metadata_lines.append(f"Model: {message.get('model')}")
    if "error" in message:
        metadata_lines.append(f"Error: {message.get('error')}")
    if "stopped" in message:
        metadata_lines.append(f"Stopped: {message.get('stopped')}")
    return "\n".join(metadata_lines)


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


def is_generation_active(session_state: SessionStateMapping | None = None) -> bool:
    """Return whether a generation request is currently active."""

    state = _get_state(session_state)
    return bool(state.get("is_generating") and state.get("active_future") is not None)


def start_generation(
    prompt: str,
    session_state: SessionStateMapping | None = None,
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


def stop_generation(session_state: SessionStateMapping | None = None) -> bool:
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
    session_state: SessionStateMapping | None = None,
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


def render_copy_button(label: str, text: str, key: str) -> None:
    """Render a small browser clipboard copy button."""

    copy_text = str(text or "")
    if not copy_text:
        return

    safe_key = "".join(character if character.isalnum() else "_" for character in key)
    if not safe_key:
        safe_key = uuid4().hex
    button_id = f"copy_button_{safe_key}"
    status_id = f"copy_status_{safe_key}"
    label_html = html.escape(label)

    component_html = f"""
    <!doctype html>
    <html>
    <head>
    <style>
    html, body {{
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: transparent;
      color-scheme: light dark;
    }}
    .copy-row {{
      box-sizing: border-box;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      height: 40px;
      font-family: sans-serif;
      color: CanvasText;
    }}
    .copy-row button {{
      padding: 0.35rem 0.65rem;
      border: 1px solid #d0d7de;
      border-radius: 6px;
      background: transparent;
      cursor: pointer;
      color: inherit;
      line-height: 1.1;
    }}
    .copy-row span {{
      font-size: 0.85rem;
      color: inherit;
      opacity: 0.72;
      line-height: 1;
    }}
    </style>
    </head>
    <body>
    <div class="copy-row">
      <button id="{button_id}" type="button">
        {label_html}
      </button>
      <span id="{status_id}"></span>
    </div>
    <script>
    const copyButton = document.getElementById({json.dumps(button_id)});
    const copyStatus = document.getElementById({json.dumps(status_id)});
    const copyText = {json.dumps(copy_text)};
    copyButton.addEventListener("click", async () => {{
      if (!navigator.clipboard) {{
        copyStatus.textContent = "Clipboard unavailable";
        return;
      }}
      try {{
        await navigator.clipboard.writeText(copyText);
        copyStatus.textContent = "Copied";
      }} catch (error) {{
        copyStatus.textContent = "Copy unavailable";
      }}
    }});
    </script>
    </body>
    </html>
    """
    _render_html_iframe(component_html, height=42)


def _render_html_iframe(markup: str, height: int) -> None:
    """Render inline HTML in an iframe without using deprecated Streamlit APIs."""

    st.iframe(markup, height=height)


def inject_custom_css() -> None:
    """Inject small local CSS improvements for the Streamlit UI."""

    st.markdown(
        """
        <style>
        .app-header {
            border: 1px solid #d0d7de;
            border-left: 6px solid #ffffff;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            background: transparent;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0 0 0.25rem 0;
            color: inherit;
        }
        .app-subtitle {
            color: inherit;
            opacity: 0.72;
            font-size: 1rem;
            margin: 0 0 0.85rem 0;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .app-badge {
            border: 1px solid #d0d7de;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            background: transparent;
            color: inherit;
            opacity: 0.78;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .source-card {
            border: 1px solid #d0d7de;
            border-left: 3px solid #d0d7de;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            margin: 0.65rem 0;
            background: transparent;
        }
        .source-title {
            font-weight: 700;
            color: inherit;
            margin-bottom: 0.35rem;
        }
        .source-meta {
            color: inherit;
            opacity: 0.72;
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .source-preview {
            border-top: 1px solid #eaeef2;
            margin-top: 0.55rem;
            padding-top: 0.55rem;
            color: inherit;
            opacity: 0.82;
            font-size: 0.9rem;
        }
        .status-note {
            color: inherit;
            opacity: 0.72;
            font-size: 0.85rem;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1 !important;
        }
        [data-testid="stChatMessageAvatarUser"] {
            background: rgba(208, 215, 222, 0.14) !important;
            color: rgba(255, 255, 255, 0.88) !important;
            box-shadow: 0 0 12px rgba(208, 215, 222, 0.18) !important;
        }
        [data-testid="stChatMessageAvatarAssistant"] {
            background: rgba(255, 75, 75, 0.20) !important;
            color: #ff8f84 !important;
            box-shadow: 0 0 18px rgba(255, 75, 75, 0.22) !important;
        }
        [data-testid="stChatMessageAvatarUser"] svg,
        [data-testid="stChatMessageAvatarAssistant"] svg,
        [data-testid="stChatMessageAvatarUser"] span,
        [data-testid="stChatMessageAvatarAssistant"] span {
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            min-height: 3rem;
            white-space: normal;
            line-height: 1.2;
            align-items: center;
            justify-content: center;
        }
        div.st-key-generation_status_bar {
            padding: 0;
            margin: 0.35rem 0 1rem;
            position: relative;
        }
        div.st-key-generation_status_bar [data-testid="stMarkdownContainer"] {
            margin: 0;
        }
        .generation-status-shell {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            background: transparent;
            color: inherit;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.55rem;
            font-size: 0.95rem;
            height: 3.2rem;
            padding: 0 0.8rem;
            box-sizing: border-box;
        }
        .generation-status-text {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            min-width: 0;
            line-height: 1.35;
        }
        .generation-status-label {
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.35;
        }
        .generation-spinner {
            width: 1rem;
            height: 1rem;
            border: 2px solid #d0d7de;
            border-top-color: #0969da;
            border-radius: 50%;
            animation: rag-spin 0.85s linear infinite;
            flex: 0 0 auto;
        }
        .generation-stop-visual {
            width: 2rem;
            height: 2rem;
            border-radius: 0.55rem;
            background: #ff4b4b;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
        }
        .generation-stop-visual::before {
            content: "";
            width: 0.72rem;
            height: 0.72rem;
            background: #ffffff;
            border-radius: 0.18rem;
        }
        @keyframes rag-spin {
            to { transform: rotate(360deg); }
        }
        div.st-key-stop_generation_in_status {
            position: absolute !important;
            top: 50% !important;
            right: 0.8rem !important;
            transform: translateY(-50%) !important;
            z-index: 3;
            width: 2rem !important;
            height: 2rem !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        div.st-key-stop_generation_in_status button {
            width: 2rem !important;
            min-width: 2rem !important;
            height: 2rem !important;
            min-height: 2rem !important;
            padding: 0 !important;
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            opacity: 0 !important;
        }
        div.st-key-main_chat_input,
        div[class*="e15xmbo00"]:has(textarea) {
            margin-right: 3.75rem !important;
        }
        div.st-key-random_prompt_button {
            position: fixed !important;
            bottom: 4.1rem !important;
            right: 1.5rem !important;
            z-index: 1002;
            width: 2.35rem !important;
            height: 2.35rem !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        div.st-key-random_prompt_button button {
            width: 2.35rem !important;
            min-width: 2.35rem !important;
            height: 2.35rem !important;
            min-height: 2.35rem !important;
            padding: 0 !important;
            border-radius: 999px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0 !important;
            line-height: 1 !important;
            position: relative !important;
            box-shadow: none !important;
        }
        div.st-key-random_prompt_button button p {
            display: none !important;
            margin: 0 !important;
            width: 0 !important;
        }
        div.st-key-random_prompt_button button svg,
        div.st-key-random_prompt_button button span {
            position: absolute !important;
            left: 50% !important;
            top: 50% !important;
            transform: translate(-50%, -50%) !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 1.55rem !important;
            height: 1.55rem !important;
            font-size: 1.55rem !important;
            line-height: 1 !important;
        }
        div.st-key-random_prompt_button button svg {
            transform: translate(-50%, -50%) scale(1.18) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    """Render the polished app header."""

    badges = ["Local-only", "Wikipedia RAG", "Ollama", "Chroma + SQLite"]
    badge_html = "".join(
        f'<span class="app-badge">{html.escape(badge)}</span>' for badge in badges
    )
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-title">{html.escape(APP_TITLE)}</div>
          <div class="app-subtitle">{html.escape(APP_CAPTION)}</div>
          <div class="badge-row">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(status: dict) -> None:
    """Render the status sidebar."""

    st.sidebar.header("Local System")

    st.sidebar.subheader("Models")
    st.sidebar.write(f"Generation: `{status['generation_model']}`")
    st.sidebar.write(f"Embeddings: `{status['embedding_model']}`")

    st.sidebar.subheader("Storage")
    st.sidebar.write(f"Collection: `{status['collection_name']}`")
    st.sidebar.caption(f"SQLite: {status['sqlite_db_path']}")
    st.sidebar.caption(f"Chroma: {status['chroma_db_path']}")

    st.sidebar.subheader("Dataset / Status")
    render_status_metrics(status)

    st.sidebar.subheader("Actions")
    action_columns = st.sidebar.columns(2)
    with action_columns[0]:
        if st.button("Refresh status", use_container_width=True):
            st.rerun()
    with action_columns[1]:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            _clear_generation_state(st.session_state, stop_requested=False)
            st.rerun()

    st.sidebar.subheader("Export")
    render_chat_export_panel()

    render_entity_catalog()


def render_status_metrics(status: dict) -> None:
    """Render compact sidebar metrics for local data readiness."""

    counts = status.get("counts", {})
    metric_columns = st.sidebar.columns(2)

    with metric_columns[0]:
        st.metric("Entities", format_status_metric_value(counts.get("entities", 0)))
        st.metric("Chunks", format_status_metric_value(counts.get("chunks", 0)))
    with metric_columns[1]:
        st.metric("Documents", format_status_metric_value(counts.get("documents", 0)))
        st.metric("Vectors", format_status_metric_value(status.get("vector_count", 0)))

    if status["database_error"]:
        st.sidebar.warning(status["database_error"])
    else:
        st.sidebar.caption(
            "People: "
            f"{format_status_metric_value(counts.get('people', 0))} · "
            "Places: "
            f"{format_status_metric_value(counts.get('places', 0))}"
        )
        st.sidebar.caption(
            "Successful docs: "
            f"{format_status_metric_value(counts.get('successful_documents', 0))} · "
            "Failed docs: "
            f"{format_status_metric_value(counts.get('failed_documents', 0))}"
        )

    if status["vector_error"]:
        st.sidebar.warning(status["vector_error"])
    elif not status.get("vector_count"):
        st.sidebar.info("No vectors found yet. Build the vector store first.")


def render_chat_export_panel() -> None:
    """Render a sidebar export panel for downloading chat history."""

    popover = getattr(st.sidebar, "popover", None)
    if popover is not None:
        with popover("Export chat", use_container_width=True):
            _render_chat_export_options()
    else:
        with st.sidebar.expander("Export chat"):
            _render_chat_export_options()


def _render_chat_export_options() -> None:
    """Render export checkboxes and TXT download button."""

    st.checkbox("Answers", value=True, disabled=True, key="export_include_answers")
    include_user_questions = st.checkbox(
        "User questions",
        value=True,
        key="export_include_user_questions",
    )
    include_sources = st.checkbox(
        "Retrieved sources",
        value=False,
        key="export_include_sources",
    )
    include_context = st.checkbox(
        "Retrieved context used",
        value=False,
        key="export_include_context",
    )
    include_metadata = st.checkbox(
        "Metadata",
        value=False,
        key="export_include_metadata",
    )

    export_text = build_chat_export_text(
        st.session_state.get("messages", []),
        include_user_questions=include_user_questions,
        include_answers=True,
        include_sources=include_sources,
        include_context=include_context,
        include_metadata=include_metadata,
    )
    st.download_button(
        "Download TXT",
        data=export_text,
        file_name=get_export_filename(),
        mime="text/plain",
    )


def render_entity_catalog() -> None:
    """Render configured people and places as sidebar reference lists."""

    people = get_people()
    places = get_places()

    st.sidebar.subheader("Configured Wikipedia Entities")
    st.sidebar.caption(
        "These are the local dataset targets. After running setup, their "
        "Wikipedia pages are stored locally, chunked, embedded, and indexed for RAG."
    )

    with st.sidebar.expander(f"People ({len(people)})"):
        st.markdown(format_entity_list_for_sidebar(people))

    with st.sidebar.expander(f"Places ({len(places)})"):
        st.markdown(format_entity_list_for_sidebar(places))


def render_chat_history() -> None:
    """Render messages stored in session state."""

    for index, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant":
                assistant_caption = format_assistant_caption(message)
                if assistant_caption:
                    st.caption(assistant_caption)
                render_sources(message.get("sources", []), key=f"sources_{index}")
                render_context(message.get("context", ""), key=f"context_{index}")


def render_sources(sources: list[dict], key: str = "sources") -> None:
    """Render retrieved source metadata for an assistant answer."""

    if not sources:
        return

    with st.expander("Retrieved sources"):
        render_copy_button("Copy sources", format_sources_for_copy(sources), key)
        for source in sources:
            render_source_card(source)


def render_source_card(source: dict) -> None:
    """Render one scannable retrieved-source card."""

    normalized = normalize_source(source)
    rank = html.escape(str(normalized["rank"]))
    entity = html.escape(normalized["entity"] or "Not available")
    entity_type = html.escape(normalized["entity_type"] or "Not available")
    source_url = html.escape(normalized["source_url"] or "Not available")
    chunk_id = html.escape(str(normalized["chunk_id"]))
    distance = html.escape(str(normalized["distance"]))
    preview = html.escape(normalized["preview"] or "")

    st.markdown(
        f"""
        <div class="source-card">
          <div class="source-title">Source {rank} · {entity} ({entity_type})</div>
          <div class="source-meta">
            URL: {source_url}<br>
            Chunk ID: <code>{chunk_id}</code><br>
            Distance: <code>{distance}</code>
          </div>
          <div class="source-preview">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context(context: str, key: str = "context") -> None:
    """Render the retrieved context used by the generator."""

    if not context:
        return

    with st.expander("Retrieved context used"):
        render_copy_button("Copy context", format_context_for_copy(context), key)
        st.text(context)


def _render_generation_controls_body() -> None:
    """Render generation status and stop control while a request is active."""

    if not is_generation_active():
        return

    if finish_generation_if_ready():
        st.rerun()

    render_generation_status()


def render_generation_status() -> None:
    """Render a stable loading indicator with an inline stop action."""

    with st.container(key="generation_status_bar"):
        st.markdown(
            f"""
            <div class="generation-status-shell">
              <div class="generation-status-text">
                <span class="generation-spinner"></span>
                <span class="generation-status-label">
                  {html.escape(GENERATION_STATUS_TEXT)}
                </span>
              </div>
              <div class="generation-stop-visual" aria-hidden="true"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Stop",
            key="stop_generation_in_status",
            help="Stop generation",
        ):
            stop_generation()
            st.rerun()


if hasattr(st, "fragment"):
    render_generation_controls = st.fragment(run_every=GENERATION_POLL_SECONDS)(
        _render_generation_controls_body
    )
else:
    render_generation_controls = _render_generation_controls_body


def render_busy_chat_composer() -> None:
    """Render the normal composer locked while generation is active."""

    st.chat_input(
        BUSY_COMPOSER_TEXT,
        disabled=True,
        key="busy_chat_input",
    )


def render_random_prompt_button() -> None:
    """Render a dice button that fills the chat input with a random starter."""

    if st.button(
        "Random prompt",
        icon=":material/playing_cards:",
        key=RANDOM_PROMPT_BUTTON_KEY,
        help="Generate a random question",
        type="primary",
    ):
        set_random_prompt_in_chat_input()
        st.rerun()


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title=APP_TITLE, page_icon=":mag:", layout="wide")
    initialize_session_state()
    inject_custom_css()
    render_app_header()

    render_sidebar(get_system_status())
    if finish_generation_if_ready():
        st.rerun()

    render_chat_history()
    generation_active = is_generation_active()
    if generation_active:
        render_generation_controls()
        render_busy_chat_composer()
        return

    render_random_prompt_button()
    prompt = st.chat_input(
        "Ask about a famous person or place",
        key=CHAT_INPUT_KEY,
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
    session_state: SessionStateMapping | None = None,
) -> SessionStateMapping:
    """Return explicit state or Streamlit session state."""

    return session_state if session_state is not None else st.session_state


def _clear_generation_state(
    session_state: SessionStateMapping,
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
