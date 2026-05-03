# 🤖 Local Wikipedia RAG Assistant

A local ChatGPT-style Retrieval-Augmented Generation assistant for famous people
and famous places. The app ingests Wikipedia pages, stores local metadata,
creates embeddings with Ollama, indexes chunks in Chroma, and answers questions
through a Streamlit chat interface.

This project was built for **HW3 - AI Aided Computer Engineering**.

---

## ✨ What It Does

- Answers questions about **25 famous people** and **25 famous places**
- Uses Wikipedia as the source dataset
- Runs generation and embeddings locally with **Ollama**
- Stores metadata in **SQLite**
- Stores vectors in **sharded Chroma collections**
- Routes questions as person, place, both, or unknown
- Shows retrieved sources and retrieved context
- Supports copy buttons, chat export, and local system status
- Returns **"I don't know."** when retrieved context does not support an answer

---

## 🧱 Architecture

```text
Wikipedia API
  -> raw local files
  -> processed local files
  -> SQLite metadata
  -> document chunks
  -> Ollama embeddings
  -> sharded Chroma vector store
  -> retriever
  -> Ollama LLM
  -> Streamlit UI
```

---

## 🛠 Tech Stack

- Python 3.10+
- Streamlit
- Ollama
- `llama3.2:3b` for local answer generation
- `nomic-embed-text` for local embeddings
- ChromaDB
- SQLite
- Requests
- Python `unittest`

---

## 🔒 Local-Only Guarantee

No external LLM API is used.

- Ollama runs the language model locally.
- Ollama generates embeddings locally.
- SQLite and Chroma data stay on your machine.
- Wikipedia is contacted only during ingestion.
- Prompts, retrieved context, embeddings, and answers are not sent to external
  LLM services.

---

## ✅ Prerequisites

Install these before running the project:

- Python 3.10 or newer
- Git
- Ollama
- Enough disk space for Wikipedia text, SQLite metadata, and Chroma vectors

Check Ollama:

```powershell
ollama --version
ollama list
```

---

## 📦 Install Dependencies

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧠 Run the Local Models

Start Ollama, then pull the required models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Keep Ollama running while building vectors and using the app.

Useful checks:

```powershell
ollama list
ollama ps
```

---

## 🗂 Build the Local Dataset

The easiest path is the full setup script:

```powershell
python scripts/setup_all.py --force-ingest --reset-db --reset-chunks --reset-collection
```

For a fast smoke test, build only a few entities:

```powershell
python scripts/setup_all.py --limit 5 --force-ingest --reset-db --reset-chunks --reset-collection
```

The setup pipeline runs:

1. Wikipedia ingestion
2. Document chunking
3. Ollama embedding generation
4. Chroma vector-store build

---

## 🔁 Manual Pipeline

Use these commands if you want to run each stage separately.

### 1. Ingest Wikipedia pages

```powershell
python scripts/ingest_wikipedia.py
```

Useful options:

```powershell
python scripts/ingest_wikipedia.py --limit 5
python scripts/ingest_wikipedia.py --entity-type person
python scripts/ingest_wikipedia.py --force
python scripts/ingest_wikipedia.py --reset-db
```

### 2. Chunk processed documents

```powershell
python scripts/chunk_documents.py --reset-chunks
```

### 3. Build the vector store

Recommended command:

```powershell
python scripts/build_vector_store.py --reset-chroma-dir --shard-count 10 --batch-size 50 --progress-every 25 --post-build-settle-seconds 20
```

Useful options:

```powershell
python scripts/build_vector_store.py --limit 100
python scripts/build_vector_store.py --entity-type place
python scripts/build_vector_store.py --skip-gpu-check
python scripts/build_vector_store.py --skip-post-build-health-check
```

---

## 🚀 Start the Application

Run Streamlit:

```powershell
streamlit run app.py
```

Then open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

---

## 💬 Example Queries

### People

- Who was Albert Einstein?
- Who is Einstein?
- What did Marie Curie discover?
- Why is Mohamed Salah famous?
- Which person is associated with electricity?
- Compare Albert Einstein and Nikola Tesla.
- Compare Lionel Messi and Cristiano Ronaldo.

### Places

- Where is the Eiffel Tower located?
- Which famous place is located in Turkey?
- Which famous place is in Egypt?
- Which famous place is in Paris?
- Which landmark is in New York?
- What was the Colosseum used for?
- Compare the Eiffel Tower and the Statue of Liberty.

### Unsupported / Failure Behavior

- Who is the president of Mars?
- Tell me about a random unknown person John Doe.

Expected behavior for unsupported questions:

```text
I don't know.
```

---

## 🧪 Run Tests

Run the full unit test suite:

```powershell
python -m unittest discover -v
```

Run a compile check:

```powershell
python -m compileall -q app.py src scripts tests
```

---

## ♻️ Reset Generated Data

Remove generated local data and Chroma files:

```powershell
python scripts/reset_system.py --yes
```

Remove only local data:

```powershell
python scripts/reset_system.py --yes --data-only
```

Remove only Chroma vectors:

```powershell
python scripts/reset_system.py --yes --chroma-only
```

Generated artifacts such as `data/`, `chroma_db/`, SQLite databases, logs, and
virtual environments are ignored by Git.

---

## 📁 Repository Structure

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
|   |-- generator.py
|   `-- utils.py
|-- scripts/
|   |-- ingest_wikipedia.py
|   |-- chunk_documents.py
|   |-- build_vector_store.py
|   |-- setup_all.py
|   `-- reset_system.py
`-- tests/
```

---

## 🧯 Troubleshooting

### Ollama is not running

Start Ollama and check:

```powershell
ollama list
```

### Models are missing

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Chroma/vector store is empty

Rebuild local data:

```powershell
python scripts/setup_all.py --force-ingest --reset-db --reset-chunks --reset-collection
```

For a cleaner vector rebuild:

```powershell
python scripts/build_vector_store.py --reset-chroma-dir --shard-count 10 --batch-size 50 --progress-every 25
```

### Wikipedia ingestion fails

- Check internet access.
- Retry with:

```powershell
python scripts/ingest_wikipedia.py --force
```

### Streamlit cannot start

- Activate the virtual environment.
- Reinstall dependencies:

```powershell
pip install -r requirements.txt
```

### GPU usage is unclear

Python does not force Ollama to use GPU. Ollama decides CPU/GPU execution.

Useful checks:

```powershell
nvidia-smi
ollama ps
```

---

## 🎬 Demo Video

Demo video link: [Youtube Link - Unlisted](https://youtu.be/vWHUtc4PYHs)
