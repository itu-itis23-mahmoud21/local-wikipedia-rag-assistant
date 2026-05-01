# Demo and Submission Checklist

## Required Repo Files

- `README.md`
- `product_prd.md`
- `recommendation.md`
- `DEMO_CHECKLIST.md`
- `requirements.txt`
- `app.py`
- `src/`
- `scripts/`
- `tests/`

## Commands to Show in Demo

```powershell
ollama list
python -m unittest discover -v
python -m compileall -q app.py src scripts tests
python scripts/setup_all.py --limit 5 --reset-db --reset-chunks --reset-collection
streamlit run app.py
```

Optional reset command:

```powershell
python scripts/reset_system.py --yes
```

## Five-Minute Demo Flow

1. System overview
   - Explain local Wikipedia RAG and no external LLM APIs.
2. Show repo structure
   - Point to `src/`, `scripts/`, `tests/`, and docs.
3. Show Ollama models
   - Run `ollama list` and show `llama3.2:3b` plus `nomic-embed-text`.
4. Run or explain setup_all
   - Use `--limit 5` for a quick demo if needed.
5. Start Streamlit app
   - Run `streamlit run app.py`.
6. Ask a people question
   - Example: `Who was Albert Einstein and what is he known for?`
7. Ask a place question
   - Example: `Where is the Eiffel Tower located?`
8. Ask a mixed or comparison question
   - Example: `Compare Albert Einstein and Nikola Tesla.`
9. Ask a failure question
   - Example: `Who is the president of Mars?`
   - Explain the expected grounded behavior.
10. Explain tradeoffs and improvements
   - Mention fixed entity list, local model limits, simple routing, and future
     evaluation improvements.

## Final Submission Checklist

- GitHub repo is public.
- README is complete.
- `product_prd.md` is complete.
- `recommendation.md` is complete.
- Demo video link is added to README.
- Tests pass with `python -m unittest discover -v`.
- Compile check passes with `python -m compileall -q app.py src scripts tests`.
- Generated `data/` directory is not committed.
- Generated `chroma_db/` directory is not committed.
- `.env`, local databases, logs, and secrets are not committed.
