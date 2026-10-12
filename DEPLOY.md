# Deployment

The trained models (`models/*.pkl`) are committed, so no training step is
needed on the host. The app is one Python process (FastAPI + uvicorn) that also
serves the pre-built React UI from `web/dist`.

> The previous Streamlit-specific targets (Hugging Face Streamlit SDK, Streamlit
> Community Cloud) no longer apply. Any host that can run Python and has Node
> available at build time works.

## Build + run

```bash
pip install -r requirements.txt        # or requirements-lite.txt, see below
(cd web && npm ci && npm run build)    # produces web/dist
PYTHONPATH=src python -m uvicorn aitext.api:app --host 0.0.0.0 --port 8000
```

- First start with the full requirements downloads GPT-2 (~500 MB); later starts
  use the cache.
- Uploads are capped at 10 MB (enforced in `src/aitext/api.py`).
- `GET /api/health` returns `{"ok": true, "models_ready": true}` for health checks.
- The `/api` routes are unauthenticated; put the service behind your platform's
  access control if it shouldn't be public.

## Memory-constrained hosts

torch + GPT-2 needs roughly 1–2 GB RAM. On smaller hosts install
`requirements-lite.txt` instead. This drops torch/transformers; the app runs the
two classical models + the rule-based checker and shows a notice that perplexity
analysis is unavailable (NFR-3 graceful degradation).

## Smoke test after deploy

- Paste ~150 words of obviously-AI text → expect **AI** with a high band.
- Paste a casual personal paragraph → expect **Human (confident)**.
- Attach a PDF/DOCX → expect the word count in the `> check` line and a result.
- Check the style/fluency block shows cliché flags and perplexity/burstiness (or
  the "unavailable" note on the lite install).
