# Deployment

The trained models (`models/*.pkl`) are committed, so no training step is
needed on the host. Two supported targets:

---

## Option A — Hugging Face Spaces (recommended)

More RAM than Streamlit's free tier, so GPT-2 perplexity stays enabled.

1. Create a new Space: <https://huggingface.co/new-space> → **SDK: Streamlit**.
2. Push this repo to the Space (or connect the GitHub repo):

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```

3. Add this YAML header to the **top of the Space's `README.md`** (HF reads it;
   it is ignored by GitHub, so keep it only on the Space copy or in a separate
   commit):

   ```yaml
   ---
   title: AI-Generated Text Detection
   emoji: 🔍
   colorFrom: red
   colorTo: gray
   sdk: streamlit
   sdk_version: "1.63.0"
   app_file: app.py
   pinned: false
   ---
   ```

4. The Space builds from `requirements.txt` and starts `app.py` automatically.
   First boot downloads GPT-2 (~500 MB) — subsequent boots are cached.

---

## Option B — Streamlit Community Cloud

Free tier is ~1 GB RAM, which is tight for torch + GPT-2. Two choices:

- **Full app:** point it at `requirements.txt`. May OOM on first GPT-2 load; if
  so, switch to the lite path below.
- **Lite app (reliable):** in the Streamlit Cloud app settings set the
  requirements file to `requirements-lite.txt`. This drops torch/transformers;
  the app runs the two classical models + rule-based checker and shows a notice
  that perplexity analysis is unavailable (NFR-3 graceful degradation).

Steps:

1. <https://share.streamlit.io> → **New app** → pick this repo, branch `main`,
   main file `app.py`.
2. **Advanced settings → Python 3.12**, requirements file as above.
3. Deploy.

---

## Smoke test after deploy

- Paste ~150 words of obviously-AI text → expect **AI** with a high band.
- Paste a casual personal paragraph → expect **Human (confident)**.
- Upload a PDF/DOCX → expect the word count and a result.
- Open **Evidence breakdown** → Stage 1/2 scores, cliché flags, and
  perplexity/burstiness (or the "unavailable" note on the lite deploy).
