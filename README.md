# AI-Generated Text Detection System

Detects whether text is **human-written**, **raw AI-generated**, or **humanized
(paraphrased) AI**. Web app + classical-ML backend. Implements
[SRS_AI_Text_Detection.md](SRS_AI_Text_Detection.md).

## Architecture (SRS §6)

```
                 ┌───────────────── always-on evidence (FR-17) ─────────────────┐
input text ──┬──▶ rule-based style checker  (rules.py)      ─┐
             ├──▶ GPT-2 perplexity/burstiness (perplexity.py) ─┤
             │                                                 ├──▶ combined result
             └──▶ Stage 1: raw AI vs human  (model_raw.pkl) ───┤        (detector.py)
                        │ confident "AI"? ──▶ done (FR-18)      │
                        │ else "Human"  ──▶ Stage 2 (FR-19) ────┘
                          Stage 2: humanized AI vs human (model_humanized.pkl)
```

- **Trained components are classical only** — TF-IDF + Logistic Regression
  (SRS constraint §2.4). GPT-2 small is used *pre-trained*, for perplexity
  scoring only — never trained here.

## Layout

| Path | Role |
|---|---|
| `src/aitext/preprocess.py` | text normalization, sentence splitting (FR-2) |
| `src/aitext/datasets_build.py` | pull HC3 / MAGE via 🤗 `datasets`, write training CSVs (FR-1) |
| `src/aitext/train.py` | train + persist Stage 1 / Stage 2 (FR-3…FR-9) |
| `src/aitext/rules.py` | em-dash, cliché, sentence-uniformity checker (FR-10…FR-13) |
| `src/aitext/perplexity.py` | GPT-2 perplexity + burstiness (FR-14…FR-16) |
| `src/aitext/detector.py` | cascade orchestration + confidence bands (FR-17…FR-25) |
| `app.py` | Streamlit UI (FR-21…FR-25) |
| `config/cliches.txt` | maintained cliché list — edit without touching code (NFR-6) |
| `models/` | persisted `.pkl` artifacts (Appendix A) — gitignored |
| `data/` | prepared training CSVs — gitignored |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train the models

```bash
# 1. Build datasets (downloads HC3 the first time; needs internet)
python -m aitext.datasets_build --max-per-class 8000

#    For a trustworthy Stage 2, supply REAL paraphrased AI text instead of the
#    built-in pseudo-humanizer fallback:
#    python -m aitext.datasets_build --humanized-csv data/my_humanized.csv

# 2. Train + persist both stages (writes models/*.pkl)
python -m aitext.train
```

Run the module commands from `src/` on the path — either
`pip install -e .`-style, or:

```bash
PYTHONPATH=src python -m aitext.datasets_build
PYTHONPATH=src python -m aitext.train
```

## Run the app

```bash
streamlit run app.py
```

The app degrades gracefully (NFR-3) if models are missing — it shows the
rule-based + perplexity evidence and tells you to train.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

## Deployment (NFR-7)

Streamlit Community Cloud or Hugging Face Spaces. Commit `models/*.pkl` (or
retrain in a build step) since `data/` and `models/` are gitignored by default.
Needs ≥1 GB RAM to hold GPT-2 + the classical models (SRS §2.5).

## Known limitations (SRS §9)

- Well-humanized AI text frequently evades detection — industry-wide.
- False positives on unusually formal/simple text and non-native English.
- English only. No cryptographic watermark verification.
