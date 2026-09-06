# AI-Generated Text Detection — Project Report

**Subject:** AIML | **Deliverable:** working system + evaluation
**Spec:** [`SRS_AI_Text_Detection (1).md`](SRS_AI_Text_Detection%20(1).md)

---

## 1. Problem

Given a piece of English text, decide whether it is:

- **human-written**,
- **raw AI-generated** (straight LLM output), or
- **humanized AI** — LLM output that has been paraphrased to evade detection.

Constraint from the SRS: **classical ML only** for anything we train. A
pre-trained GPT-2 is allowed for *scoring* (perplexity), not for classification.

---

## 2. Approach — a two-stage cascade + always-on evidence

```
                    ┌───── always runs (FR-17) ─────┐
  input text ──┬───▶│  rule-based style checker      │
               │    │  GPT-2 perplexity / burstiness │
               │    └───────────────────────────────┘
               │
               ▼
        Stage 1  (raw AI vs human)         model_raw.pkl
        │  P(AI) ≥ 0.85  ─────────────▶  final = "AI (raw)"      (FR-18)
        │  otherwise
        ▼
        Stage 2  (humanized AI vs human)   model_humanized.pkl
        │  P(AI) ≥ 0.5   ─────────────▶  final = "AI (humanized)" (FR-19)
        └  else          ─────────────▶  final = "Human"
```

Both stage models are **TF-IDF → Logistic Regression**. Each vectoriser is a
union of

- **word** 1–2 grams (vocabulary / phrasing), and
- **character** 3–5 grams (`char_wb` — punctuation habits, function-word
  morphology, spacing),

which matters most for Stage 2, where AI and human text share vocabulary and the
signal is stylistic. `C` is picked per stage by 3-fold CV ROC-AUC.

The **rule-based checker** (em-dash density, AI-cliché phrases, sentence-length
uniformity) and the **GPT-2 perplexity/burstiness** module run on every input
regardless of the cascade path and are shown as supporting evidence. The
perplexity signal is turned into a probability by a small logistic **calibrator**
fitted on real data (`aitext.calibrate_perplexity`), falling back to a
hand-tuned heuristic if the calibrator file is absent.

---

## 3. Data

| Split | Source | Notes |
|---|---|---|
| Stage 1 — human | **MAGE + HC3** (`--source both`) | MAGE: 10 domains (reddit-CMV, ELI5, TL;DR, XSum, WritingPrompts, ROCStories, HellaSwag, SQuAD, SciGen, Yelp); HC3: human Q&A answers |
| Stage 1 — raw AI | **MAGE + HC3** | MAGE: 300+ generators; HC3: ChatGPT answers |
| Stage 2 — human | *same* human pool as Stage 1 | (SRS §7.2) |
| Stage 2 — humanized AI | the Stage-1 AI texts, paraphrased | **mixed humanizer** (below) |

**Why this mix of sources:**

- **HC3 alone** (ChatGPT Q&A vs human Q&A) is *easy* — one model, one domain.
  A classifier hits ~96 %, but it has really learned "encyclopaedic / formal ⇒
  AI" and false-positives on formal human writing (SRS §9).
- **MAGE alone** is *hard* — 300+ generators, 10 domains, short and long texts.
  Classical TF-IDF tops out around 76 % on it (transformer detectors in the MAGE
  paper get ~85 % in-distribution, ~65 % out-of-distribution — this is a hard
  benchmark).
- **MAGE + HC3** keeps MAGE's domain robustness and adds HC3's clean single-model
  cases, landing in between. `--source mage` / `hc3` reproduce the extremes.

**Mixed humanizer (SRS §9 — avoid learning one tool's fingerprint):**

| Humanizer | Share | What it is |
|---|---|---|
| `t5` | ~55 % | `humarin/chatgpt_paraphraser_on_T5_base`, neural, run on GPU |
| `realpara` | ~25 % | genuine **GPT-4-paraphrased** text from MAGE's OOD para set |
| `pseudo` | ~20 % | rule-based rewrite (contractions, synonym swaps, clause shuffle) |

Ratios are configurable (`--mix`), and you can bypass all of it with
`--humanized-csv` to drop in Quillbot / Undetectable.ai output.

**Split:** every model uses a stratified **80 / 20 train/test split**
(`random_state=42`). The 20 % test rows are written to `data/*_test.csv` so the
evaluation never scores on training data.

---

## 4. Results

Trained on **MAGE + HC3**, 9,000 samples/class, stratified 80/20 split.
Full tables in [`docs/results.md`](docs/results.md) (`python -m aitext.report`).

| Model (held-out) | Accuracy | ROC-AUC |
|---|---|---|
| Stage 1 — raw AI vs human | **82.0 %** | 0.912 |
| Stage 2 — humanized AI vs human | **85.8 %** | 0.941 |
| **Cascade — binary (AI vs human)** | **85.8 %** | — |
| Cascade — 3-way (human / raw-ai / humanized-ai) | 62.4 % | — |
| GPT-2 perplexity signal alone | — | 0.77 (5-fold CV) |

**The cascade is deliberately conservative** — Stage 2 (the more careful
"humanized vs human" check) is the decisive vote whenever Stage 1 isn't *very*
confident (P(AI) < 0.95). On the held-out binary task:

| | precision | recall |
|---|---|---|
| predicted **AI** | **0.98** | 0.80 |
| predicted **human** | 0.71 | **0.97** |

- **When it says "AI", it is right 98 % of the time.** It misses ~20 % of AI
  (including ~16 % of *humanized* AI), but almost never accuses a human — the
  right bias for an academic-integrity tool.
- 3-way accuracy is low because *raw* AI vs *humanized* AI is barely separable
  in TF-IDF space (both are AI text). Humanized AI is still caught *as AI*
  83.6 % of the time.
- HC3-only would score ~96 % but false-positives on any formal human writing;
  MAGE + HC3 trades headline accuracy for a model that generalises.

![Stage 1](docs/figures/confusion_stage1.png)
![Stage 2](docs/figures/confusion_stage2.png)
![Binary cascade](docs/figures/confusion_binary.png)
![3-way cascade](docs/figures/confusion_cascade.png)
![Perplexity](docs/figures/perplexity_hist.png)

### The 80/20 split isn't lucky — 5-fold CV (`python -m aitext.robustness`)

| Stage | Accuracy | ROC-AUC |
|---|---|---|
| Stage 1 — raw AI vs human | 0.821 ± 0.006 | 0.909 ± 0.004 |
| Stage 2 — humanized AI vs human | 0.863 ± 0.004 | 0.939 ± 0.004 |

Standard deviation under 0.007 on every metric, so the headline numbers are the
real thing, not a favourable split. Perplexity calibrator: 0.765 ± 0.018 (5-fold).

---

## 5. Limitations (SRS §9)

- **Humanized detection is optimistic here.** Stage 2 partly learns the
  paraphrasers' fingerprints. A truly novel humanizer will do better against it —
  this is an industry-wide limitation, not specific to this system.
- **False positives** remain on unusually terse or non-native English.
- **English only.** No watermark verification (needs provider cooperation and
  does not survive paraphrasing).
- GPT-2 perplexity is a weak individual signal (see histogram overlap); it is
  used only as corroborating evidence.

---

## 6. Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m aitext.datasets_build --source both --max-per-class 9000  # ~20 min (GPU)
python -m aitext.train                                               # ~5 min
python -m aitext.calibrate_perplexity --sample 450                   # ~3 min (GPU)
python -m aitext.report                                              # figures + results.md
python -m aitext.evaluate                                            # console metrics
python -m aitext.robustness --folds 5                                # ~10 min, CV error bars
python -m pytest -q                                                  # 18 tests

streamlit run app.py
```

The committed `models/*.pkl` already correspond to this configuration, so the
app and `evaluate` run without retraining.

Deployment: see [`DEPLOY.md`](DEPLOY.md).
