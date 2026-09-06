"""Calibrate the GPT-2 perplexity signal on real data (FR-14..FR-16 support).

The raw perplexity/burstiness heuristic in ``perplexity.py`` uses hand-picked
cut-offs. This script fits a tiny logistic regression on
``[log perplexity, log(burstiness+1), log mean-sentence-perplexity]`` against the
Stage 1 labels and saves it to ``models/perplexity_calib.pkl``. When that file is
present, ``perplexity.analyze`` uses it to produce ``perplexity_ai_score``
instead of the heuristic.

This is still only *supporting* evidence — it never overrides the cascade label.

Run (uses GPT-2, so give it a few minutes):
    python -m aitext.calibrate_perplexity --sample 500
"""

from __future__ import annotations

import argparse
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .paths import MODELS_DIR, STAGE1_CSV, ensure_dirs
from .perplexity import _load_model, _text_perplexity  # noqa: PLC2701
from .preprocess import split_sentences

CALIB_PATH = MODELS_DIR / "perplexity_calib.pkl"
FEATURES = ("log_ppl", "log_burst", "log_mean_sppl")


def _features(text: str, tok, model, device) -> list[float] | None:
    overall = _text_perplexity(text, tok, model, device)
    if overall is None or not math.isfinite(overall):
        return None
    sp = []
    for s in split_sentences(text):
        if len(s.split()) >= 4:
            p = _text_perplexity(s, tok, model, device)
            if p is not None and math.isfinite(p):
                sp.append(p)
    if len(sp) >= 2:
        mean_sp = sum(sp) / len(sp)
        burst = sum((p - mean_sp) ** 2 for p in sp) / len(sp)
    else:
        mean_sp, burst = overall, 0.0
    return [math.log(overall), math.log1p(burst), math.log(max(mean_sp, 1e-6))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the perplexity calibrator")
    ap.add_argument("--sample", type=int, default=500,
                    help="rows per class to score with GPT-2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not STAGE1_CSV.exists():
        raise SystemExit("Run `python -m aitext.datasets_build` first.")

    df = pd.read_csv(STAGE1_CSV).dropna(subset=["text", "label"])
    df = df[df["text"].str.split().str.len().between(20, 400)]
    per = int(min(args.sample, df["label"].value_counts().min()))
    df = df.groupby("label", group_keys=False).sample(n=per,
                                                      random_state=args.seed)
    print(f"scoring {len(df)} texts with GPT-2 ...")

    tok, model, device = _load_model()
    rows, labels = [], []
    for i, (text, label) in enumerate(zip(df["text"], df["label"])):
        feats = _features(text, tok, model, device)
        if feats is not None:
            rows.append(feats)
            labels.append(1 if label == "ai" else 0)
        if i % 100 == 0:
            print(f"  {i}/{len(df)}", flush=True)

    X = np.array(rows)
    y = np.array(labels)

    # Honest estimate: 5-fold CV ROC-AUC, then refit on everything for the
    # saved model.
    cv = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X, y, cv=5, scoring="roc_auc")
    print(f"\nperplexity-only ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f} "
          f"(5-fold CV, n={len(y)})")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    print("coefficients:", dict(zip(FEATURES, clf.coef_[0].round(3))))

    ensure_dirs()
    joblib.dump({"model": clf, "features": FEATURES,
                 "cv_roc_auc": float(cv.mean())}, CALIB_PATH)
    print(f"saved -> {CALIB_PATH}")


if __name__ == "__main__":
    main()
