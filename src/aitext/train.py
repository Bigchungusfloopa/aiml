"""Offline training pipeline for the two classical classifiers.

FR-3 .. FR-9. Trains TF-IDF + Logistic Regression for each stage, reports
held-out metrics, and persists (model, vectorizer) pairs to models/ per the
Appendix A manifest. Not on the runtime request path (SRS 6).

Run:
    python -m aitext.train                 # trains both stages
    python -m aitext.train --stage 1
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from .paths import (
    MODEL_HUMANIZED,
    MODEL_RAW,
    STAGE1_CSV,
    STAGE2_CSV,
    VECTORIZER_HUMANIZED,
    VECTORIZER_RAW,
    ensure_dirs,
)
from .preprocess import normalize_series

POSITIVE = "ai"  # positive class for both stages


def _load(csv_path) -> tuple[list[str], list[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run:  python -m aitext.datasets_build"
        )
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().astype(bool)]
    X = normalize_series(df["text"].tolist())
    y = (df["label"].str.lower() == POSITIVE).astype(int).tolist()
    return X, y


def _new_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=100_000,
    )


def train_stage(name: str, csv_path, model_path, vec_path,
                test_size: float = 0.2, seed: int = 42) -> dict:
    X, y = _load(csv_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    vectorizer = _new_vectorizer()
    Xtr = vectorizer.fit_transform(X_tr)
    Xte = vectorizer.transform(X_te)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, y_tr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"\n=== {name} ===")
    print(classification_report(y_te, pred, target_names=["human", "ai"], digits=3))
    auc = roc_auc_score(y_te, proba)
    print(f"ROC-AUC: {auc:.3f}")

    ensure_dirs()
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"saved -> {model_path.name}, {vec_path.name}")

    return {"name": name, "roc_auc": float(auc),
            "n_train": len(y_tr), "n_test": len(y_te)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage 1 / Stage 2 classifiers")
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = []
    if args.stage in ("1", "both"):
        results.append(train_stage(
            "Stage 1 — raw AI vs human", STAGE1_CSV,
            MODEL_RAW, VECTORIZER_RAW, args.test_size, args.seed))
    if args.stage in ("2", "both"):
        results.append(train_stage(
            "Stage 2 — humanized AI vs human", STAGE2_CSV,
            MODEL_HUMANIZED, VECTORIZER_HUMANIZED, args.test_size, args.seed))

    print("\nSummary:")
    for r in results:
        print(f"  {r['name']}: ROC-AUC={r['roc_auc']:.3f} "
              f"(train={r['n_train']}, test={r['n_test']})")


if __name__ == "__main__":
    main()
