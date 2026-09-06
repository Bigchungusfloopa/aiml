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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

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


def _new_vectorizer() -> FeatureUnion:
    """Word (1-2 gram) + character (3-5 gram) TF-IDF.

    Char n-grams pick up stylistic regularities — punctuation habits, function-
    word morphology, spacing — that word n-grams miss, and they help most on the
    humanized/paraphrased task where vocabulary overlaps heavily with human text.
    """
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.9,
        sublinear_tf=True, strip_accents="unicode", max_features=200_000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_df=0.95,
        sublinear_tf=True, strip_accents="unicode", max_features=200_000,
    )
    return FeatureUnion([("word", word), ("char", char)])


def train_stage(name: str, csv_path, model_path, vec_path,
                test_size: float = 0.2, seed: int = 42) -> dict:
    X, y = _load(csv_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    vectorizer = _new_vectorizer()
    Xtr = vectorizer.fit_transform(X_tr)
    Xte = vectorizer.transform(X_te)

    print(f"\n=== {name} ===")
    print(f"features: {Xtr.shape[1]:,}")

    # Candidate estimators; pick the best by 3-fold CV ROC-AUC on the train split.
    candidates: dict[str, object] = {}
    for c in (0.3, 1.0, 3.0):
        candidates[f"logreg C={c}"] = LogisticRegression(
            max_iter=2000, C=c, class_weight="balanced")
    for c in (0.1, 0.5, 1.0):
        candidates[f"linsvc C={c}"] = CalibratedClassifierCV(
            LinearSVC(C=c, class_weight="balanced", dual="auto"), cv=3)

    best_name, best_cv, best_est = "", -1.0, None
    for cand_name, est in candidates.items():
        cv = cross_val_score(est, Xtr, y_tr, cv=3, scoring="roc_auc",
                             n_jobs=-1).mean()
        print(f"  {cand_name:<14} CV ROC-AUC={cv:.3f}")
        if cv > best_cv:
            best_name, best_cv, best_est = cand_name, cv, est
    print(f"chosen: {best_name} (CV ROC-AUC={best_cv:.3f})")

    clf = best_est
    clf.fit(Xtr, y_tr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(classification_report(y_te, pred, target_names=["human", "ai"], digits=3))
    auc = roc_auc_score(y_te, proba)
    print(f"held-out ROC-AUC: {auc:.3f}")

    ensure_dirs()
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    # Persist the held-out split so evaluate.py never scores on training rows.
    test_path = csv_path.with_name(csv_path.stem + "_test.csv")
    pd.DataFrame({"text": X_te,
                  "label": ["ai" if v else "human" for v in y_te]}).to_csv(
        test_path, index=False)
    print(f"saved -> {model_path.name}, {vec_path.name}, {test_path.name}")

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
