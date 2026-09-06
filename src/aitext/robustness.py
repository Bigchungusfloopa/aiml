"""Cross-validated robustness check for the two stage classifiers.

`train.py` reports a single 80/20 split. This runs a 5-fold stratified CV of the
*chosen* estimator (calibrated LinearSVC) end to end — vectoriser refit inside
each fold — and reports mean ± std for accuracy and ROC-AUC, so the headline
numbers come with an error bar.

Run:
    python -m aitext.robustness            # both stages, 5-fold
    python -m aitext.robustness --folds 10
"""

from __future__ import annotations

import argparse

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from .paths import STAGE1_CSV, STAGE2_CSV
from .train import _load, _new_vectorizer


def _run(name: str, csv_path, folds: int, seed: int) -> None:
    X, y = _load(csv_path)
    pipe = make_pipeline(
        _new_vectorizer(),
        CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", dual="auto"), cv=3),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=["accuracy", "roc_auc"], n_jobs=-1)
    acc, auc = scores["test_accuracy"], scores["test_roc_auc"]
    print(f"\n=== {name}  ({folds}-fold, n={len(y)}) ===")
    print(f"  accuracy  {acc.mean():.3f} ± {acc.std():.3f}   "
          f"(folds: {', '.join(f'{a:.3f}' for a in acc)})")
    print(f"  roc_auc   {auc.mean():.3f} ± {auc.std():.3f}   "
          f"(folds: {', '.join(f'{a:.3f}' for a in auc)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-validated robustness check")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()

    if args.stage in ("1", "both"):
        _run("Stage 1 — raw AI vs human", STAGE1_CSV, args.folds, args.seed)
    if args.stage in ("2", "both"):
        _run("Stage 2 — humanized AI vs human", STAGE2_CSV, args.folds, args.seed)


if __name__ == "__main__":
    main()
