"""End-to-end cascade evaluation.

Runs the full Detector (Stage 1 -> conditional Stage 2 -> combined) over a
held-out slice of the prepared datasets and reports:

  * Stage 1 in isolation  : raw AI vs human
  * Stage 2 in isolation   : humanized AI vs human
  * Cascade, 3-way          : human / raw-AI / humanized-AI
    (predicted class = final label + which stage was decisive)

Run:
    python -m aitext.evaluate --sample 1500
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .detector import Detector
from .paths import STAGE1_CSV, STAGE2_CSV


def _load_holdout(csv_path, sample: int, seed: int) -> pd.DataFrame:
    """Prefer the held-out *_test.csv written by train.py; fall back to the
    full dataset (with a warning) if training hasn't produced one yet."""
    test_path = csv_path.with_name(csv_path.stem + "_test.csv")
    if test_path.exists():
        df = pd.read_csv(test_path)
    else:
        print(f"[evaluate] WARNING: {test_path.name} missing — sampling the "
              f"full dataset, so scores include training rows.")
        df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.split().str.len() >= 10]
    n = min(sample, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _cascade_3way(det: Detector, text: str) -> str:
    label, decisive = det.classify_cascade(text)
    if label == "AI":
        return "humanized-ai" if decisive == "stage2" else "raw-ai"
    return "human"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the detection cascade")
    ap.add_argument("--sample", type=int, default=1500,
                    help="rows to draw from each dataset")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    det = Detector()
    if not det.ready:
        raise SystemExit("Models not trained. Run: python -m aitext.train")

    s1 = _load_holdout(STAGE1_CSV, args.sample, args.seed)
    s2 = _load_holdout(STAGE2_CSV, args.sample, args.seed)

    from .preprocess import normalize

    def _stage_preds(stage, frame):
        preds, ys = [], []
        for t, lab in zip(frame["text"], frame["label"].str.lower()):
            _, label, _ = det._predict(stage, normalize(t))  # noqa: SLF001
            preds.append("ai" if label == "AI" else "human")
            ys.append("ai" if lab == "ai" else "human")
        return ys, preds

    # --- Stage 1 isolated ---
    print("\n===== Stage 1 (raw AI vs human) =====")
    y1, pred1 = _stage_preds(det._stage1, s1)  # noqa: SLF001
    print(classification_report(y1, pred1, digits=3, zero_division=0))

    # --- Stage 2 isolated (run Stage 2 on every row) ---
    print("\n===== Stage 2 (humanized AI vs human) =====")
    if det._stage2 is None:  # noqa: SLF001 - eval introspection
        print("Stage 2 model missing — skipped.")
    else:
        y2, pred2 = _stage_preds(det._stage2, s2)  # noqa: SLF001
        print(classification_report(y2, pred2, digits=3, zero_division=0))

    # --- Cascade 3-way ---
    print("\n===== Cascade (3-way: human / raw-ai / humanized-ai) =====")
    rows = []
    for t, lab in zip(s1["text"], s1["label"].str.lower()):
        gold = "raw-ai" if lab == "ai" else "human"
        rows.append((gold, _cascade_3way(det, t)))
    for t, lab in zip(s2["text"], s2["label"].str.lower()):
        if lab != "ai":
            continue
        rows.append(("humanized-ai", _cascade_3way(det, t)))

    gold = [g for g, _ in rows]
    pred = [p for _, p in rows]
    labels = ["human", "raw-ai", "humanized-ai"]
    print(classification_report(gold, pred, labels=labels, digits=3,
                                zero_division=0))
    cm = confusion_matrix(gold, pred, labels=labels)
    print("confusion matrix (rows=gold, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # Humanized-AI recall is the headline SRS §9 limitation number.
    hi = labels.index("humanized-ai")
    recall = cm[hi, hi] / cm[hi].sum() if cm[hi].sum() else float("nan")
    print(f"\nHumanized-AI caught as AI (any stage): "
          f"{(cm[hi, 1] + cm[hi, 2]) / cm[hi].sum():.1%}")
    print(f"Humanized-AI correctly routed via Stage 2: {recall:.1%}")


if __name__ == "__main__":
    main()
