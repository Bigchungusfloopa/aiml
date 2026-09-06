"""Central path configuration.

Model files, vectorizers and the cliché list live outside the application code
so they can be retrained / edited without code changes (NFR-6).
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (src/aitext/paths.py -> repo/)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("AITEXT_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("AITEXT_MODELS_DIR", ROOT / "models"))
CONFIG_DIR = Path(os.environ.get("AITEXT_CONFIG_DIR", ROOT / "config"))

CLICHES_FILE = CONFIG_DIR / "cliches.txt"

# Prepared training CSVs (produced by datasets_build.py)
STAGE1_CSV = DATA_DIR / "stage1_raw.csv"
STAGE2_CSV = DATA_DIR / "stage2_humanized.csv"

# Persisted artifacts (Appendix A manifest)
MODEL_RAW = MODELS_DIR / "model_raw.pkl"
VECTORIZER_RAW = MODELS_DIR / "vectorizer_raw.pkl"
MODEL_HUMANIZED = MODELS_DIR / "model_humanized.pkl"
VECTORIZER_HUMANIZED = MODELS_DIR / "vectorizer_humanized.pkl"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
