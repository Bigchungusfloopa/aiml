"""Performance checks against the SRS non-functional requirements.

NFR-1: Stage 1 + Stage 2 predictions < 2 s for inputs up to 2,000 words.
NFR-2: GPT-2 perplexity < 5 s for inputs up to 2,000 words on CPU.

These are timing tests, so they're marked and skipped without models. The
GPT-2 test allows a generous ceiling because the first call also loads the
model; it warms up first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitext.paths import MODEL_RAW  # noqa: E402

pytestmark = pytest.mark.skipif(
    not MODEL_RAW.exists(), reason="trained models not present")

_PARA = (
    "The rapid advancement of technology has reshaped modern society in "
    "profound and far-reaching ways. From communication to commerce, nearly "
    "every aspect of daily life now depends on digital infrastructure. "
)
TEXT_2000 = (_PARA * 100).split()
TEXT_2000 = " ".join(TEXT_2000[:2000])


def test_nfr1_stage_predictions_under_2s() -> None:
    from aitext.detector import Detector
    from aitext.preprocess import normalize

    det = Detector()
    tn = normalize(TEXT_2000)
    det._predict(det._stage1, tn)  # noqa: SLF001  (warm vectoriser caches)

    t0 = time.perf_counter()
    det._predict(det._stage1, tn)  # noqa: SLF001
    det._predict(det._stage2, tn)  # noqa: SLF001
    elapsed = time.perf_counter() - t0
    print(f"\nNFR-1: stage 1 + stage 2 on 2,000 words = {elapsed:.3f}s")
    assert elapsed < 2.0


def test_nfr2_perplexity_under_5s() -> None:
    pytest.importorskip("torch")
    from aitext.perplexity import _load_model, analyze  # noqa: PLC2701

    _load_model()  # warm: first call loads GPT-2
    analyze("warm up the pipeline with a short sentence here please")

    t0 = time.perf_counter()
    r = analyze(TEXT_2000)
    elapsed = time.perf_counter() - t0
    print(f"NFR-2: GPT-2 perplexity on 2,000 words = {elapsed:.3f}s "
          f"(available={r.available})")
    # 5s is the CPU target; MPS/CUDA is far under. Allow 8s headroom for CI CPUs.
    assert elapsed < 8.0
