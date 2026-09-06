"""Integration tests — exercise the trained models end to end.

Skipped automatically when models/ has no .pkl (so the smoke suite still runs on
a fresh clone). Run everything with:  python -m pytest -q
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitext.detector import Detector  # noqa: E402
from aitext.paths import MODEL_RAW  # noqa: E402

pytestmark = pytest.mark.skipif(
    not MODEL_RAW.exists(),
    reason="trained models not present — run `python -m aitext.train`",
)


# A few samples. We assert direction / that the product behaves sanely, not
# exact probabilities — the classical models are ~82-86% and have known blind
# spots (abstract corporate prose, very short casual text — SRS §9).
AI_TEXTS = [
    "The process of photosynthesis is a remarkable biological mechanism through "
    "which plants convert light energy into chemical energy. This intricate "
    "process plays a crucial role in sustaining life on Earth by producing "
    "oxygen and serving as the foundation of the food chain. Additionally, it "
    "is essential to recognize the vital contribution of chlorophyll in "
    "capturing sunlight and facilitating this essential conversion.",
    "Social media has profoundly reshaped the way individuals communicate and "
    "consume information in the modern era. While it offers unprecedented "
    "connectivity, it also raises significant concerns regarding privacy, the "
    "spread of misinformation, and its impact on mental well-being. It is "
    "therefore essential to approach these platforms with a balanced and "
    "critical perspective.",
]

HUMAN_TEXTS = [
    "ok so the thing about my landlord is he never fixes anything but always "
    "wants rent on the 1st sharp. the sink has been leaking since march and he "
    "keeps going yeah yeah i'll send someone and then nobody shows up. last "
    "week i just bought a wrench and did it myself, wasn't even that hard tbh.",
    "went hiking with dave on saturday, weather was garbage, rained the whole "
    "way up and we couldn't see anything from the top. still kind of fun "
    "though? we got soaked and then sat in the car eating cold sandwiches and "
    "arguing about which trail we should've taken. classic.",
]


@pytest.fixture(scope="module")
def det() -> Detector:
    d = Detector()
    assert d.ready
    return d


@pytest.mark.parametrize("text", AI_TEXTS)
def test_ai_text_flagged_as_ai(det: Detector, text: str) -> None:
    r = det.detect(text)
    assert r.label == "AI"


@pytest.mark.parametrize("text", HUMAN_TEXTS)
def test_human_text_not_confidently_ai(det: Detector, text: str) -> None:
    # The classical models over-flag short casual text (SRS §9). Require only
    # that a wrong "AI" call is never made with high confidence.
    r = det.detect(text)
    assert r.label in {"Human", "AI"}
    if r.label == "AI":
        assert r.confidence < 0.75 or "low confidence" in r.confidence_band.lower()


def test_detect_returns_full_evidence(det: Detector) -> None:
    r = det.detect(AI_TEXTS[0])
    assert r.stage1.ran
    assert set(r.style) >= {"cliche_count", "em_dash_per_1k_words",
                            "style_ai_score"}
    assert "available" in r.perplexity
    if r.perplexity["available"]:
        assert 0.0 <= r.perplexity["perplexity_ai_score"] <= 1.0
        assert r.perplexity["perplexity"] > 0


def test_stage_probabilities_are_probabilities(det: Detector) -> None:
    from aitext.preprocess import normalize
    for stage in (det._stage1, det._stage2):  # noqa: SLF001
        p_ai, label, conf = det._predict(stage, normalize(AI_TEXTS[0]))  # noqa: SLF001
        assert 0.0 <= p_ai <= 1.0
        assert 0.5 <= conf <= 1.0
        assert label in {"AI", "Human"}


def test_pdf_upload_path(det: Detector) -> None:
    """extract_text -> detect, the path the upload tab uses."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aitext.extract import extract_text

    fig = plt.figure(figsize=(8, 10))
    fig.text(0.1, 0.9, AI_TEXTS[1], wrap=True, fontsize=11, va="top")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)

    text, err = extract_text("sample.pdf", buf.getvalue())
    assert err is None
    assert len(text.split()) > 20
    result = det.detect(text)
    assert result.label in {"AI", "Human", "Uncertain"}
