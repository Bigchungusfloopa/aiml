"""Smoke tests — run with:  python -m pytest -q  (from repo root)

These don't need trained models; they exercise the always-on evidence modules
and NFR-3 graceful degradation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitext import rules
from aitext.detector import Detector
from aitext.preprocess import normalize, split_sentences


def test_normalize():
    assert normalize("  Hello   WORLD\n\t") == "hello world"
    assert normalize("") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("   ") == []


def test_rules_cliche_and_emdash():
    txt = ("Let's delve into this — it is worth noting that in today's world "
           "we must navigate the complexities of change.")
    rep = rules.analyze(txt)
    assert rep.em_dash_count >= 1
    assert rep.cliche_count >= 2
    assert 0.0 <= rep.style_ai_score <= 1.0


def test_rules_empty():
    rep = rules.analyze("")
    assert rep.word_count == 0
    assert rep.sentence_count == 0


def test_detector_short_input_is_graceful():
    det = Detector()
    res = det.detect("too short")
    assert res.label == "Uncertain"
    assert res.decisive_stage == "none"
    assert res.messages


def test_detector_empty_input_is_graceful():
    det = Detector()
    res = det.detect("")
    assert res.label == "Uncertain"
    assert not res.stage1.ran
