"""API tests — request handling and JSON shaping for the React UI.

These avoid the heavy models: they cover input validation, short-text handling
and the NaN-safe serialisation. Model-backed behaviour is in test_integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from aitext.api import _json_safe, ai_percent, app, generate_summary  # noqa: E402
from aitext.detector import DetectionResult, StageOutput  # noqa: E402

client = TestClient(app)


def _result(label: str, confidence: float, **kw) -> DetectionResult:
    return DetectionResult(
        label=label, confidence=confidence, confidence_band="band",
        decisive_stage=kw.pop("decisive_stage", "stage2"),
        stage1=StageOutput(ran=False), stage2=StageOutput(ran=False), **kw)


def test_empty_input_is_a_user_error_not_a_crash():
    r = client.post("/api/check", data={"text": "   "})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] is None and "nothing to check" in body["error"]


def test_too_short_text_has_no_gauge_value():
    body = client.post("/api/check", data={"text": "way too short"}).json()
    assert body["error"] is None
    assert body["result"]["label"] == "Uncertain"
    assert body["result"]["ai_pct"] is None
    assert body["result"]["summary"] == "not enough text to analyze"


def test_unsupported_upload_returns_error_with_filename():
    r = client.post("/api/check", files={"file": ("notes.exe", b"MZ", "application/octet-stream")})
    body = r.json()
    assert body["src"] == "notes.exe"
    assert "Unsupported file type" in body["error"]


def test_text_file_upload_is_extracted():
    r = client.post("/api/check", files={"file": ("a.txt", b"hello world", "text/plain")})
    body = r.json()
    assert body["src"] == "a.txt" and body["words"] == 2
    assert body["text"] == "hello world"  # echoed so the UI can show what was checked


def test_nan_perplexity_serialises_as_null():
    payload = _json_safe({"perplexity": float("nan"), "xs": [float("inf"), 1.5]})
    assert payload == {"perplexity": None, "xs": [None, 1.5]}


def test_ai_percent_matches_label():
    assert ai_percent(_result("AI", 0.8)) == 80.0
    assert round(ai_percent(_result("Human", 0.7)), 6) == 30.0
    assert ai_percent(_result("Uncertain", 0.0)) is None
    assert ai_percent(_result("Uncertain", 0.5)) == 50.0


def test_summary_flags_cliches_and_stage():
    r = _result("AI", 0.9, style={"cliches_found": ["moreover"],
                                  "sentence_length_uniformity": 0.5})
    assert generate_summary(r) == (
        "flagged due to: paraphrased AI signatures (Stage 2), "
        "1 AI-cliché phrase detected")
