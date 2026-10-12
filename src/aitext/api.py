"""HTTP API for the React UI.

Run locally:   uvicorn aitext.api:app --app-dir src --reload
Serves the built frontend from ``web/dist`` when it exists, so one process
handles both the UI and ``/api/check``.
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles

from .detector import DetectionResult, Detector
from .extract import extract_text
from .paths import ROOT

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # same cap the Streamlit UI had
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="czechtext", docs_url="/api/docs", openapi_url="/api/openapi.json")

# GPT-2 / sklearn inference is not worth parallelising; serialise it.
_detect_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector()


def generate_summary(result: DetectionResult) -> str:
    """Plain-text explanation of why the verdict was reached."""
    perp = result.perplexity or {}
    stl = result.style or {}
    if not stl and not perp:
        return "not enough text to analyze"

    reasons: list[str] = []

    ppl = perp.get("perplexity")
    if perp.get("available") and ppl is not None:
        if ppl < 35:
            reasons.append(f"low perplexity ({ppl:.1f})")
        elif ppl > 70:
            reasons.append(f"high human perplexity ({ppl:.1f})")

        burst = perp.get("burstiness")
        if burst is not None and burst < 25:
            reasons.append(f"low burstiness ({burst:.1f})")
        elif burst is not None and burst >= 60:
            reasons.append(f"varied burstiness ({burst:.1f})")

    cliches = stl.get("cliches_found", [])
    if cliches:
        reasons.append(f"{len(cliches)} AI-cliché phrase{'s' if len(cliches) > 1 else ''} detected")

    u = stl.get("sentence_length_uniformity", 0) or 0
    if u >= 0.75:
        reasons.append(f"uniform sentence structure ({u * 100:.0f}%)")
    elif u <= 0.45:
        reasons.append("varied sentence rhythm")

    em = stl.get("em_dash_per_1k_words", 0.0)
    if em >= 8.0:
        reasons.append(f"high em-dash frequency ({em:.1f}/1k)")

    if result.decisive_stage == "stage1" and result.label == "AI":
        reasons.insert(0, "strong raw-AI n-grams")
    elif result.decisive_stage == "stage2" and result.label == "AI":
        reasons.insert(0, "paraphrased AI signatures (Stage 2)")

    if reasons:
        return "flagged due to: " + ", ".join(reasons)
    return f"classified as {result.confidence_band.lower()} via {result.decisive_stage}"


def ai_percent(result: DetectionResult) -> float | None:
    """P(AI) as a percentage for the gauge; None when there is no verdict."""
    if result.label == "AI":
        return result.confidence * 100
    if result.label == "Human":
        return (1.0 - result.confidence) * 100
    # "Uncertain": too-short input has confidence 0.0 and nothing to plot.
    return None if result.confidence == 0.0 else 50.0


def _json_safe(value: Any) -> Any:
    """Replace NaN/inf (e.g. unavailable perplexity) with None for strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _error(src: str | None, message: str) -> dict:
    return {"words": 0, "src": src, "error": message, "text": "", "result": None}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "models_ready": get_detector().ready}


@app.post("/api/check")
async def check(
    text: str = Form(""),
    file: UploadFile | None = File(None),
) -> dict:
    text = (text or "").strip()
    src: str | None = None

    if file is not None and file.filename:
        src = file.filename
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return _error(src, f"{src} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        extracted, err = extract_text(src, data)
        if err:
            return _error(src, err)
        text = (text + "\n\n" + extracted).strip() if text else extracted.strip()

    if not text:
        return _error(src, "nothing to check — type text or attach a file")

    detector = get_detector()
    with _detect_lock:
        res = detector.detect(text)

    payload = _json_safe(res.as_dict())
    payload["ai_pct"] = ai_percent(res)
    payload["summary"] = generate_summary(res)
    return {"words": len(text.split()), "src": src, "error": None, "text": text,
            "result": payload}


# Mounted last so it never shadows /api routes.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
