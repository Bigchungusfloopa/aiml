"""Streamlit web interface (FR-21 .. FR-25, SRS 4.1).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud / Hugging Face Spaces (NFR-7)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aitext.detector import Detector  # noqa: E402
from aitext.extract import extract_text  # noqa: E402

st.set_page_config(page_title="AI-Generated Text Detection", page_icon="🔍")


@st.cache_resource(show_spinner="Loading models…")
def get_detector() -> Detector:
    return Detector()


def band_color(label: str, confidence: float) -> str:
    if label == "AI":
        return "#c0392b"
    if label == "Human":
        return "#1e8449" if confidence >= 0.75 else "#b9770e"
    return "#5d6d7e"


st.title("🔍 AI-Generated Text Detection")
st.caption(
    "Two-stage classical ML cascade + rule-based style checker + GPT-2 "
    "perplexity/burstiness. Humanized AI text is hard to catch — treat results "
    "as evidence, not proof."
)

detector = get_detector()

tab_paste, tab_upload = st.tabs(["Paste text", "Upload a file"])

with tab_paste:
    pasted = st.text_area(
        "Paste text to check",
        height=260,
        placeholder="Paste at least ~10 words of English text…",
        label_visibility="collapsed",
    )

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a PDF, Word document, or text file",
        type=["pdf", "docx", "txt", "md"],
    )
    extracted = ""
    if uploaded is not None:
        extracted, err = extract_text(uploaded.name, uploaded.getvalue())
        if err:
            st.error(err)
        elif extracted.strip():
            st.success(f"Extracted {len(extracted.split()):,} words from "
                       f"{uploaded.name}")
            with st.expander("Preview extracted text"):
                st.text(extracted[:3000] + ("…" if len(extracted) > 3000 else ""))

# The file tab wins when it has content, otherwise use the pasted text.
text = extracted if extracted.strip() else pasted

if st.button("Check text", type="primary"):
    if not text.strip():
        st.warning("Paste some text or upload a file first.")
    else:
        with st.spinner("Analyzing…"):
            result = detector.detect(text)

        for msg in result.messages:
            st.info(msg)

        color = band_color(result.label, result.confidence)
        st.markdown(
            f"<div style='padding:1rem;border-radius:8px;background:{color};"
            f"color:white;'>"
            f"<div style='font-size:1.6rem;font-weight:700;'>{result.label}</div>"
            f"<div style='font-size:1.05rem;'>{result.confidence_band}</div>"
            f"<div style='opacity:.85;'>Confidence: "
            f"{result.confidence * 100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Evidence breakdown", expanded=True):
            st.markdown("**Cascade**")
            c1, c2 = st.columns(2)
            with c1:
                s1 = result.stage1
                st.write("**Stage 1 — raw AI vs human**")
                if s1.ran:
                    st.write(f"- Label: `{s1.label}`")
                    st.write(f"- P(AI): `{s1.p_ai}`")
                    st.write(f"- Confidence: `{s1.confidence}`")
                else:
                    st.write(f"_did not run — {s1.note}_")
            with c2:
                s2 = result.stage2
                st.write("**Stage 2 — humanized AI vs human**")
                if s2.ran:
                    st.write(f"- Label: `{s2.label}`")
                    st.write(f"- P(AI): `{s2.p_ai}`")
                    st.write(f"- Confidence: `{s2.confidence}`")
                else:
                    st.write(f"_did not run — {s2.note}_")
            st.write(f"Decisive stage: **{result.decisive_stage}**")

            st.divider()
            st.markdown("**Rule-based style checker**")
            s = result.style
            st.write(f"- Words / sentences: {s.get('word_count')} / "
                     f"{s.get('sentence_count')}")
            st.write(f"- Em-dashes: {s.get('em_dash_count')} "
                     f"({s.get('em_dash_per_1k_words')} per 1k words)")
            found = s.get("cliches_found") or []
            st.write(f"- AI-cliché phrases ({len(found)}): "
                     f"{', '.join(found) if found else '—'}")
            st.write(f"- Mean sentence length: {s.get('mean_sentence_length')} "
                     f"(variance {s.get('sentence_length_variance')}, "
                     f"uniformity {s.get('sentence_length_uniformity')})")
            st.write(f"- Style AI-likeness (supporting): "
                     f"{s.get('style_ai_score')}")

            st.divider()
            st.markdown("**GPT-2 perplexity / burstiness**")
            p = result.perplexity
            if p.get("available"):
                st.write(f"- Perplexity: {p.get('perplexity')} "
                         f"(lower ⇒ more AI-like)")
                st.write(f"- Burstiness: {p.get('burstiness')} "
                         f"(higher ⇒ more human-like)")
                st.write(f"- Mean sentence perplexity: "
                         f"{p.get('mean_sentence_perplexity')} "
                         f"({p.get('n_sentences_scored')} sentences scored)")
                st.write(f"- Perplexity AI-likeness (supporting): "
                         f"{p.get('perplexity_ai_score')}")
            else:
                st.write(f"_unavailable — {p.get('note')}_")

        with st.expander("Raw result (JSON)"):
            st.json(result.as_dict())

st.divider()
st.caption(
    "Known limits (SRS §9): false positives on unusually formal, simple, or "
    "non-native English writing; well-humanized AI often evades detection; "
    "English only; no watermark verification."
)
