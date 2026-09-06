"""Streamlit web interface — terminal-styled chat (FR-21 .. FR-25, SRS 4.1).

Run locally:   streamlit run app.py
Deploy:        Hugging Face Spaces / Streamlit Community Cloud (see DEPLOY.md)
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aitext.detector import Detector  # noqa: E402
from aitext.extract import extract_text  # noqa: E402

st.set_page_config(page_title="ai-text-detect", layout="centered")

MONO = '"SF Mono","SFMono-Regular","Menlo","Monaco","Consolas","Liberation Mono",monospace'
COLOR = {"AI": "#ff6b6b", "Human": "#5ef2a0", "Uncertain": "#c9c9c9"}

st.markdown(
    f"""
    <style>
      #MainMenu, header, footer {{ visibility: hidden; }}
      .stApp {{ background: #14161a; }}
      .block-container {{ max-width: 860px; padding-top: 2.2rem; padding-bottom: 7rem; }}

      .term {{
        border: 1px solid #2c313a; border-radius: 10px; overflow: hidden;
        background: #1b1e24; box-shadow: 0 20px 60px rgba(0,0,0,.45);
        font-family: {MONO}; font-size: 13px; line-height: 1.55;
      }}
      .term-bar {{
        display: flex; align-items: center; gap: 8px;
        padding: 9px 12px; background: #23262d; border-bottom: 1px solid #2c313a;
      }}
      .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
      .dot.r {{ background: #ff5f56; }} .dot.y {{ background: #ffbd2e; }} .dot.g {{ background: #27c93f; }}
      .term-title {{ margin-left: 8px; color: #8b929e; font-family: {MONO}; font-size: 12px; }}
      .term-body {{
        padding: 14px 16px; color: #d6dae1; white-space: pre-wrap;
        word-break: break-word; min-height: 240px;
        max-height: 66vh; overflow-y: auto;
      }}

      .cmd {{ color: #7f8794; }}
      .cmd .p {{ color: #5ef2a0; }}
      .verdict {{ font-weight: 700; }}
      .k {{ color: #7f8794; }}
      .muted {{ color: #6b7280; }}
      .rule {{ color: #2c313a; }}

      /* chat input -> terminal prompt line */
      [data-testid="stBottom"], [data-testid="stBottom"] > div,
      [data-testid="stBottomBlockContainer"] {{ background: #14161a !important; }}
      [data-testid="stBottomBlockContainer"] {{ max-width: 860px; padding-bottom: 1.4rem; }}

      [data-testid="stChatInput"],
      [data-testid="stChatInput"] > div,
      [data-testid="stChatInput"] [data-baseweb="textarea"],
      [data-testid="stChatInput"] [data-baseweb="base-input"] {{
        background: #1b1e24 !important;
      }}
      [data-testid="stChatInput"] {{
        border: 1px solid #2c313a !important; border-radius: 10px;
      }}
      [data-testid="stChatInput"] textarea {{
        font-family: {MONO} !important; font-size: 13px !important; color: #d6dae1 !important;
      }}
      [data-testid="stChatInput"] textarea::placeholder {{ color: #5b626d !important; }}
      [data-testid="stChatInput"] button {{ color: #7f8794 !important; background: transparent !important; }}
      [data-testid="stChatInput"] button:hover {{ color: #d6dae1 !important; }}
      [data-testid="stChatInput"] svg {{ fill: currentColor; }}

      /* file-drop overlay */
      [data-testid="stChatInputFileUploadButton"] + div,
      section[data-testid="stFileUploaderDropzone"] {{
        background: #1b1e24 !important; color: #8b929e !important;
        border-color: #2c313a !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_detector() -> Detector:
    return Detector()


def _fmt_result(result) -> str:
    """Plain-text terminal rendering of a DetectionResult."""
    c = COLOR.get(result.label, "#c9c9c9")
    lines = [
        f'<span class="verdict" style="color:{c}">{result.label.lower()}</span>'
        f'  <span class="muted">::</span>  {html.escape(result.confidence_band)}',
        f'<span class="k">confidence </span> {result.confidence * 100:.1f}%',
        f'<span class="k">path       </span> {result.decisive_stage}',
        "",
    ]

    s1, s2 = result.stage1, result.stage2
    lines.append(
        f'<span class="k">stage 1    </span> '
        + (f'{s1.label:<6} p(ai) {s1.p_ai:.2f}' if s1.ran
           else f'<span class="muted">-- {html.escape(s1.note)}</span>'))
    lines.append(
        f'<span class="k">stage 2    </span> '
        + (f'{s2.label:<6} p(ai) {s2.p_ai:.2f}' if s2.ran
           else f'<span class="muted">-- {html.escape(s2.note)}</span>'))

    stl = result.style
    if stl:
        lines.append(
            f'<span class="k">style      </span> '
            f'score {stl.get("style_ai_score")}   '
            f'cliches {stl.get("cliche_count")}   '
            f'em-dash {stl.get("em_dash_per_1k_words")}/1k   '
            f'uniformity {stl.get("sentence_length_uniformity")}')
        found = stl.get("cliches_found") or []
        if found:
            lines.append(f'<span class="muted">           '
                         f'{html.escape(", ".join(found))}</span>')

    p = result.perplexity
    if p:
        if p.get("available"):
            lines.append(
                f'<span class="k">perplexity </span> '
                f'score {p.get("perplexity_ai_score")}   '
                f'ppl {p.get("perplexity")}   '
                f'burstiness {p.get("burstiness")}   '
                f'[{html.escape(p.get("note") or "")}]')
        else:
            lines.append(f'<span class="k">perplexity </span> '
                         f'<span class="muted">unavailable — '
                         f'{html.escape(p.get("note") or "")}</span>')

    for m in result.messages:
        lines.append(f'<span class="muted">note: {html.escape(m)}</span>')

    return "\n".join(lines)


def _render(history: list[dict]) -> str:
    banner = (
        '<span class="muted">ai-text-detect 1.0  ::  human / raw-ai / humanized-ai\n'
        'enter text below, or drop a .pdf / .docx / .txt file. '
        'humanized AI often evades detection — treat results as evidence.</span>'
    )
    blocks = [banner]
    for turn in history:
        src = f' &lt; {html.escape(turn["src"])}' if turn.get("src") else ""
        blocks.append(
            f'<span class="cmd"><span class="p">&gt;</span> check '
            f'<span class="muted">({turn["words"]} words{src})</span></span>')
        if turn.get("error"):
            blocks.append(f'<span style="color:#ff6b6b">{html.escape(turn["error"])}</span>')
        else:
            blocks.append(turn["body"])
        blocks.append('<span class="rule">' + "-" * 52 + "</span>")
    return (
        '<div class="term">'
        '<div class="term-bar"><span class="dot r"></span>'
        '<span class="dot y"></span><span class="dot g"></span>'
        '<span class="term-title">ai-text-detect — zsh</span></div>'
        f'<div class="term-body">{chr(10).join(blocks)}</div></div>'
    )


detector = get_detector()
history: list[dict] = st.session_state.setdefault("history", [])

st.markdown(_render(history), unsafe_allow_html=True)

msg = st.chat_input(
    "check text…",
    accept_file=True,
    file_type=["pdf", "docx", "txt", "md"],
)

if msg is not None:
    text = (msg.text or "").strip()
    src = None
    if msg.files:
        f = msg.files[0]
        extracted, err = extract_text(f.name, f.getvalue())
        if err:
            st.session_state.history.append(
                {"words": 0, "src": f.name, "error": err})
            st.rerun()
        text = (text + "\n\n" + extracted).strip() if text else extracted
        src = f.name

    if not text:
        st.session_state.history.append(
            {"words": 0, "error": "nothing to check — type text or attach a file"})
        st.rerun()

    with st.spinner(""):
        result = detector.detect(text)
    st.session_state.history.append({
        "words": len(text.split()),
        "src": src,
        "body": _fmt_result(result),
    })
    st.rerun()
