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

st.set_page_config(page_title="czechtext", layout="centered")

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


def _lean(p_ai: float) -> str:
    """Plain-English read of a P(AI) value."""
    if p_ai >= 0.80:
        return "strongly AI"
    if p_ai >= 0.60:
        return "leans AI"
    if p_ai > 0.40:
        return "borderline"
    if p_ai > 0.20:
        return "leans human"
    return "strongly human"


def _score_word(s: float) -> str:
    if s >= 0.66:
        return "very AI-like"
    if s >= 0.45:
        return "somewhat AI-like"
    if s >= 0.30:
        return "mixed"
    return "human-like"


def _K(label: str) -> str:
    return f'<span class="k">{label:<11}</span>'


def _sub(label: str, text: str) -> str:
    return f'  <span class="muted">&middot; {label:<10}</span> {text}'


def _fmt_result(result) -> str:
    """Layman-readable terminal rendering that still shows every parameter."""
    c = COLOR.get(result.label, "#c9c9c9")
    L: list[str] = []

    # --- verdict --------------------------------------------------------
    L.append(f'{_K("verdict")}<span class="verdict" style="color:{c}">'
             f'{html.escape(result.confidence_band)}</span>')
    L.append(_sub("how sure", f'{result.confidence * 100:.0f}%'))
    decided = {"stage1": "the raw-AI check",
               "stage2": "the reworded-AI check",
               "none": "not enough text"}.get(result.decisive_stage,
                                              result.decisive_stage)
    L.append(_sub("decided by", decided))
    L.append("")

    # --- check 1 -------------------------------------------------------
    s1, s2 = result.stage1, result.stage2
    L.append(f'{_K("check 1")}Is this raw AI, straight from a chatbot?')
    if s1.ran:
        L.append(_sub("result", f'{_lean(s1.p_ai)} '
                              f'<span class="muted">({s1.p_ai * 100:.0f}% chance it is AI)</span>'))
    else:
        L.append(_sub("result", f'<span class="muted">did not run &mdash; '
                              f'{html.escape(s1.note)}</span>'))
    L.append("")

    # --- check 2 -------------------------------------------------------
    L.append(f'{_K("check 2")}Is this AI text reworded to look human?')
    if s2.ran:
        L.append(_sub("result", f'{_lean(s2.p_ai)} '
                              f'<span class="muted">({s2.p_ai * 100:.0f}% chance it is AI)</span>'))
    else:
        L.append(_sub("result", f'<span class="muted">did not run &mdash; '
                              f'{html.escape(s2.note)}</span>'))
    L.append("")

    # --- writing style ----------------------------------------------
    stl = result.style
    if stl:
        L.append(f'{_K("style")}Habits that AI writing tends to over-use')
        found = stl.get("cliches_found") or []
        n = stl.get("cliche_count", 0)
        if found:
            shown = ", ".join(found[:6]) + ("&hellip;" if len(found) > 6 else "")
            L.append(_sub("phrases", f'{n} found: <span class="muted">'
                                     f'{html.escape(shown)}</span>'))
        else:
            L.append(_sub("phrases", '0 found'))
        L.append(_sub("dashes", f'{stl.get("em_dash_per_1k_words")} em-dashes '
                              f'per 1000 words'))
        u = stl.get("sentence_length_uniformity", 0) or 0
        rhythm = ("very even &mdash; robotic" if u >= 0.8
                  else "fairly even" if u >= 0.6 else "varied &mdash; human-like")
        L.append(_sub("rhythm", f'sentence lengths {u * 100:.0f}% uniform '
                              f'<span class="muted">({rhythm})</span>'))
        sc = stl.get("style_ai_score", 0)
        L.append(_sub("summary", f'{_score_word(sc)} '
                               f'<span class="muted">(style score {sc:.2f} / 1.00)</span>'))
        L.append("")

    # --- fluency / perplexity --------------------------------------
    p = result.perplexity
    if p:
        L.append(f'{_K("fluency")}How predictable the text is to GPT-2')
        if p.get("available"):
            ppl = p.get("perplexity")
            pw = ("low &mdash; very predictable, typical of AI" if ppl and ppl < 35
                  else "medium" if ppl and ppl < 80
                  else "high &mdash; surprising, typical of humans")
            L.append(_sub("level", f'{ppl} '
                                    f'<span class="muted">({pw})</span>'))
            L.append(_sub("variety", f'{p.get("burstiness")} sentence-to-sentence '
                                   f'<span class="muted">(humans vary more)</span>'))
            sc = p.get("perplexity_ai_score", 0)
            L.append(_sub("summary", f'{_score_word(sc)} '
                                   f'<span class="muted">(fluency score {sc:.2f} / 1.00, '
                                   f'{html.escape(p.get("note") or "")})</span>'))
        else:
            L.append(_sub("summary", f'<span class="muted">unavailable &mdash; '
                                   f'{html.escape(p.get("note") or "")}</span>'))
        L.append("")

    for m in result.messages:
        L.append(f'<span class="muted">note: {html.escape(m)}</span>')

    return "\n".join(L).rstrip()


def _render(history: list[dict]) -> str:
    banner = (
        '<span class="muted">note: reworded ("humanized") AI is hard to catch '
        '&mdash; read this as evidence, not proof.</span>'
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
        '<span class="term-title">czechtext — zsh</span></div>'
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
