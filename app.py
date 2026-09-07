"""Streamlit web interface — terminal-styled AI text detector with inline charts.

Run locally:   streamlit run app.py
Deploy:        Hugging Face Spaces / Streamlit Community Cloud (see DEPLOY.md)
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aitext.detector import DetectionResult, Detector  # noqa: E402
from aitext.extract import extract_text  # noqa: E402

st.set_page_config(page_title="czechtext", layout="centered")

MONO = '"SF Mono","SFMono-Regular","Menlo","Monaco","Consolas","Liberation Mono",monospace'
MONO_CLEAN = 'SF Mono, SFMono-Regular, Menlo, Monaco, Consolas, monospace'
COLOR = {"AI": "#ff6b6b", "Human": "#5ef2a0", "Uncertain": "#c9c9c9"}

# ==============================================================================
# TERMINAL STYLING (DARK MONOSPACE CSS)
# ==============================================================================
st.markdown(
    f"""
    <style>
      #MainMenu, header, footer {{ visibility: hidden; }}
      .stApp {{ background: #14161a; }}
      .block-container {{ max-width: 880px; padding-top: 1.8rem; padding-bottom: 7rem; }}

      /* Unified single-window terminal wrapper */
      [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #1b1e24 !important;
        border: 1px solid #2c313a !important;
        border-radius: 10px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,.45) !important;
        padding: 0 !important;
        overflow: hidden !important;
        margin-bottom: 24px !important;
      }}
      [data-testid="stVerticalBlockBorderWrapper"] > div {{
        background: #1b1e24 !important;
        padding: 0 !important;
        gap: 0 !important;
      }}
      [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {{
        gap: 0 !important;
        padding: 0 !important;
        background: #1b1e24 !important;
      }}

      .term-bar {{
        display: flex; align-items: center; gap: 8px;
        padding: 10px 14px; background: #23262d; border-bottom: 1px solid #2c313a;
      }}
      .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
      .dot.r {{ background: #ff5f56; }} .dot.y {{ background: #ffbd2e; }} .dot.g {{ background: #27c93f; }}
      .term-title {{ margin-left: 8px; color: #8b929e; font-family: {MONO}; font-size: 12px; }}

      .term-block {{
        padding: 10px 18px;
        font-family: {MONO_CLEAN};
        font-size: 13px;
        color: #d6dae1;
        line-height: 1.55;
      }}

      .cmd {{ color: #7f8794; font-family: {MONO}; }}
      .cmd .p {{ color: #5ef2a0; }}
      .verdict {{ font-weight: 700; }}
      .k {{ color: #7f8794; font-family: {MONO}; }}
      .muted {{ color: #6b7280; }}
      .rule {{ color: #2c313a; font-family: {MONO}; }}
      .banner {{ color: #6b7280; font-family: {MONO}; font-size: 12px; margin-bottom: 8px; }}

      /* chat input -> terminal prompt line */
      [data-testid="stBottom"], [data-testid="stBottom"] > div,
      [data-testid="stBottomBlockContainer"] {{ background: #14161a !important; }}
      [data-testid="stBottomBlockContainer"] {{ max-width: 880px; padding-bottom: 1.4rem; }}

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


def _generate_summary(result: DetectionResult) -> str:
    """Generate plain-text explanation of why the verdict was reached."""
    reasons = []
    perp = result.perplexity or {}
    stl = result.style or {}

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


def _fmt_style_and_fluency(result: DetectionResult) -> str:
    """Terminal plain-text block for style, fluency, and summary."""
    blocks: list[str] = []

    # --- writing style ----------------------------------------------
    stl = result.style
    if stl:
        L: list[str] = []
        L.append(f'{_K("style")}AI stylistic habits')
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
        blocks.append("\n".join(L))

    # --- fluency / perplexity --------------------------------------
    p = result.perplexity
    if p:
        L_p: list[str] = []
        L_p.append(f'{_K("fluency")}GPT-2 predictability')
        if p.get("available"):
            ppl = p.get("perplexity")
            pw = ("low &mdash; predictable, typical of AI" if ppl and ppl < 35
                  else "medium" if ppl and ppl < 80
                  else "high &mdash; surprising, typical of humans")
            L_p.append(_sub("level", f'{ppl} '
                                     f'<span class="muted">({pw})</span>'))
            L_p.append(_sub("variety", f'{p.get("burstiness")} sentence-to-sentence '
                                       f'<span class="muted">(burstiness)</span>'))
            sc = p.get("perplexity_ai_score", 0)
            L_p.append(_sub("summary", f'{_score_word(sc)} '
                                       f'<span class="muted">(fluency score {sc:.2f} / 1.00, '
                                       f'{html.escape(p.get("note") or "")})</span>'))
        else:
            L_p.append(_sub("summary", f'<span class="muted">unavailable &mdash; '
                                       f'{html.escape(p.get("note") or "")}</span>'))
        blocks.append("\n".join(L_p))

    # --- explanation summary --------------------------------------
    summary_text = _generate_summary(result)
    L_s = [f'{_K("summary")}<span style="color:#d6dae1">{html.escape(summary_text)}</span>']
    for m in result.messages:
        L_s.append(f'<span class="muted">note: {html.escape(m)}</span>')
    blocks.append("\n".join(L_s))

    # Separate each major block with a visible gap
    return ('<div style="height:14px;"></div>').join(
        f'<div style="white-space:pre-wrap;">{b}</div>' for b in blocks
    )


# ==============================================================================
# PLOTLY CHARTS (TERMINAL MONOSPACE THEME)
# ==============================================================================
def _make_gauge_fig(ai_pct: float, color: str) -> go.Figure:
    """Compact dark terminal-styled gauge for final AI probability beside verdict."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(ai_pct, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "%", "font": {"size": 28, "color": color, "family": MONO_CLEAN}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#2c313a",
                    "tickfont": {"size": 9, "color": "#7f8794", "family": MONO_CLEAN},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "#2c313a",
                "steps": [
                    {"range": [0, 40], "color": "rgba(94, 242, 160, 0.12)"},
                    {"range": [40, 70], "color": "rgba(255, 189, 46, 0.12)"},
                    {"range": [70, 100], "color": "rgba(255, 107, 107, 0.12)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=4),
        height=120,
    )
    return fig


def _make_models_fig(s1, s2) -> go.Figure:
    """Horizontal bar chart in place of check 1 and check 2."""
    s1_val = (s1.p_ai * 100) if (s1.ran and s1.p_ai is not None) else 0.0
    s2_val = (s2.p_ai * 100) if (s2.ran and s2.p_ai is not None) else 0.0

    stages = ["check 1: raw-ai", "check 2: reworded-ai"]
    vals = [s1_val, s2_val]
    colors = [
        "#4b5263" if (not s1.ran) else ("#ff6b6b" if s1_val >= 50 else "#5ef2a0"),
        "#4b5263" if (not s2.ran) else ("#ff6b6b" if s2_val >= 50 else "#5ef2a0"),
    ]
    texts = [
        f"{s1_val:.1f}% ({s1.label})" if s1.ran else f"not run ({s1.note})",
        f"{s2_val:.1f}% ({s2.label})" if s2.ran else f"not run ({s2.note})",
    ]

    # Contrast fix: dark #0f172a text on light green (#5ef2a0), white on red/gray
    text_colors = [
        "#0f172a" if (val < 50 and s.ran) else "#ffffff"
        for val, s in zip(vals, [s1, s2])
    ]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=stages,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#2c313a", width=1)),
            text=texts,
            textposition="auto",
            textfont=dict(family=MONO_CLEAN, color=text_colors, size=11),
        )
    )
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            ticksuffix="%",
            gridcolor="#23262d",
            tickfont=dict(family=MONO_CLEAN, color="#7f8794", size=10),
        ),
        yaxis=dict(
            autorange="reversed",  # check 1 on top, check 2 on bottom
            gridcolor="#23262d",
            tickfont=dict(family=MONO_CLEAN, color="#d6dae1", size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=6, b=6),
        height=100,
    )
    return fig


# ==============================================================================
# MAIN APPLICATION FLOW
# ==============================================================================
detector = get_detector()
history: list[dict] = st.session_state.setdefault("history", [])

# Render single continuous terminal window
with st.container(border=True):
    # Terminal header bar
    st.markdown(
        """
        <div class="term-bar">
          <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
          <span class="term-title">czechtext — zsh</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not history:
        st.markdown(
            f"""
            <div class="term-block">
              <div class="banner">note: reworded ("humanized") AI is hard to catch &mdash; read this as evidence, not proof.</div>
              <span class="cmd"><span class="p">&gt;</span> ready. paste text or attach file to check.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render each check turn seamlessly inside the same terminal window
    for idx, turn in enumerate(history):
        src = f' &lt; {html.escape(turn["src"])}' if turn.get("src") else ""

        # Command line prompt
        st.markdown(
            f'<div class="term-block" style="padding-bottom: 2px;">'
            f'<span class="cmd"><span class="p">&gt;</span> check <span class="muted">({turn["words"]} words{src})</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if turn.get("error"):
            st.markdown(
                f'<div class="term-block" style="color:#ff6b6b; padding-top: 0;">'
                f'{html.escape(turn["error"])}'
                f'</div>',
                unsafe_allow_html=True,
            )
            continue

        res: DetectionResult = turn["result"]
        ai_pct = (res.confidence * 100) if res.label == "AI" else ((1.0 - res.confidence) * 100)
        verdict_c = COLOR.get(res.label, "#c9c9c9")
        decided = {
            "stage1": "the raw-AI check",
            "stage2": "the reworded-AI check",
            "none": "not enough text",
        }.get(res.decisive_stage, res.decisive_stage)

        # 1. VERDICT ROW: Left = Verdict text, Right = % Gauge graph with AI PROBABILITY label
        col_verdict, col_gauge = st.columns([1.5, 1.0], gap="small")
        with col_verdict:
            st.markdown(
                f'<div class="term-block" style="padding-top: 10px; line-height: 1.7;">'
                f'<span class="k">verdict    </span><span class="verdict" style="color:{verdict_c}">{html.escape(res.confidence_band)}</span><br>'
                f'<span class="muted">&middot; how sure  </span> {res.confidence * 100:.0f}%<br>'
                f'<span class="muted">&middot; decided by</span> {decided}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_gauge:
            st.markdown(
                f'<div style="text-align:center; font-family:{MONO_CLEAN}; font-size:11px; font-weight:600; color:#8b929e; letter-spacing:0.06em; padding-top:8px; margin-bottom:-4px;">'
                f'P(AI) PROBABILITY'
                f'</div>',
                unsafe_allow_html=True,
            )
            fig_g = _make_gauge_fig(ai_pct, verdict_c)
            st.plotly_chart(fig_g, use_container_width=True,
                            config={"displayModeBar": False}, key=f"gauge_{idx}")

        # 2. CHECK 1 & CHECK 2: Inlined horizontal bar graphs with readable high-contrast text
        fig_bars = _make_models_fig(res.stage1, res.stage2)
        st.plotly_chart(fig_bars, use_container_width=True,
                        config={"displayModeBar": False}, key=f"bars_{idx}")

        # 3. REMAINING TERMINAL OUTPUT: Style, fluency, and summary explanation
        style_fluency_text = _fmt_style_and_fluency(res)
        st.markdown(
            f'<div class="term-block" style="padding-top: 4px;">'
            f'{style_fluency_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Divider between multiple checks
        if idx < len(history) - 1:
            st.markdown(
                f'<div style="color:#2c313a; padding: 4px 18px 12px 18px; font-family:{MONO_CLEAN};">'
                f'{"-" * 64}'
                f'</div>',
                unsafe_allow_html=True,
            )


# Bottom terminal chat prompt line
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
                {"words": 0, "src": f.name, "error": err, "result": None}
            )
            st.rerun()
        text = (text + "\n\n" + extracted).strip() if text else extracted
        src = f.name

    if not text:
        st.session_state.history.append(
            {"words": 0, "error": "nothing to check — type text or attach a file", "result": None}
        )
        st.rerun()

    with st.spinner(""):
        detection_res = detector.detect(text)

    st.session_state.history.append({
        "words": len(text.split()),
        "src": src,
        "result": detection_res,
    })
    st.rerun()
