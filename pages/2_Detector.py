"""Page 2: Detector (Main Page) — Interactive AI Text Detection Studio.

Provides:
- Large text area input with "Check Text" button (plus presets & file upload).
- Loading spinner while inference runs.
- Top Section: Overall verdict with Plotly gauge/donut chart, color-coded confidence badge,
  and plain-text probability summary.
- Middle Section: 3 tabs with Plotly charts (Model Scores horizontal bar chart, Style Analysis
  radar chart, Perplexity per-sentence line chart + burstiness metric).
- Bottom Section: Auto-generated plain-text explanation of why the verdict was reached.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

try:
    st.set_page_config(page_title="Detector - AI Text Detector", page_icon="🔍", layout="wide")
except Exception:
    pass

# Ensure src/ package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitext.detector import Detector  # noqa: E402
from aitext.extract import extract_text  # noqa: E402

# ==============================================================================
# BACKEND DETECTOR INSTANCE (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_detector() -> Detector:
    """Load and cache the detector singleton for process lifetime."""
    return Detector()


detector = get_detector()

# ==============================================================================
# SAMPLE PRESET TEXTS (FOR QUICK DEMONSTRATION & TESTING)
# ==============================================================================
PRESET_AI = (
    "The process of photosynthesis is a remarkable biological mechanism through "
    "which plants convert light energy into chemical energy. This intricate "
    "process plays a crucial role in sustaining life on Earth by producing "
    "oxygen and serving as the foundation of the food chain. Additionally, it "
    "is essential to recognize the vital contribution of chlorophyll in "
    "capturing sunlight and facilitating this essential conversion."
)

PRESET_HUMAN = (
    "ok so the thing about my landlord is he never fixes anything but always "
    "wants rent on the 1st sharp. the sink has been leaking since march and he "
    "keeps going yeah yeah i'll send someone and then nobody shows up. last "
    "week i just bought a wrench and did it myself, wasn't even that hard tbh."
)

PRESET_HUMANIZED = (
    "In today's fast-paced digital world, navigating modern education requires "
    "both discipline and adaptive thinking. It is worth noting that while students "
    "encounter numerous distractions, establishing clear routines plays a pivotal "
    "role in academic longevity — a true testament to the power of deliberate practice."
)

# ==============================================================================
# SECTION 1: HEADER & USER INPUT AREA
# ==============================================================================
st.title("🔍 AI Text Detector")
st.markdown(
    "Paste any excerpt or essay below to analyze its probability of being "
    "AI-generated, human-written, or humanized AI."
)

def _on_preset_change() -> None:
    choice = st.session_state.get("preset_selector")
    if choice == "Raw AI Sample":
        st.session_state["text_input_area"] = PRESET_AI
    elif choice == "Human Sample":
        st.session_state["text_input_area"] = PRESET_HUMAN
    elif choice == "Humanized AI Sample":
        st.session_state["text_input_area"] = PRESET_HUMANIZED


# Optional presets & file upload in a compact expander to keep UI clean
with st.expander("📁 Optional: Load Sample Presets or Upload Document"):
    col_pre, col_up = st.columns([1, 1], gap="medium")
    with col_pre:
        st.markdown("**Sample Presets**")
        st.selectbox(
            "Load a test sample:",
            ["(None)", "Raw AI Sample", "Human Sample", "Humanized AI Sample"],
            index=0,
            key="preset_selector",
            on_change=_on_preset_change,
        )
    with col_up:
        st.markdown("**File Upload**")
        uploaded_file = st.file_uploader(
            "Upload .txt, .pdf, .docx, or .md",
            type=["txt", "pdf", "docx", "md"],
            key="file_uploader",
        )

if uploaded_file is not None and "last_uploaded" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
    extracted, err = extract_text(uploaded_file.name, uploaded_file.getvalue())
    if err:
        st.error(f"Error reading file: {err}")
    elif extracted:
        st.session_state["text_input_area"] = extracted
        st.session_state["last_uploaded"] = uploaded_file.name

if "text_input_area" not in st.session_state:
    st.session_state["text_input_area"] = ""

# Large text area for pasting text to check
user_input = st.text_area(
    label="Text to analyze (minimum 10 words):",
    height=200,
    placeholder="Paste paragraph, essay, or article here...",
    key="text_input_area",
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    check_clicked = st.button("🚀 Check Text", type="primary", use_container_width=True)
with col_info:
    word_count = len((user_input or "").split())
    st.caption(f"Word count: **{word_count}** words (minimum 10 required for full cascade).")

# Save state across reruns
if check_clicked:
    if not user_input or len(user_input.split()) < 10:
        st.warning("Please provide at least 10 words for accurate analysis.")
    else:
        # Show loading spinner while detection and GPT-2 inference execute
        with st.spinner("Analyzing text with dual classifiers, stylistic rules, and GPT-2 perplexity..."):
            result = detector.detect(user_input)
            st.session_state["last_result"] = result
            st.session_state["analyzed_text"] = user_input

# Retrieve last result if available
result = st.session_state.get("last_result")

# ==============================================================================
# RESULTS DISPLAY (RENDERED ONLY AFTER RUNNING CHECK)
# ==============================================================================
if result is not None:
    st.divider()

    # Calculate final AI-probability percentage for display
    if result.label == "AI":
        ai_percentage = float(result.confidence * 100)
    elif result.label == "Human":
        ai_percentage = float((1.0 - result.confidence) * 100)
    else:
        ai_percentage = 50.0

    # Determine confidence badge configuration
    # green = "Human (confident)"
    # yellow = "Possibly humanized AI (low confidence)"
    # red = "AI-generated (high confidence)"
    if result.label == "AI":
        if result.confidence >= 0.75 and result.decisive_stage != "stage2":
            badge_text = "AI-generated (high confidence)"
            badge_color = "#ef4444"  # Red
            badge_bg = "rgba(239, 68, 68, 0.15)"
            badge_border = "#ef4444"
            badge_icon = "🔴"
        else:
            badge_text = "Possibly humanized AI (low confidence)"
            badge_color = "#f59e0b"  # Yellow
            badge_bg = "rgba(245, 158, 11, 0.15)"
            badge_border = "#f59e0b"
            badge_icon = "🟡"
    elif result.label == "Human":
        if result.confidence >= 0.75 and "low confidence" not in result.confidence_band.lower():
            badge_text = "Human (confident)"
            badge_color = "#10b981"  # Green
            badge_bg = "rgba(16, 185, 129, 0.15)"
            badge_border = "#10b981"
            badge_icon = "🟢"
        else:
            badge_text = "Possibly humanized AI (low confidence)"
            badge_color = "#f59e0b"  # Yellow
            badge_bg = "rgba(245, 158, 11, 0.15)"
            badge_border = "#f59e0b"
            badge_icon = "🟡"
    else:
        badge_text = "Possibly humanized AI (low confidence)"
        badge_color = "#f59e0b"  # Yellow
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "#f59e0b"
        badge_icon = "🟡"

    # ==========================================================================
    # 1. TOP SECTION - OVERALL VERDICT
    # ==========================================================================
    # Requirement: One-line plain text summary above the chart
    st.markdown(f"### This text is {ai_percentage:.0f}% likely AI-generated")

    col_gauge, col_badge = st.columns([1.3, 1], gap="large")

    with col_gauge:
        # Plotly Gauge Chart showing final AI-probability percentage
        gauge_bar_color = badge_color
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(ai_percentage, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": "AI Probability",
                    "font": {"size": 18, "color": "#e5e7eb", "family": "Inter, sans-serif"},
                },
                number={
                    "suffix": "%",
                    "font": {"size": 42, "color": gauge_bar_color, "family": "Inter, sans-serif"},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
                    "bar": {"color": gauge_bar_color, "thickness": 0.28},
                    "bgcolor": "rgba(255, 255, 255, 0.04)",
                    "borderwidth": 1,
                    "bordercolor": "#374151",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16, 185, 129, 0.20)"},
                        {"range": [40, 70], "color": "rgba(245, 158, 11, 0.20)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.20)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=40, b=20),
            height=280,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_badge:
        # Color-coded confidence badge & verdict card
        st.markdown(
            f"""
            <div style="
                background: {badge_bg};
                border: 2px solid {badge_border};
                border-radius: 12px;
                padding: 16px 20px;
                margin-top: 25px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.25);
            ">
                <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; color: {badge_color}; letter-spacing: 0.05em;">
                    Verdict Status
                </div>
                <div style="font-size: 22px; font-weight: 700; color: #f9fafb; margin: 6px 0;">
                    {badge_icon} {badge_text}
                </div>
                <div style="font-size: 14px; color: #d1d5db; line-height: 1.5; margin-top: 8px;">
                    <b>Confidence Band:</b> {html.escape(result.confidence_band)}<br>
                    <b>Decisive Stage:</b> {html.escape(result.decisive_stage.upper())}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.metric("Words Analyzed", len(st.session_state.get("analyzed_text", "").split()))
        with c_sub2:
            st.metric("Model Certainty", f"{result.confidence * 100:.0f}%")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # 2. MIDDLE SECTION - MODEL BREAKDOWN (ST.TABS WITH 3 TABS)
    # ==========================================================================
    st.subheader("📊 Detailed Model Breakdown")
    tab1, tab2, tab3 = st.tabs(["Model Scores", "Style Analysis", "Perplexity"])

    # --------------------------------------------------------------------------
    # TAB 1: MODEL SCORES (HORIZONTAL BAR CHART COMPARING STAGE 1 & STAGE 2)
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("#### Classifier Confidence Comparison")
        st.caption("Compares prediction confidence and probabilities across Stage 1 (Raw AI) and Stage 2 (Humanized AI).")

        s1, s2 = result.stage1, result.stage2
        s1_ai_score = (s1.p_ai * 100) if (s1.ran and s1.p_ai is not None) else 0.0
        s2_ai_score = (s2.p_ai * 100) if (s2.ran and s2.p_ai is not None) else 0.0

        stages = ["Stage 2 (Humanized AI)", "Stage 1 (Raw AI)"]
        scores = [s2_ai_score, s1_ai_score]
        bar_colors = [
            "#ef4444" if s >= 50 else "#10b981" for s in scores
        ]

        text_labels = [
            f"{s2_ai_score:.1f}% AI ({s2.label})" if s2.ran else f"Not run ({s2.note})",
            f"{s1_ai_score:.1f}% AI ({s1.label})" if s1.ran else f"Not run ({s1.note})",
        ]

        fig_bars = go.Figure(
            go.Bar(
                x=scores,
                y=stages,
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#374151", width=1),
                ),
                text=text_labels,
                textposition="auto",
                textfont=dict(color="#ffffff", size=13),
            )
        )
        fig_bars.update_layout(
            xaxis=dict(
                range=[0, 100],
                title="P(AI) Probability Percentage",
                ticksuffix="%",
                gridcolor="#2d3748",
            ),
            yaxis=dict(gridcolor="#2d3748"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=240,
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.info(
                f"**Stage 1 (Raw AI Classifier):** "
                f"{'Ran — ' + str(round(s1_ai_score, 1)) + '% AI confidence' if s1.ran else s1.note}"
            )
        with col_m2:
            st.info(
                f"**Stage 2 (Humanized AI Classifier):** "
                f"{'Ran — ' + str(round(s2_ai_score, 1)) + '% AI confidence' if s2.ran else s2.note}"
            )

    # --------------------------------------------------------------------------
    # TAB 2: STYLE ANALYSIS (RADAR/SPIDER CHART VIA SCATTERPOLAR)
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("#### Rule-Based Stylistic Fingerprints")
        st.caption("Radar chart visualizing linguistic features that modern LLMs tend to over-index on.")

        stl = result.style or {}
        cliche_count = stl.get("cliche_count", 0)
        cliches_found = stl.get("cliches_found", [])
        em_dash_per_1k = stl.get("em_dash_per_1k_words", 0.0)
        uniformity = stl.get("sentence_length_uniformity", 0.0) or 0.0
        style_ai_score = stl.get("style_ai_score", 0.0)

        # Normalize metrics to 0-100 scale for radar visualization
        # Clichés: 3+ distinct clichés is high risk (100)
        norm_cliche = min(100.0, (cliche_count / 3.0) * 100.0)
        # Em-dash: 10 per 1k words is high density (100)
        norm_emdash = min(100.0, (em_dash_per_1k / 10.0) * 100.0)
        # Uniformity: 0.0 to 1.0 mapped to 0-100
        norm_uniformity = float(uniformity * 100.0)

        categories = [
            "Cliché Phrases",
            "Em-Dash Frequency",
            "Sentence Length Uniformity",
        ]
        values = [norm_cliche, norm_emdash, norm_uniformity]

        # Close polygon for radar chart
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        col_radar, col_style_stats = st.columns([1.2, 1], gap="medium")

        with col_radar:
            fig_radar = go.Figure(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself",
                    fillcolor="rgba(245, 158, 11, 0.25)",
                    line=dict(color="#f59e0b", width=2.5),
                    marker=dict(size=7, color="#fbbf24"),
                    hoverinfo="theta+r",
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickfont=dict(size=10, color="#9ca3af"),
                        gridcolor="#374151",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=12, color="#e5e7eb"),
                        gridcolor="#374151",
                    ),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=30, b=30),
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_style_stats:
            st.markdown(f"**Overall Style AI Risk:** `{style_ai_score:.2f} / 1.00`")
            st.metric("Cliché Phrases Detected", f"{cliche_count}")
            st.metric("Em-Dash Frequency", f"{em_dash_per_1k:.1f} per 1k words")
            st.metric("Sentence Uniformity", f"{uniformity * 100:.0f}%")

        if cliches_found:
            st.markdown(
                f"**Identified Clichés:** "
                + " ".join([f"`{c}`" for c in cliches_found[:8]])
                + (" *(and more)*" if len(cliches_found) > 8 else "")
            )

    # --------------------------------------------------------------------------
    # TAB 3: PERPLEXITY (LINE CHART ACROSS SENTENCES + BURSTINESS METRIC)
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("#### GPT-2 Perplexity & Burstiness Curve")
        st.caption("Measures sentence-by-sentence word predictability. Humans vary widely; AI remains predictable and flat.")

        perp = result.perplexity or {}
        sent_ppls = perp.get("sentence_perplexities", [])
        overall_ppl = perp.get("perplexity")
        burstiness = perp.get("burstiness", 0.0)
        mean_sp = perp.get("mean_sentence_perplexity", 0.0)
        perp_available = perp.get("available", False)

        if perp_available and sent_ppls:
            # Display the line chart: x-axis = sentence number, y-axis = perplexity score
            x_sent = [f"S{i+1}" for i in range(len(sent_ppls))]
            fig_line = go.Figure()

            fig_line.add_trace(
                go.Scatter(
                    x=x_sent,
                    y=sent_ppls,
                    mode="lines+markers",
                    name="Sentence Perplexity",
                    line=dict(color="#38bdf8", width=3),
                    marker=dict(size=8, color="#0ea5e9"),
                    hovertemplate="Sentence %{x}: Perplexity %{y:.1f}<extra></extra>",
                )
            )

            # Reference baseline for typical AI threshold (~35)
            fig_line.add_hline(
                y=35,
                line_dash="dot",
                line_color="#ef4444",
                annotation_text="Typical AI Predictability Threshold (< 35)",
                annotation_position="top left",
                annotation_font=dict(color="#ef4444", size=11),
            )

            fig_line.update_layout(
                xaxis=dict(title="Sentence Number", gridcolor="#2d3748"),
                yaxis=dict(title="Perplexity Score (lower = more predictable)", gridcolor="#2d3748"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
            )
            st.plotly_chart(fig_line, use_container_width=True)

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                # Requirement: burstiness displayed as a metric using st.metric
                st.metric(
                    label="Burstiness (Variance)",
                    value=f"{burstiness:.2f}",
                    help="Sentence-to-sentence perplexity variance. Higher burstiness reflects organic human variance.",
                )
            with col_p2:
                st.metric(
                    label="Overall Perplexity",
                    value=f"{overall_ppl:.1f}",
                    help="exp(mean token NLL). Under 35 indicates highly predictable, AI-like phrasing.",
                )
            with col_p3:
                st.metric(
                    label="Mean Sentence Perplexity",
                    value=f"{mean_sp:.1f}",
                )
        else:
            st.info(
                f"Per-sentence perplexity requires multiple sentences (minimum 4 words each). "
                f"Note: {perp.get('note', 'Insufficient input sentences')}"
            )
            if perp.get("burstiness") is not None:
                st.metric("Burstiness", f"{perp.get('burstiness', 0.0):.2f}")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # 3. BOTTOM SECTION - PLAIN-TEXT EXPLANATION
    # ==========================================================================
    st.subheader("💡 Verdict Explanation")

    # Construct auto-generated plain text explanation of WHY the verdict was reached
    reasons = []
    stl = result.style or {}
    perp = result.perplexity or {}

    # 1. Decisive model factor
    if result.decisive_stage == "stage1":
        reasons.append("Stage 1 raw-AI classifier detected strong direct machine-generation patterns")
    elif result.decisive_stage == "stage2":
        reasons.append("Stage 2 specialist classifier detected humanized/paraphrased AI signatures")
    elif result.label == "Human":
        reasons.append("both Stage 1 and Stage 2 models concurred on human linguistic patterns")

    # 2. Perplexity & burstiness factor
    ppl_val = perp.get("perplexity")
    burst_val = perp.get("burstiness")
    if perp.get("available") and ppl_val is not None:
        if ppl_val < 35:
            reasons.append(f"low perplexity ({ppl_val:.1f}), indicating high word predictability characteristic of LLMs")
        elif ppl_val > 70:
            reasons.append(f"high perplexity ({ppl_val:.1f}), showing diverse and surprising human vocabulary")

        if burst_val is not None and burst_val < 25:
            reasons.append("low burstiness across sentences (monotonous pacing)")
        elif burst_val is not None and burst_val >= 60:
            reasons.append(f"healthy burstiness ({burst_val:.1f}) reflecting varied sentence structures")

    # 3. Style factors
    cliches = stl.get("cliches_found", [])
    if cliches:
        reasons.append(f"{len(cliches)} AI-cliché phrase{'s' if len(cliches) > 1 else ''} detected ({', '.join(cliches[:3])})")
    
    uniformity_val = stl.get("sentence_length_uniformity", 0.0) or 0.0
    if uniformity_val >= 0.75:
        reasons.append(f"uniform sentence structure ({uniformity_val * 100:.0f}% length uniformity)")
    elif uniformity_val <= 0.45:
        reasons.append("natural variation in sentence lengths")

    emdash_val = stl.get("em_dash_per_1k_words", 0.0)
    if emdash_val >= 8.0:
        reasons.append(f"elevated em-dash frequency ({emdash_val:.1f} per 1,000 words)")

    explanation_str = "; ".join(reasons)
    if explanation_str:
        explanation_full = f"**Flagged due to:** {explanation_str}."
    else:
        explanation_full = f"**Summary:** Evaluated as {result.confidence_band} with {result.confidence * 100:.0f}% confidence."

    st.markdown(
        f"""
        <div style="
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid {badge_color};
            border-radius: 6px;
            padding: 14px 18px;
            font-size: 15px;
            color: #e5e7eb;
            line-height: 1.6;
        ">
            {explanation_full}
        </div>
        """,
        unsafe_allow_html=True,
    )
