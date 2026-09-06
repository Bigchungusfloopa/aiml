"""Page 1: How It Works — Pipeline Architecture & Technical Overview.

Explains the multi-stage cascade detection architecture, supporting evidence
channels (stylistic heuristics & GPT-2 perplexity/burstiness), and system limitations.
"""

from __future__ import annotations

import streamlit as st

try:
    st.set_page_config(page_title="How It Works - AI Text Detector", page_icon="📖", layout="wide")
except Exception:
    pass

# ==============================================================================
# SECTION 1: HEADER & INTRO
# ==============================================================================
st.title("🛡️ How It Works: Multi-Stage AI Detection Pipeline")
st.markdown(
    """
    This application analyzes text to distinguish between **human-written**, 
    **raw AI-generated** (e.g., direct outputs from ChatGPT, Claude, Gemini), and 
    **humanized/paraphrased AI** text (text disguised via prompting or paraphrasing tools).
    
    Rather than relying on a single fallible indicator, the detector utilizes a **hierarchical cascade** 
    coupled with **always-on supporting evidence** to produce calibrated, explainable verdicts.
    """
)

st.divider()

# ==============================================================================
# SECTION 2: DETECTION PIPELINE (STEP-BY-STEP BREAKDOWN)
# ==============================================================================
st.subheader("🔄 The 4-Stage Detection Architecture")
st.caption("How text flows through our cascading classifiers and statistical analyzers:")

col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("### `Stage 1` Raw AI Detection")
        st.markdown(
            """
            **Primary Screening Classifier**
            - **Input**: Normalized word & character n-grams ($1$–$3$ words, $3$–$5$ chars).
            - **Model**: Regularized Logistic Regression trained on diverse human vs. LLM corpora.
            - **Behavior**: If the model is **confidently AI** ($P(\\text{AI}) \\ge 0.95$), 
              it short-circuits immediately to avoid false flags on obvious chatbot text.
            - **Decision**: If confidence is below $0.95$, the text moves to Stage 2.
            """
        )

    with st.container(border=True):
        st.markdown("### `Stage 3` Always-On Evidence (Parallel)")
        st.markdown(
            """
            **Statistical & Stylistic Triangulation**
            - Runs in parallel regardless of Stage 1/2 outputs.
            - **Rule-Based Style Analysis**:
              - Overused AI cliché phrases (e.g., *"delve into"*, *"testament to"*).
              - Em-dash ($\u2014$) density per 1,000 words.
              - Sentence length uniformity and cadence consistency.
            - **GPT-2 Perplexity & Burstiness**:
              - Evaluates text predictability and sentence-to-sentence variance.
            """
        )

with col2:
    with st.container(border=True):
        st.markdown("### `Stage 2` Humanized AI Detection")
        st.markdown(
            """
            **Paraphrase & Rewriter Specialist**
            - **Triggered**: When Stage 1 reports Human or low-confidence AI.
            - **Model**: Specialized classifier trained on humanized datasets (including 
              prompts explicitly designed to bypass AI detectors).
            - **Purpose**: Catches subtle rephrasing, synonym replacements, and 
              prompt-engineered camouflage while keeping human false-positive rates low.
            - **Outcome**: Becomes the decisive second opinion for ambiguous text.
            """
        )

    with st.container(border=True):
        st.markdown("### `Stage 4` Evidence Blending & Confidence Band")
        st.markdown(
            """
            **Calibrated Synthesis & Verdict**
            - The decisive stage's probability is combined conservatively with the parallel evidence.
            - Parallel evidence **nudges** confidence and flags contradictions:
              - *E.g.*: If Stage 2 says AI but perplexity is very high, confidence is adjusted downward.
            - Assigns an actionable confidence band:
              - 🟢 `Human (confident)`
              - 🟡 `Possibly humanized AI (low confidence)`
              - 🔴 `AI-generated (high confidence)`
            """
        )

st.divider()

# ==============================================================================
# SECTION 3: DEEP-DIVE EXPANDERS
# ==============================================================================
st.subheader("🔬 Deep-Dive: Core Technologies & Metrics")

with st.expander("📚 Understanding Perplexity & Burstiness (GPT-2 Analysis)"):
    st.markdown(
        """
        - **Perplexity (PPL)**: Measures how 'surprised' a reference language model (GPT-2) is by each word sequence.
          - Lower perplexity means the words are highly probable and predictable—a signature of LLMs sampling from top probability tokens.
          - Human writing typically has higher perplexity because humans choose creative, idiosyncratic phrasings.
        - **Burstiness**: Measures the variance in perplexity and sentence structure across consecutive sentences.
          - Humans write with 'bursts'—mixing short punchy clauses with long, complex descriptive sentences.
          - AI models tend to produce uniform, steady perplexity scores across the entire passage.
        """
    )

with st.expander("🔍 Rule-Based Stylistic Indicators"):
    st.markdown(
        """
        - **AI Clichés**: Modern LLMs are prone to repetitive stylistic tropes such as:
          - *"delve into"*, *"it is worth noting"*, *"plays a crucial role"*, *"navigating the complexities"*, *"testament to"*, *"in today's digital landscape"*.
        - **Em-Dash Overuse**: Many models exhibit disproportionate frequency of em-dashes ($\u2014$) compared to natural informal or academic human text.
        - **Sentence-Length Uniformity**: Calculates $1 - \\text{CV}$ (coefficient of variation). A high score ($> 0.75$) flags robotic cadence with unvarying sentence lengths.
        """
    )

with st.expander("📊 Dataset & Model Training"):
    st.markdown(
        """
        - **Datasets**: Built on diverse benchmark datasets including **HC3** (Human-ChatGPT Comparison Corpus), **MAGE**, and rephrased corpus splits.
        - **Evaluation**: Optimized on held-out splits to minimize false positives on human text while maintaining high sensitivity to paraphrased AI content.
        - **Lightweight & Efficient**: The dual-stage pipeline uses classical TF-IDF n-grams with logistic classification for sub-second inference, while GPT-2 runs locally on CPU for statistical verification.
        """
    )

st.divider()

# ==============================================================================
# SECTION 4: LIMITATIONS & ETHICAL NOTICE
# ==============================================================================
st.subheader("⚠️ Important Limitations")

st.warning(
    """
    **Read Results as Probabilistic Evidence, Not Absolute Proof:**
    
    1. **Humanized AI is Inherently Challenging**: Sophisticated paraphrasing, heavy human post-editing, or multi-turn prompting can obscure AI fingerprints.
    2. **Short Texts**: Text samples with fewer than 50–100 words carry higher variance and reduced statistical confidence.
    3. **Formal & Technical Writing**: Highly structured human prose (e.g., legal briefs, corporate memos, scientific abstracts) naturally exhibits lower perplexity and formal syntax, which may occasionally trigger cautionary alerts.
    4. **Do Not Use for High-Stakes Disciplinary Action Alone**: Automated detectors should only serve as an auxiliary review tool in combination with human oversight and provenance tracking.
    """,
    icon="⚠️",
)
