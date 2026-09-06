"""Streamlit AI Text Detection Studio — Multi-page Application Entry Point.

Configures:
- Global wide-layout page configuration (st.set_page_config).
- Multi-page navigation linking:
    PAGE 1: "How It Works" (pages/1_How_It_Works.py)
    PAGE 2: "Detector" (pages/2_Detector.py)
- Sidebar navigation and branding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is in sys.path across all pages
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# ==============================================================================
# 1. GLOBAL APP CONFIGURATION
# ==============================================================================
# Technical requirement: Use st.set_page_config with a wide layout
st.set_page_config(
    page_title="AI Text Detection Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. MULTI-PAGE NAVIGATION SETUP (ST.NAVIGATION)
# ==============================================================================
# Define the two pages for sidebar navigation
pages = [
    st.Page(
        "pages/1_How_It_Works.py",
        title="How It Works",
        icon="📖",
        default=True,
    ),
    st.Page(
        "pages/2_Detector.py",
        title="Detector",
        icon="🔍",
        default=False,
    ),
]

pg = st.navigation(pages)

# ==============================================================================
# 3. SIDEBAR BRANDING & METADATA
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛡️ AI Text Studio")
    st.caption("Cascade Classifier + GPT-2 Perplexity")
    st.divider()
    st.markdown(
        """
        **Pages:**
        - **📖 How It Works**: Pipeline explanation & limitations
        - **🔍 Detector**: Real-time analysis with interactive Plotly charts
        """
    )
    st.divider()
    st.caption("v2.0 • Classical NLP + GPT-2 Triangulation")

# Run the selected page
pg.run()
