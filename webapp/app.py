"""NSF Signal Separation — Interactive Demo

Main page with project overview.
Run: streamlit run webapp/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="NSF Signal Separation",
    page_icon="pulse",
    layout="wide",
)

st.title("NSF -- Arterial Pulse Tonometry: Signal Separation")

st.markdown("""
### Project Overview

Non-invasive blood pressure monitoring via optical tonometry. A camera observes a
deformable elastomer surface with **266 fiducial markers** (19x14 grid, 2mm spacing)
placed over the radial artery at the wrist.

The surface deforms from two sources:
1. **Arterial pulse** -- spatially localized Gaussian (sigma ~ 3mm cross, 8mm along)
2. **Motion artifacts** -- spatially global, smooth deformations

This app demonstrates the separation pipeline and lets you explore all parameters interactively.

---

### Pages

| Page | Description |
|------|-------------|
| **Data Generator** | Generate synthetic data with full parameter control |
| **Signal Separation** | Run separation, inspect before/after at any marker |
| **Method Comparison** | Compare all 5 methods side-by-side |
| **BP Estimation** | Full pipeline: separation -> pulse extraction -> BP |
| **Parameter Sweep** | Sweep one parameter, plot metric trends |
| **Real-Time Demo** | Frame-by-frame processing with latency measurement |

---

### Key Results

| Metric | Polynomial Artifacts | Physics-Based |
|--------|---------------------|---------------|
| SNR Improvement | 27 dB | 24.5 dB |
| Waveform Correlation | 0.984 | 0.32 (variable) |
| Artifact Residual | 0.007% | 65% |
| Real-Time Latency | < 1 ms/frame | |
""")
