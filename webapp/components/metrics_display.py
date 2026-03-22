"""Metric cards and display helpers."""

from __future__ import annotations

import streamlit as st
import numpy as np

from src.separation.metrics import SeparationMetrics


def display_metrics(m: SeparationMetrics, raw_snr: float = 0.0) -> None:
    """Display 4 metric cards in a row."""
    cols = st.columns(4)
    snr_imp = m.separation_snr_db - raw_snr

    cols[0].metric("SNR Improvement", f"{snr_imp:.1f} dB")
    cols[1].metric("Waveform Corr", f"{m.waveform_correlation:.3f}")
    cols[2].metric("Spatial Corr", f"{m.spatial_correlation:.3f}")
    cols[3].metric("Artifact Residual", f"{m.artifact_residual_fraction:.4f}")


def display_timing(timing: dict[str, float]) -> None:
    """Display per-stage timing."""
    cols = st.columns(len(timing))
    for col, (name, ms) in zip(cols, timing.items()):
        col.metric(name.replace("_ms", "").replace("_", " ").title(), f"{ms:.1f} ms")
