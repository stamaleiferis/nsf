"""Session state management for the webapp."""

from __future__ import annotations

import streamlit as st
import numpy as np

from src.data.markers import SyntheticDataset
from src.synth.generator import GeneratorConfig, generate


def get_dataset() -> SyntheticDataset | None:
    return st.session_state.get("dataset")


def set_dataset(ds: SyntheticDataset) -> None:
    st.session_state["dataset"] = ds
    # Clear downstream results when data changes
    for key in ["separation_result", "pipeline_result", "comparison_results"]:
        st.session_state.pop(key, None)


def ensure_dataset() -> SyntheticDataset | None:
    """Return existing dataset or offer quick generation."""
    ds = get_dataset()
    if ds is not None:
        return ds
    st.warning("No dataset generated yet.")
    if st.button("Quick Generate with Defaults"):
        ds = generate(GeneratorConfig(num_frames=300))
        set_dataset(ds)
        st.rerun()
    return None
