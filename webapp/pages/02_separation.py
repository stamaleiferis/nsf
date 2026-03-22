"""Page 2: Signal Separation

Run a separation method, inspect results with metrics and before/after plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.separation.separator import SeparationConfig, separate
from src.separation.joint_model import joint_separate
from src.separation.decomposition import decomposition_separate
from src.separation.subspace_separation import subspace_separate
from src.separation.metrics import evaluate, separation_snr
from src.estimation.pulse_extractor import estimate_pulse_snr_map
from webapp.components.sidebar_configs import render_separation_method, render_separation_config
from webapp.components.metrics_display import display_metrics
from webapp.components.plot_builders import (
    plot_before_after, plot_pulse_zoom, plot_heatmap, plot_quiver,
)
from webapp.components.state import ensure_dataset

st.set_page_config(page_title="Signal Separation", layout="wide")
st.title("Signal Separation")

ds = ensure_dataset()
if ds is None:
    st.stop()

gt = ds.ground_truth
gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

# Sidebar
st.sidebar.header("Separation Method")
method = render_separation_method()
config = render_separation_config(method)

if st.sidebar.button("Run Separation", type="primary"):
    with st.spinner(f"Running {method}..."):
        disp = ds.markers.displacements_from_rest()

        if method == "Sequential Polynomial":
            result = separate(ds.markers, gx, gy, gt.artery_mask, config)
            rec_pulse = result.recovered_pulse
            est_artifact = result.estimated_artifact
        elif method == "Joint Model":
            rec_pulse, est_artifact = joint_separate(gx, gy, disp, gt.artery_mask, config)
        elif method in ("PCA", "ICA"):
            rec_pulse, est_artifact = decomposition_separate(disp, gt.artery_mask, config)
        elif method == "Subspace":
            rec_pulse, est_artifact = subspace_separate(gx, gy, disp, gt.artery_mask, config)

        st.session_state["sep_rec_pulse"] = rec_pulse
        st.session_state["sep_est_artifact"] = est_artifact
        st.session_state["sep_method"] = method

# Check for results
if "sep_rec_pulse" not in st.session_state:
    st.info("Select a method and click **Run Separation**.")
    st.stop()

rec_pulse = st.session_state["sep_rec_pulse"]
est_artifact = st.session_state["sep_est_artifact"]

# Relative ground truth
pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
raw = ds.markers.displacements_from_rest()

# Metrics
m = evaluate(rec_pulse, est_artifact, pulse_rel, artifact_rel, gt.artery_mask)
raw_snr = separation_snr(raw, pulse_rel, artifact_rel)

st.subheader(f"Results: {st.session_state.get('sep_method', '')}")
display_metrics(m, raw_snr)

# Tabs
tab1d, tab2d, tab_evo, tab_snr = st.tabs(["1D Time Series", "2D Spatial", "Temporal Evolution", "SNR Map"])

t = np.arange(ds.markers.num_frames) / ds.markers.fps

with tab1d:
    # Marker selector
    peak = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
    presets = {
        "Artery Center": peak,
        "Custom": None,
    }
    # Find edge and off-artery
    edge_cands = np.argwhere((gt.artery_mask > 0.2) & (gt.artery_mask < 0.5))
    if len(edge_cands) > 0:
        dists = np.abs(edge_cands[:, 0] - peak[0])
        presets["Artery Edge"] = tuple(edge_cands[np.argmin(dists)])
    off_col = np.argmin(gt.artery_mask[peak[0], :])
    presets["Off-Artery"] = (peak[0], off_col)

    preset = st.selectbox("Marker", list(presets.keys()))
    if preset == "Custom":
        c1, c2 = st.columns(2)
        mr = c1.number_input("Row", 0, ds.markers.num_rows - 1, int(peak[0]), key="sep_mr")
        mc = c2.number_input("Col", 0, ds.markers.num_cols - 1, int(peak[1]), key="sep_mc")
    else:
        mr, mc = presets[preset]

    label = f"({mr},{mc}) mask={gt.artery_mask[mr, mc]:.3f}"

    # Before/after
    st.plotly_chart(plot_before_after(
        t, raw[:, mr, mc, 1],
        rec_pulse[:, mr, mc, 1], pulse_rel[:, mr, mc, 1],
        est_artifact[:, mr, mc, 1], artifact_rel[:, mr, mc, 1],
        marker_label=label,
    ), use_container_width=True)

    # Pulse zoom
    beat_frames = int(ds.markers.fps / (72.0 / 60.0) * 2.5)
    f_start = ds.markers.num_frames // 5
    f_end = min(f_start + beat_frames, ds.markers.num_frames)
    st.plotly_chart(plot_pulse_zoom(
        t[f_start:f_end],
        pulse_rel[f_start:f_end, mr, mc, 1],
        rec_pulse[f_start:f_end, mr, mc, 1],
        marker_label=label,
    ), use_container_width=True)

with tab2d:
    frame = st.slider("Frame", 0, ds.markers.num_frames - 1, ds.markers.num_frames // 6, key="sep_frame")

    st.markdown("**Y-Displacement Heatmaps**")
    cols = st.columns(3)
    with cols[0]:
        st.plotly_chart(plot_heatmap(pulse_rel[frame, ..., 1], gx, gy, "GT Pulse"), use_container_width=True)
    with cols[1]:
        st.plotly_chart(plot_heatmap(rec_pulse[frame, ..., 1], gx, gy, "Recovered Pulse"), use_container_width=True)
    with cols[2]:
        error = rec_pulse[frame, ..., 1] - pulse_rel[frame, ..., 1]
        st.plotly_chart(plot_heatmap(error, gx, gy, "Pulse Error"), use_container_width=True)

    cols2 = st.columns(3)
    with cols2[0]:
        st.plotly_chart(plot_heatmap(artifact_rel[frame, ..., 1], gx, gy, "GT Artifact"), use_container_width=True)
    with cols2[1]:
        st.plotly_chart(plot_heatmap(est_artifact[frame, ..., 1], gx, gy, "Estimated Artifact"), use_container_width=True)
    with cols2[2]:
        error_a = est_artifact[frame, ..., 1] - artifact_rel[frame, ..., 1]
        st.plotly_chart(plot_heatmap(error_a, gx, gy, "Artifact Error"), use_container_width=True)

with tab_evo:
    st.markdown("**Pulse Spatial Pattern Over 1 Cardiac Cycle**")
    beat_start = ds.markers.num_frames // 5
    n_steps = 6
    step = max(int(ds.markers.fps / (72.0 / 60.0) / n_steps), 1)
    frames = [beat_start + i * step for i in range(n_steps) if beat_start + i * step < ds.markers.num_frames]

    for row_label, data in [("GT Pulse", pulse_rel), ("Recovered", rec_pulse)]:
        st.markdown(f"**{row_label}**")
        cols = st.columns(len(frames))
        for col, f in zip(cols, frames):
            with col:
                st.plotly_chart(plot_heatmap(
                    data[f, ..., 1], gx, gy,
                    title=f"t={f/ds.markers.fps:.2f}s",
                ), use_container_width=True)

with tab_snr:
    snr_map = estimate_pulse_snr_map(raw, ds.markers.fps)
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(plot_heatmap(gt.artery_mask, gx, gy, "True Artery Mask",
                                     colorscale="Hot", symmetric=False), use_container_width=True)
    with cols[1]:
        st.plotly_chart(plot_heatmap(snr_map, gx, gy, "Pulse SNR Map",
                                     colorscale="Viridis", symmetric=False), use_container_width=True)
