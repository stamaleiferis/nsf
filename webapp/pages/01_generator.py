"""Page 1: Synthetic Data Generator

Generate data with full parameter control and visualize ground truth components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.synth.generator import generate
from src.synth.pulse import pulse_waveform, artery_mask
from webapp.components.sidebar_configs import render_generator_config
from webapp.components.plot_builders import (
    plot_pulse_waveform, plot_artery_mask, plot_signal_components,
    plot_heatmap, plot_quiver,
)
from webapp.components.state import set_dataset, get_dataset

st.set_page_config(page_title="Data Generator", layout="wide")
st.title("Synthetic Data Generator")

# Sidebar
st.sidebar.header("Parameters")
gen_cfg = render_generator_config()

if st.sidebar.button("Generate", type="primary"):
    with st.spinner("Generating..."):
        ds = generate(gen_cfg)
        set_dataset(ds)

ds = get_dataset()
if ds is None:
    st.info("Configure parameters and click **Generate**.")
    st.stop()

gt = ds.ground_truth
gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]
t = np.arange(ds.markers.num_frames) / ds.markers.fps

# --- Row 1: Waveform + Mask ---
col1, col2 = st.columns(2)
with col1:
    wf = pulse_waveform(t, gen_cfg.pulse)
    st.plotly_chart(plot_pulse_waveform(t, wf), use_container_width=True)
with col2:
    st.plotly_chart(plot_artery_mask(gt.artery_mask, gx, gy), use_container_width=True)

# --- Row 2: Signal components at selected marker ---
st.subheader("Signal Components at Selected Marker")
peak = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    marker_r = st.number_input("Row", 0, ds.markers.num_rows - 1, int(peak[0]))
with col_sel2:
    marker_c = st.number_input("Col", 0, ds.markers.num_cols - 1, int(peak[1]))

mask_val = gt.artery_mask[marker_r, marker_c]
label = f"({marker_r},{marker_c}) mask={mask_val:.3f}"

st.plotly_chart(plot_signal_components(
    t,
    gt.pulse_displacement[:, marker_r, marker_c, 1],
    gt.artifact_displacement[:, marker_r, marker_c, 1],
    gt.noise[:, marker_r, marker_c, 1],
    (gt.pulse_displacement + gt.artifact_displacement + gt.noise)[:, marker_r, marker_c, 1],
    marker_label=label,
), use_container_width=True)

# --- Row 3: 2D spatial components ---
st.subheader("2D Spatial Components")
frame = st.slider("Frame", 0, ds.markers.num_frames - 1, ds.markers.num_frames // 6,
                   key="gt_frame")

components = {
    "Pulse": gt.pulse_displacement[frame, ..., 1],
    "Artifact": gt.artifact_displacement[frame, ..., 1],
    "Noise": gt.noise[frame, ..., 1],
    "Total": (gt.pulse_displacement + gt.artifact_displacement + gt.noise)[frame, ..., 1],
}

cols = st.columns(4)
for col, (name, data) in zip(cols, components.items()):
    with col:
        st.plotly_chart(plot_heatmap(data, gx, gy, title=f"{name} Y-disp"), use_container_width=True)

# --- Row 4: Quiver ---
st.subheader("Displacement Vectors")
q_cols = st.columns(2)
with q_cols[0]:
    st.plotly_chart(plot_quiver(gx, gy, gt.pulse_displacement[frame], title="Pulse"), use_container_width=True)
with q_cols[1]:
    st.plotly_chart(plot_quiver(gx, gy, gt.artifact_displacement[frame], title="Artifact"), use_container_width=True)

# --- Info ---
with st.expander("Dataset Info"):
    st.write(f"Grid: {ds.markers.grid_shape}, Frames: {ds.markers.num_frames}, "
             f"Duration: {ds.markers.duration_sec:.1f}s, FPS: {ds.markers.fps}")
    st.write(f"Pulse RMS: {np.sqrt(np.mean(gt.pulse_displacement**2)):.4f} mm")
    st.write(f"Artifact RMS: {np.sqrt(np.mean(gt.artifact_displacement**2)):.4f} mm")
    st.write(f"Noise RMS: {np.sqrt(np.mean(gt.noise**2)):.4f} mm")
    st.write(f"Input SNR: {ds.separation_snr():.1f} dB")
