"""Page 4: BP Estimation

Full pipeline: separation -> pulse extraction -> beat segmentation -> BP estimate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.pipeline import PipelineConfig, run_pipeline
from src.separation.separator import SeparationConfig
from src.separation.polynomial_fit import PolyFitConfig
from src.estimation.bp_estimation import segment_beats, extract_beat_morphology, extract_features, FEATURE_NAMES
from webapp.components.plot_builders import plot_beat_morphology
from webapp.components.metrics_display import display_timing
from webapp.components.state import ensure_dataset

st.set_page_config(page_title="BP Estimation", layout="wide")
st.title("BP Estimation Pipeline")

ds = ensure_dataset()
if ds is None:
    st.stop()

# Sidebar
st.sidebar.header("Pipeline Config")
with st.sidebar.expander("Separation"):
    degree = st.selectbox("Poly Degree", [1, 2, 3], index=1, key="bp_deg")
    gaussian = st.checkbox("Gaussian Extraction", value=True, key="bp_gauss")

with st.sidebar.expander("Pulse Extraction"):
    bp_low = st.slider("Bandpass Low (Hz)", 0.1, 2.0, 0.5, 0.1)
    bp_high = st.slider("Bandpass High (Hz)", 5.0, 20.0, 15.0, 1.0)
    n_best = st.slider("N Best Markers", 3, 30, 10)

if st.sidebar.button("Run Pipeline", type="primary"):
    with st.spinner("Running full pipeline..."):
        gt = ds.ground_truth
        pipeline_cfg = PipelineConfig(
            separation=SeparationConfig(
                polyfit=PolyFitConfig(degree=degree),
                use_temporal_prefilter=False,
                use_gaussian_extraction=gaussian,
            ),
            bandpass=(bp_low, bp_high),
            grid_x_mm=gt.rest_positions[..., 0],
            grid_y_mm=gt.rest_positions[..., 1],
        )
        result = run_pipeline(ds.markers, pipeline_cfg)
        st.session_state["pipeline_result"] = result

if "pipeline_result" not in st.session_state:
    st.info("Configure and click **Run Pipeline**.")
    st.stop()

result = st.session_state["pipeline_result"]
pr = result.pulse_extraction
bp = result.bp_estimate

# Timing
st.subheader("Pipeline Timing")
display_timing(result.timing_ms)

# HR and BP
col1, col2, col3, col4 = st.columns(4)
col1.metric("Heart Rate", f"{bp.heart_rate_bpm:.0f} BPM")
col2.metric("Confidence", f"{bp.confidence:.2f}")
col3.metric("Systolic BP", f"{bp.systolic_mmhg:.0f} mmHg" if bp.systolic_mmhg > 0 else "N/A (uncalibrated)")
col4.metric("Diastolic BP", f"{bp.diastolic_mmhg:.0f} mmHg" if bp.diastolic_mmhg > 0 else "N/A (uncalibrated)")

# Pulse waveform with beat markers
st.subheader("Extracted Pulse Waveform")
wf = pr.waveform
t = np.arange(len(wf)) / ds.markers.fps
beats = segment_beats(wf, ds.markers.fps)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=wf, mode="lines", line=dict(color="blue", width=1.5), name="Pulse"))
for s, e in beats:
    fig.add_vline(x=t[s], line_dash="dot", line_color="green", opacity=0.4)
fig.update_layout(title=f"Extracted Pulse ({len(beats)} beats detected)",
                  xaxis_title="Time (s)", yaxis_title="Amplitude", height=350)
st.plotly_chart(fig, use_container_width=True)

# Beat morphology
if beats:
    st.subheader("Beat Morphology")
    beat_idx = st.slider("Beat #", 0, len(beats) - 1, len(beats) // 2)
    s, e = beats[beat_idx]
    morph = extract_beat_morphology(wf, s, e, ds.markers.fps)
    t_beat = np.arange(e - s) / ds.markers.fps

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_beat_morphology(t_beat, wf[s:e], morph), use_container_width=True)

    with col2:
        st.markdown("**Morphology Features**")
        st.write(f"Systolic Peak: {morph.systolic_peak_value:.4f}")
        st.write(f"Dicrotic Notch: {morph.dicrotic_notch_value:.4f}")
        st.write(f"Diastolic Peak: {morph.diastolic_peak_value:.4f}")
        st.write(f"Sys/Dias Ratio: {morph.systolic_diastolic_ratio:.2f}")
        st.write(f"Augmentation Index: {morph.augmentation_index:.3f}")
        st.write(f"Rise Time: {morph.rise_time:.4f} s")
        st.write(f"Beat Duration: {morph.beat_duration:.4f} s")
        st.write(f"AUC: {morph.area_under_curve:.6f}")

    # Beat feature table
    if len(beats) > 1:
        st.subheader("All Beats")
        import pandas as pd
        rows = []
        for i, (s, e) in enumerate(beats):
            m = extract_beat_morphology(wf, s, e, ds.markers.fps)
            rows.append({
                "Beat": i,
                "Duration (s)": f"{m.beat_duration:.3f}",
                "Systolic": f"{m.systolic_peak_value:.4f}",
                "Dicrotic": f"{m.dicrotic_notch_value:.4f}",
                "Diastolic": f"{m.diastolic_peak_value:.4f}",
                "Aug Index": f"{m.augmentation_index:.3f}",
                "Rise Time (s)": f"{m.rise_time:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
