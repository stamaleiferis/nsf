"""Page 3: Method Comparison

Run multiple separation methods side-by-side on the same data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.joint_model import JointModelConfig, joint_separate
from src.separation.decomposition import DecompositionConfig, decomposition_separate
from src.separation.subspace_separation import SubspaceConfig, subspace_separate
from src.separation.metrics import evaluate, separation_snr, waveform_correlation
from webapp.components.plot_builders import plot_method_comparison_bars
from webapp.components.state import ensure_dataset

st.set_page_config(page_title="Method Comparison", layout="wide")
st.title("Method Comparison")

ds = ensure_dataset()
if ds is None:
    st.stop()

gt = ds.ground_truth
gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]
disp = ds.markers.displacements_from_rest()
pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
raw_snr = separation_snr(disp, pulse_rel, artifact_rel)

# Method selection
methods = st.sidebar.multiselect(
    "Methods to compare",
    ["Sequential Polynomial", "Joint Model", "PCA", "ICA", "Subspace"],
    default=["Sequential Polynomial", "Joint Model", "PCA"],
)

if st.sidebar.button("Run All", type="primary"):
    results = {}
    progress = st.progress(0)
    for i, method in enumerate(methods):
        with st.spinner(f"Running {method}..."):
            if method == "Sequential Polynomial":
                cfg = SeparationConfig(polyfit=PolyFitConfig(degree=2),
                                       use_temporal_prefilter=False, use_gaussian_extraction=True)
                r = separate(ds.markers, gx, gy, gt.artery_mask, cfg)
                rp, ea = r.recovered_pulse, r.estimated_artifact
            elif method == "Joint Model":
                rp, ea = joint_separate(gx, gy, disp, gt.artery_mask)
            elif method == "PCA":
                rp, ea = decomposition_separate(disp, gt.artery_mask, DecompositionConfig(method="pca"))
            elif method == "ICA":
                rp, ea = decomposition_separate(disp, gt.artery_mask, DecompositionConfig(method="ica"))
            elif method == "Subspace":
                rp, ea = subspace_separate(gx, gy, disp, gt.artery_mask)

            m = evaluate(rp, ea, pulse_rel, artifact_rel, gt.artery_mask)
            results[method] = {
                "snr_improvement_db": m.separation_snr_db - raw_snr,
                "waveform_correlation": m.waveform_correlation,
                "spatial_correlation": m.spatial_correlation,
                "artifact_residual_fraction": m.artifact_residual_fraction,
                "recovered_pulse": rp,
            }
        progress.progress((i + 1) / len(methods))

    st.session_state["comparison_results"] = results

if "comparison_results" not in st.session_state:
    st.info("Select methods and click **Run All**.")
    st.stop()

results = st.session_state["comparison_results"]

# Metrics table
st.subheader("Metrics")
import pandas as pd
rows = []
for name, r in results.items():
    rows.append({
        "Method": name,
        "SNR Improvement (dB)": f"{r['snr_improvement_db']:.1f}",
        "Waveform Corr": f"{r['waveform_correlation']:.3f}",
        "Spatial Corr": f"{r['spatial_correlation']:.3f}",
        "Artifact Residual": f"{r['artifact_residual_fraction']:.4f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Bar charts
st.subheader("Comparison Charts")
st.plotly_chart(plot_method_comparison_bars(results), use_container_width=True)

# Waveform overlay at artery center
st.subheader("Recovered Pulse Overlay (Artery Center)")
peak = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
r, c = peak
t = np.arange(ds.markers.num_frames) / ds.markers.fps

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=pulse_rel[:, r, c, 1], name="GT Pulse",
                          line=dict(color="black", width=2, dash="dash")))
colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
for i, (name, res) in enumerate(results.items()):
    fig.add_trace(go.Scatter(x=t, y=res["recovered_pulse"][:, r, c, 1], name=name,
                              line=dict(color=colors[i % len(colors)], width=1.5)))
fig.update_layout(title=f"Recovered Pulse at ({r},{c})", xaxis_title="Time (s)",
                  yaxis_title="Y displacement (mm)", height=400)
st.plotly_chart(fig, use_container_width=True)
