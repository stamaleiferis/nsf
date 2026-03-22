"""Page 5: Parameter Sweep

Sweep one parameter while holding others fixed, plot metric trends.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.metrics import evaluate, separation_snr
from webapp.components.plot_builders import plot_sweep

st.set_page_config(page_title="Parameter Sweep", layout="wide")
st.title("Parameter Sweep")

# Sidebar
st.sidebar.header("Sweep Config")
param = st.sidebar.selectbox("Parameter to Sweep", [
    "pulse_amplitude_mm", "artifact_amplitude_mm", "artifact_degree",
    "noise_sigma_mm", "heart_rate_bpm", "polyfit_degree", "artery_offset_mm",
])

defaults = {
    "pulse_amplitude_mm": (0.01, 0.5, 8),
    "artifact_amplitude_mm": (0.1, 5.0, 8),
    "artifact_degree": (1, 3, 3),
    "noise_sigma_mm": (0.001, 0.2, 8),
    "heart_rate_bpm": (50, 130, 6),
    "polyfit_degree": (1, 3, 3),
    "artery_offset_mm": (0.0, 5.0, 6),
}

d = defaults[param]
col1, col2, col3 = st.sidebar.columns(3)
v_min = col1.number_input("Min", value=float(d[0]))
v_max = col2.number_input("Max", value=float(d[1]))
n_points = col3.number_input("Points", value=d[2], min_value=2, max_value=20)

sep_degree = st.sidebar.selectbox("Separation Poly Degree", [1, 2, 3], index=1, key="sweep_sep")

if st.sidebar.button("Run Sweep", type="primary"):
    if param == "artifact_degree" or param == "polyfit_degree":
        values = list(range(int(v_min), int(v_max) + 1))
    else:
        values = np.linspace(v_min, v_max, int(n_points)).tolist()

    metrics = {"snr_improvement_db": [], "waveform_correlation": [], "artifact_residual_fraction": []}
    progress = st.progress(0)

    for i, val in enumerate(values):
        # Build configs with the swept parameter
        pulse_kw = {}
        artifact_kw = {"seed": 42}
        noise_kw = {"seed": 43}
        fit_degree = sep_degree

        if param == "pulse_amplitude_mm":
            pulse_kw["amplitude_mm"] = val
        elif param == "artifact_amplitude_mm":
            artifact_kw["amplitude_mm"] = val
        elif param == "artifact_degree":
            artifact_kw["degree"] = int(val)
        elif param == "noise_sigma_mm":
            noise_kw["sigma_mm"] = val
        elif param == "heart_rate_bpm":
            pulse_kw["heart_rate_bpm"] = val
        elif param == "polyfit_degree":
            fit_degree = int(val)
        elif param == "artery_offset_mm":
            pulse_kw["artery_center_x_mm"] = val

        gen_cfg = GeneratorConfig(
            num_frames=300, fps=30.0,
            pulse=PulseConfig(**pulse_kw),
            artifact=ArtifactConfig(**artifact_kw),
            noise=NoiseConfig(**noise_kw),
        )
        ds = generate(gen_cfg)
        gt = ds.ground_truth
        gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

        # For artery offset, use the default mask (not shifted)
        mask = gt.artery_mask

        sep_cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=fit_degree),
            use_temporal_prefilter=False,
            use_gaussian_extraction=True,
        )
        result = separate(ds.markers, gx, gy, mask, sep_cfg)

        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
        artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
        m = evaluate(result.recovered_pulse, result.estimated_artifact,
                     pulse_rel, artifact_rel, gt.artery_mask)
        raw_snr_val = separation_snr(ds.markers.displacements_from_rest(), pulse_rel, artifact_rel)

        metrics["snr_improvement_db"].append(m.separation_snr_db - raw_snr_val)
        metrics["waveform_correlation"].append(m.waveform_correlation)
        metrics["artifact_residual_fraction"].append(m.artifact_residual_fraction)

        progress.progress((i + 1) / len(values))

    st.session_state["sweep_values"] = values
    st.session_state["sweep_metrics"] = metrics
    st.session_state["sweep_param"] = param

if "sweep_values" not in st.session_state:
    st.info("Configure sweep and click **Run Sweep**.")
    st.stop()

values = st.session_state["sweep_values"]
metrics = st.session_state["sweep_metrics"]
param_name = st.session_state["sweep_param"]

st.subheader(f"Sweep: {param_name}")
fig = plot_sweep(values, metrics)
fig.update_layout(xaxis_title=param_name.replace("_", " "))
st.plotly_chart(fig, use_container_width=True)

# Artifact residual bar chart
import plotly.graph_objects as go
fig2 = go.Figure(go.Bar(
    x=[str(v) for v in values],
    y=[r * 100 for r in metrics["artifact_residual_fraction"]],
    marker_color="steelblue",
))
fig2.add_hline(y=5.0, line_dash="dash", line_color="red")
fig2.update_layout(title="Artifact Residual (%)", xaxis_title=param_name.replace("_", " "),
                   yaxis_title="%", height=350)
st.plotly_chart(fig2, use_container_width=True)
