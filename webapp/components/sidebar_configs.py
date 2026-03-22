"""Sidebar widgets that return config dataclasses."""

from __future__ import annotations

import streamlit as st

from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.synth.generator import GeneratorConfig
from src.separation.separator import SeparationConfig
from src.separation.temporal_filter import FilterConfig
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.gaussian_extractor import GaussianExtractorConfig
from src.separation.joint_model import JointModelConfig
from src.separation.decomposition import DecompositionConfig
from src.separation.subspace_separation import SubspaceConfig


def render_pulse_config() -> PulseConfig:
    with st.sidebar.expander("Pulse Model", expanded=True):
        return PulseConfig(
            heart_rate_bpm=st.slider("Heart Rate (BPM)", 40.0, 140.0, 72.0, 1.0),
            amplitude_mm=st.slider("Amplitude (mm)", 0.01, 1.0, 0.15, 0.01),
            artery_center_x_mm=st.slider("Artery Center X (mm)", -13.0, 13.0, 0.0, 0.5),
            artery_center_y_mm=st.slider("Artery Center Y (mm)", -18.0, 18.0, 0.0, 0.5),
            artery_angle_deg=st.slider("Artery Angle (deg)", -30.0, 30.0, 0.0, 1.0),
            sigma_mm=st.slider("Sigma Cross-Artery (mm)", 1.0, 8.0, 3.0, 0.5),
            sigma_along_mm=st.slider("Sigma Along-Artery (mm)", 2.0, 20.0, 8.0, 0.5),
            lateral_shear_ratio=st.slider("Lateral Shear Ratio", 0.0, 1.0, 0.3, 0.05),
            camera_tilt_deg=st.slider("Camera Tilt (deg)", 10.0, 70.0, 40.0, 5.0),
        )


def render_artifact_config() -> ArtifactConfig:
    with st.sidebar.expander("Artifact Model"):
        return ArtifactConfig(
            degree=st.selectbox("Polynomial Degree", [1, 2, 3], index=1),
            amplitude_mm=st.slider("Artifact Amplitude (mm)", 0.0, 5.0, 1.0, 0.1),
            max_freq_hz=st.slider("Max Frequency (Hz)", 0.1, 5.0, 2.0, 0.1),
            seed=st.number_input("Seed", value=42, step=1),
        )


def render_noise_config() -> NoiseConfig:
    with st.sidebar.expander("Noise Model"):
        return NoiseConfig(
            sigma_mm=st.slider("Noise Sigma (mm)", 0.0, 0.2, 0.02, 0.005),
            seed=st.number_input("Noise Seed", value=43, step=1),
        )


def render_generator_config() -> GeneratorConfig:
    with st.sidebar.expander("Grid & Timing"):
        num_rows = st.slider("Grid Rows", 5, 30, 19)
        num_cols = st.slider("Grid Columns", 5, 20, 14)
        num_frames = st.slider("Frames", 60, 900, 300, 30)
        fps = st.slider("FPS", 15.0, 120.0, 30.0, 5.0)
        spacing = st.slider("Grid Spacing (mm)", 1.0, 4.0, 2.0, 0.5)

    pulse = render_pulse_config()
    artifact = render_artifact_config()
    noise = render_noise_config()

    return GeneratorConfig(
        num_rows=num_rows, num_cols=num_cols,
        grid_spacing_mm=spacing, num_frames=num_frames, fps=fps,
        pulse=pulse, artifact=artifact, noise=noise,
    )


def render_separation_method() -> str:
    return st.sidebar.selectbox(
        "Separation Method",
        ["Sequential Polynomial", "Joint Model", "PCA", "ICA", "Subspace"],
    )


def render_separation_config(method: str):
    """Return the appropriate config for the selected method."""
    if method == "Sequential Polynomial":
        with st.sidebar.expander("Separation Config", expanded=True):
            degree = st.selectbox("Poly Degree", [1, 2, 3], index=1, key="sep_deg")
            reg = st.select_slider("Regularization", options=[1e-9, 1e-7, 1e-6, 1e-5, 1e-3, 0.01], value=1e-6)
            prefilter = st.checkbox("Temporal Pre-filter", value=False)
            gaussian = st.checkbox("Gaussian Extraction", value=True)
            n_iter = st.slider("Iterations", 1, 5, 1)

            filter_cfg = FilterConfig(lowpass_cutoff_hz=0.5)
            if prefilter:
                filter_cfg = FilterConfig(
                    lowpass_cutoff_hz=st.slider("Lowpass Cutoff (Hz)", 0.1, 5.0, 0.5, 0.1)
                )

            return SeparationConfig(
                polyfit=PolyFitConfig(degree=degree, regularization=reg),
                filter=filter_cfg,
                use_temporal_prefilter=prefilter,
                use_gaussian_extraction=gaussian,
                n_iterations=n_iter,
            )

    elif method == "Joint Model":
        with st.sidebar.expander("Joint Model Config", expanded=True):
            degree = st.selectbox("Poly Degree", [1, 2, 3], index=1, key="joint_deg")
            reg = st.select_slider("Regularization", options=[1e-9, 1e-7, 1e-6, 1e-5, 1e-3], value=1e-6, key="joint_reg")
            return JointModelConfig(poly_degree=degree, regularization=reg)

    elif method in ("PCA", "ICA"):
        with st.sidebar.expander(f"{method} Config", expanded=True):
            n_comp = st.slider("Components", 2, 50, 10, key=f"{method}_comp")
            return DecompositionConfig(n_components=n_comp, method=method.lower())

    elif method == "Subspace":
        with st.sidebar.expander("Subspace Config", expanded=True):
            n_comp = st.slider("Components", 2, 50, 15, key="sub_comp")
            threshold = st.slider("Mask Threshold", 0.0, 1.0, 0.3, 0.05)
            return SubspaceConfig(n_components=n_comp, mask_threshold=threshold)
