"""Processing layer — all computation, no GUI dependencies.

Every public function returns plain data (numpy arrays, dicts, dataclasses).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.markers import SyntheticDataset
from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig, pulse_waveform, artery_mask
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.temporal_filter import FilterConfig
from src.separation.gaussian_extractor import GaussianExtractorConfig
from src.separation.joint_model import JointModelConfig, joint_separate
from src.separation.decomposition import DecompositionConfig, decomposition_separate
from src.separation.subspace_separation import SubspaceConfig, subspace_separate
from src.separation.metrics import evaluate, separation_snr, SeparationMetrics
from src.estimation.pulse_extractor import estimate_pulse_snr_map
from src.estimation.bp_estimation import segment_beats, extract_beat_morphology
from src.pipeline import PipelineConfig, run_pipeline, RealTimeSeparator


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_rows: int = 19, num_cols: int = 14, num_frames: int = 300,
    fps: float = 30.0, spacing: float = 2.0,
    heart_rate: float = 72.0, pulse_amp: float = 0.15,
    artery_cx: float = 0.0, artery_cy: float = 0.0,
    artery_angle: float = 0.0, sigma_cross: float = 3.0,
    sigma_along: float = 8.0, shear_ratio: float = 0.3,
    camera_tilt: float = 40.0,
    artifact_degree: int = 2, artifact_amp: float = 1.0,
    artifact_freq: float = 2.0, artifact_seed: int = 42,
    noise_sigma: float = 0.02, noise_seed: int = 43,
) -> SyntheticDataset:
    """Generate a synthetic dataset with the given parameters."""
    cfg = GeneratorConfig(
        num_rows=num_rows, num_cols=num_cols,
        grid_spacing_mm=spacing, num_frames=num_frames, fps=fps,
        pulse=PulseConfig(
            heart_rate_bpm=heart_rate, amplitude_mm=pulse_amp,
            artery_center_x_mm=artery_cx, artery_center_y_mm=artery_cy,
            artery_angle_deg=artery_angle, sigma_mm=sigma_cross,
            sigma_along_mm=sigma_along, lateral_shear_ratio=shear_ratio,
            camera_tilt_deg=camera_tilt,
        ),
        artifact=ArtifactConfig(
            degree=artifact_degree, amplitude_mm=artifact_amp,
            max_freq_hz=artifact_freq, seed=artifact_seed,
        ),
        noise=NoiseConfig(sigma_mm=noise_sigma, seed=noise_seed),
    )
    return generate(cfg)


def get_generator_config_from_dataset(ds: SyntheticDataset) -> GeneratorConfig:
    """Recover the generator config stored in a dataset (for pulse waveform display)."""
    return GeneratorConfig(num_frames=ds.markers.num_frames, fps=ds.markers.fps)


# ---------------------------------------------------------------------------
# Signal separation
# ---------------------------------------------------------------------------

METHODS = ["Sequential Polynomial", "Joint Model", "PCA", "ICA", "Subspace"]


@dataclass
class SeparationResult:
    method: str
    rec_pulse: np.ndarray
    est_artifact: np.ndarray
    metrics: SeparationMetrics
    raw_snr: float
    snr_improvement: float


def run_separation(
    ds: SyntheticDataset,
    method: str = "Sequential Polynomial",
    poly_degree: int = 2,
    regularization: float = 1e-6,
    use_prefilter: bool = False,
    lowpass_cutoff: float = 0.5,
    use_gaussian: bool = True,
    n_iterations: int = 1,
    n_components: int = 10,
    mask_threshold: float = 0.3,
) -> SeparationResult:
    """Run a separation method and return results with metrics."""
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]
    disp = ds.markers.displacements_from_rest()

    if method == "Sequential Polynomial":
        cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=poly_degree, regularization=regularization),
            filter=FilterConfig(lowpass_cutoff_hz=lowpass_cutoff),
            use_temporal_prefilter=use_prefilter,
            use_gaussian_extraction=use_gaussian,
            n_iterations=n_iterations,
        )
        result = separate(ds.markers, gx, gy, gt.artery_mask, cfg)
        rec_pulse, est_artifact = result.recovered_pulse, result.estimated_artifact
    elif method == "Joint Model":
        cfg = JointModelConfig(poly_degree=poly_degree, regularization=regularization)
        rec_pulse, est_artifact = joint_separate(gx, gy, disp, gt.artery_mask, cfg)
    elif method in ("PCA", "ICA"):
        cfg = DecompositionConfig(n_components=n_components, method=method.lower())
        rec_pulse, est_artifact = decomposition_separate(disp, gt.artery_mask, cfg)
    elif method == "Subspace":
        cfg = SubspaceConfig(n_components=n_components, mask_threshold=mask_threshold)
        rec_pulse, est_artifact = subspace_separate(gx, gy, disp, gt.artery_mask, cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
    m = evaluate(rec_pulse, est_artifact, pulse_rel, artifact_rel, gt.artery_mask)
    raw_snr = separation_snr(disp, pulse_rel, artifact_rel)

    return SeparationResult(
        method=method, rec_pulse=rec_pulse, est_artifact=est_artifact,
        metrics=m, raw_snr=raw_snr,
        snr_improvement=m.separation_snr_db - raw_snr,
    )


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------

def run_comparison(
    ds: SyntheticDataset,
    methods: list[str] | None = None,
) -> dict[str, SeparationResult]:
    """Run multiple methods and return results dict."""
    if methods is None:
        methods = ["Sequential Polynomial", "Joint Model", "PCA"]

    results = {}
    for method in methods:
        results[method] = run_separation(ds, method=method)
    return results


# ---------------------------------------------------------------------------
# BP estimation
# ---------------------------------------------------------------------------

@dataclass
class BPResult:
    pipeline_result: Any  # from src.pipeline
    waveform: np.ndarray
    beats: list[tuple[int, int]]
    heart_rate: float
    confidence: float
    systolic: float
    diastolic: float
    timing: dict[str, float]


def run_bp_pipeline(
    ds: SyntheticDataset,
    poly_degree: int = 2,
    use_gaussian: bool = True,
    bp_low: float = 0.5,
    bp_high: float = 15.0,
) -> BPResult:
    """Run the full BP estimation pipeline."""
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

    pipeline_cfg = PipelineConfig(
        separation=SeparationConfig(
            polyfit=PolyFitConfig(degree=poly_degree),
            use_temporal_prefilter=False,
            use_gaussian_extraction=use_gaussian,
        ),
        bandpass=(bp_low, bp_high),
        grid_x_mm=gx, grid_y_mm=gy,
    )
    result = run_pipeline(ds.markers, pipeline_cfg)
    wf = result.pulse_extraction.waveform
    beats = segment_beats(wf, ds.markers.fps)
    bp = result.bp_estimate

    return BPResult(
        pipeline_result=result, waveform=wf, beats=beats,
        heart_rate=bp.heart_rate_bpm, confidence=bp.confidence,
        systolic=bp.systolic_mmhg, diastolic=bp.diastolic_mmhg,
        timing=result.timing_ms,
    )


def get_beat_morphology(waveform: np.ndarray, beat_start: int, beat_end: int, fps: float):
    """Extract morphology for a single beat."""
    return extract_beat_morphology(waveform, beat_start, beat_end, fps)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    param_name: str
    values: list[float]
    snr_improvement: list[float]
    waveform_correlation: list[float]
    artifact_residual: list[float]


SWEEP_DEFAULTS = {
    "pulse_amplitude_mm": (0.01, 0.5, 8),
    "artifact_amplitude_mm": (0.1, 5.0, 8),
    "artifact_degree": (1, 3, 3),
    "noise_sigma_mm": (0.001, 0.2, 8),
    "heart_rate_bpm": (50, 130, 6),
    "polyfit_degree": (1, 3, 3),
    "artery_offset_mm": (0.0, 5.0, 6),
}


def run_sweep(
    param: str,
    v_min: float, v_max: float, n_points: int,
    sep_degree: int = 2,
    progress_callback=None,
) -> SweepResult:
    """Sweep one parameter, return metric trends."""
    if param in ("artifact_degree", "polyfit_degree"):
        values = list(range(int(v_min), int(v_max) + 1))
    else:
        values = np.linspace(v_min, v_max, int(n_points)).tolist()

    snr_imp, wf_corr, art_res = [], [], []

    for i, val in enumerate(values):
        pulse_kw, artifact_kw, noise_kw = {}, {"seed": 42}, {"seed": 43}
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

        sep_cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=fit_degree),
            use_temporal_prefilter=False,
            use_gaussian_extraction=True,
        )
        result = separate(ds.markers, gx, gy, gt.artery_mask, sep_cfg)

        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
        artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
        m = evaluate(result.recovered_pulse, result.estimated_artifact,
                     pulse_rel, artifact_rel, gt.artery_mask)
        raw_snr = separation_snr(ds.markers.displacements_from_rest(), pulse_rel, artifact_rel)

        snr_imp.append(m.separation_snr_db - raw_snr)
        wf_corr.append(m.waveform_correlation)
        art_res.append(m.artifact_residual_fraction)

        if progress_callback:
            progress_callback(i + 1, len(values))

    return SweepResult(
        param_name=param, values=values,
        snr_improvement=snr_imp, waveform_correlation=wf_corr,
        artifact_residual=art_res,
    )


# ---------------------------------------------------------------------------
# Real-time latency measurement
# ---------------------------------------------------------------------------

@dataclass
class RealtimeResult:
    latencies: list[float]
    grid_shape: tuple[int, int]
    n_markers: int


def run_realtime(ds: SyntheticDataset, n_frames: int = 300,
                 progress_callback=None) -> RealtimeResult:
    """Measure per-frame processing latency."""
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]
    n_frames = min(n_frames, ds.markers.num_frames)

    rt = RealTimeSeparator(gx, gy, gt.artery_mask)
    latencies = []

    for i in range(n_frames):
        _, lat = rt.process_frame(ds.markers.positions[i])
        latencies.append(lat)
        if progress_callback and i % 30 == 0:
            progress_callback(i + 1, n_frames)

    if progress_callback:
        progress_callback(n_frames, n_frames)

    return RealtimeResult(
        latencies=latencies,
        grid_shape=ds.markers.grid_shape,
        n_markers=ds.markers.grid_shape[0] * ds.markers.grid_shape[1],
    )
