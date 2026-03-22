"""End-to-end pipeline: marker positions → separated pulse → BP estimate.

Provides a single entry point that chains data loading, signal separation,
pulse extraction, and BP estimation. Designed for both batch and real-time use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data.markers import MarkerTimeSeries
from src.separation.separator import SeparationConfig, SeparationResult, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.gaussian_extractor import GaussianExtractorConfig
from src.estimation.pulse_extractor import PulseExtractionResult, extract_pulse
from src.estimation.bp_estimation import BPEstimate, BPCalibration, estimate_bp
from src.synth.pulse import artery_mask as compute_artery_mask, PulseConfig


@dataclass
class PipelineConfig:
    """Configuration for the end-to-end pipeline.

    Attributes:
        separation: Signal separation configuration.
        artery_prior: Prior knowledge of artery location (PulseConfig for mask).
        bandpass: Bandpass filter range for pulse extraction.
        bp_calibration: Optional BP calibration model.
        grid_x_mm: (R, C) X coordinates of marker grid in mm.
        grid_y_mm: (R, C) Y coordinates of marker grid in mm.
    """

    separation: SeparationConfig = field(default_factory=lambda: SeparationConfig(
        polyfit=PolyFitConfig(degree=2),
        use_temporal_prefilter=False,
        use_gaussian_extraction=True,
    ))
    artery_prior: PulseConfig = field(default_factory=PulseConfig)
    bandpass: tuple[float, float] = (0.5, 15.0)
    bp_calibration: BPCalibration | None = None
    grid_x_mm: np.ndarray | None = None
    grid_y_mm: np.ndarray | None = None


@dataclass
class PipelineResult:
    """Result of the end-to-end pipeline.

    Attributes:
        separation: Signal separation result.
        pulse_extraction: Pulse waveform extraction result.
        bp_estimate: Blood pressure estimate (if calibrated).
        artery_mask: Artery mask used for separation.
        timing_ms: Dict of per-stage timing in milliseconds.
    """

    separation: SeparationResult
    pulse_extraction: PulseExtractionResult
    bp_estimate: BPEstimate
    artery_mask: np.ndarray
    timing_ms: dict[str, float] = field(default_factory=dict)


def make_default_grid(num_rows: int = 19, num_cols: int = 14, spacing_mm: float = 2.0):
    """Create a default centered marker grid."""
    xs = np.arange(num_cols) * spacing_mm
    ys = np.arange(num_rows) * spacing_mm
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    return np.meshgrid(xs, ys)


def run_pipeline(
    markers: MarkerTimeSeries,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full end-to-end pipeline.

    Args:
        markers: Input marker time series.
        config: Pipeline configuration.

    Returns:
        PipelineResult with all intermediate and final results.
    """
    if config is None:
        config = PipelineConfig()

    timing = {}

    # Grid coordinates
    if config.grid_x_mm is not None:
        grid_x = config.grid_x_mm
        grid_y = config.grid_y_mm
    else:
        grid_x, grid_y = make_default_grid(markers.num_rows, markers.num_cols)

    # Artery mask
    t0 = time.perf_counter()
    mask = compute_artery_mask(grid_x, grid_y, config.artery_prior)
    timing["artery_mask_ms"] = (time.perf_counter() - t0) * 1000

    # Signal separation
    t0 = time.perf_counter()
    sep_result = separate(markers, grid_x, grid_y, mask, config.separation)
    timing["separation_ms"] = (time.perf_counter() - t0) * 1000

    # Pulse extraction
    t0 = time.perf_counter()
    pulse_result = extract_pulse(
        sep_result.recovered_pulse, markers.fps, mask,
        bandpass=config.bandpass,
    )
    timing["pulse_extraction_ms"] = (time.perf_counter() - t0) * 1000

    # BP estimation
    t0 = time.perf_counter()
    bp_result = estimate_bp(
        pulse_result.waveform, markers.fps, config.bp_calibration
    )
    timing["bp_estimation_ms"] = (time.perf_counter() - t0) * 1000

    timing["total_ms"] = sum(timing.values())

    return PipelineResult(
        separation=sep_result,
        pulse_extraction=pulse_result,
        bp_estimate=bp_result,
        artery_mask=mask,
        timing_ms=timing,
    )


class RealTimeSeparator:
    """Frame-by-frame separator for real-time operation.

    Maintains state across frames and processes each new frame
    with minimal latency. Uses a sliding window for temporal context.
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        artery_mask: np.ndarray,
        config: SeparationConfig | None = None,
        window_size: int = 30,
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.artery_mask = artery_mask
        self.config = config or SeparationConfig(
            polyfit=PolyFitConfig(degree=2),
            use_temporal_prefilter=False,
        )
        self.window_size = window_size

        R, C = grid_x.shape
        self._buffer = np.zeros((window_size, R, C, 2))
        self._frame_count = 0
        self._rest_frame: np.ndarray | None = None

        # Precompute polynomial fit matrix
        from src.separation.polynomial_fit import _build_design_matrix
        from src.separation.separator import weights_from_artery_mask

        x_flat = grid_x.ravel()
        y_flat = grid_y.ravel()
        x_range = x_flat.max() - x_flat.min() or 1.0
        y_range = y_flat.max() - y_flat.min() or 1.0
        x_norm = 2.0 * (x_flat - x_flat.mean()) / x_range
        y_norm = 2.0 * (y_flat - y_flat.mean()) / y_range

        A = _build_design_matrix(x_norm, y_norm, self.config.polyfit.degree)
        M = A.shape[1]
        w = weights_from_artery_mask(artery_mask).ravel()
        W = np.diag(w)
        WA = W @ A
        reg = self.config.polyfit.regularization * np.eye(M)
        self._solve_matrix = np.linalg.solve(WA.T @ WA + reg, WA.T @ W)
        self._A = A
        self._mask_flat = artery_mask.ravel()
        self._mask_norm = self._mask_flat / (np.sum(self._mask_flat ** 2) + 1e-12)

    def process_frame(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        """Process a single frame of marker positions.

        Args:
            positions: (R, C, 2) marker positions for this frame.

        Returns:
            (pulse_displacement, latency_ms) — (R, C, 2) pulse estimate and timing.
        """
        t0 = time.perf_counter()

        R, C, _ = positions.shape

        if self._rest_frame is None:
            self._rest_frame = positions.copy()

        displacement = positions - self._rest_frame  # (R, C, 2)

        # Polynomial artifact estimate
        artifact = np.empty((R, C, 2))
        for ax in range(2):
            d = displacement[:, :, ax].ravel()
            coeffs = self._solve_matrix @ d
            artifact[:, :, ax] = (self._A @ coeffs).reshape(R, C)

        # Residual
        residual = displacement - artifact

        # Gaussian pulse extraction (project onto mask)
        pulse = np.empty((R, C, 2))
        mask = self.artery_mask
        for ax in range(2):
            amplitude = np.sum(residual[:, :, ax].ravel() * self._mask_norm)
            pulse[:, :, ax] = amplitude * mask

        latency_ms = (time.perf_counter() - t0) * 1000
        self._frame_count += 1

        return pulse, latency_ms
