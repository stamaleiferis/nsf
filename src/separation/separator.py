"""Full signal separation pipeline.

Combines temporal filtering, spatial polynomial fitting, and artifact
subtraction to separate pulse from motion artifacts.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.data.markers import MarkerTimeSeries
from src.separation.temporal_filter import FilterConfig, lowpass_positions
from src.separation.polynomial_fit import (
    PolyFitConfig, fit_polynomial_all_frames, fit_polynomial_smooth_coeffs,
)
from src.separation.gaussian_extractor import (
    GaussianExtractorConfig, extract_pulse_gaussian,
)


@dataclass
class SeparationConfig:
    """Configuration for the full separation pipeline.

    Attributes:
        filter: Temporal filter configuration.
        polyfit: Polynomial fit configuration.
    """

    filter: FilterConfig | None = None
    polyfit: PolyFitConfig | None = None
    gaussian: GaussianExtractorConfig | None = None
    use_temporal_prefilter: bool = True
    use_gaussian_extraction: bool = False
    n_iterations: int = 1

    def __post_init__(self) -> None:
        if self.filter is None:
            self.filter = FilterConfig()
        if self.polyfit is None:
            self.polyfit = PolyFitConfig()
        if self.gaussian is None:
            self.gaussian = GaussianExtractorConfig()


@dataclass
class SeparationResult:
    """Result of signal separation.

    Attributes:
        recovered_pulse: (T, R, C, 2) estimated pulse displacement.
        estimated_artifact: (T, R, C, 2) estimated artifact displacement.
        lowpassed_displacements: (T, R, C, 2) temporally filtered displacements.
    """

    recovered_pulse: np.ndarray
    estimated_artifact: np.ndarray
    lowpassed_displacements: np.ndarray


def weights_from_artery_mask(artery_mask: np.ndarray) -> np.ndarray:
    """Convert artery probability mask to polynomial fitting weights.

    Markers far from the artery (low mask value) get high weight because
    they are pure artifact references. Markers near the artery get low
    weight to avoid pulse contamination of the artifact fit.

    Args:
        artery_mask: (R, C) artery influence mask in [0, 1].

    Returns:
        (R, C) fitting weights in [0, 1]. High = trusted for artifact fit.
    """
    return 1.0 - artery_mask


def separate(
    markers: MarkerTimeSeries,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    artery_mask: np.ndarray,
    config: SeparationConfig | None = None,
) -> SeparationResult:
    """Run the full signal separation pipeline.

    Steps:
    1. Compute displacements from rest (frame 0).
    2. Lowpass filter to remove pulse-band content.
    3. Fit 2D polynomial to lowpassed displacements (weighted by artery mask).
    4. Subtract polynomial artifact estimate from raw displacements.

    Args:
        markers: Observed marker time series.
        grid_x: (R, C) X coordinates of the rest-position grid in mm.
        grid_y: (R, C) Y coordinates of the rest-position grid in mm.
        artery_mask: (R, C) artery probability/influence mask in [0, 1].
        config: Separation pipeline configuration.

    Returns:
        SeparationResult with recovered pulse and estimated artifact.
    """
    if config is None:
        config = SeparationConfig()

    # Step 1: Displacements from rest
    displacements = markers.displacements_from_rest(rest_frame=0)
    weights = weights_from_artery_mask(artery_mask)

    # Iterative separation loop:
    #   1. Fit polynomial to (data - current_pulse_estimate)
    #   2. Subtract polynomial → residual
    #   3. Extract pulse from residual via Gaussian projection
    #   4. Repeat with refined pulse estimate
    current_pulse_estimate = np.zeros_like(displacements)

    for iteration in range(config.n_iterations):
        # Fit polynomial to data with pulse removed
        fit_data = displacements - current_pulse_estimate

        if config.use_temporal_prefilter:
            estimated_artifact = fit_polynomial_smooth_coeffs(
                grid_x, grid_y, fit_data, weights, config.polyfit,
                fps=markers.fps,
                smooth_cutoff_hz=config.filter.lowpass_cutoff_hz,
            )
        else:
            estimated_artifact = fit_polynomial_all_frames(
                grid_x, grid_y, fit_data, weights, config.polyfit
            )

        # Subtract artifact from original data
        residual = displacements - estimated_artifact

        # Extract pulse via Gaussian projection
        if config.use_gaussian_extraction or config.n_iterations > 1:
            current_pulse_estimate = extract_pulse_gaussian(
                residual, grid_x, grid_y, artery_mask, config.gaussian
            )
        else:
            current_pulse_estimate = residual

    recovered_pulse = current_pulse_estimate

    return SeparationResult(
        recovered_pulse=recovered_pulse,
        estimated_artifact=estimated_artifact,
        lowpassed_displacements=displacements,
    )
