"""Gaussian spatial model extraction for pulse refinement.

After polynomial artifact subtraction, the residual contains pulse signal
plus non-polynomial artifact residual. The pulse has a known spatial structure
(Gaussian centered on artery), so we can fit this model to extract only the
pulse-shaped component and reject spatially inconsistent residuals.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class GaussianExtractorConfig:
    """Configuration for Gaussian pulse extraction.

    Attributes:
        use_known_mask: If True, project residual onto known artery mask.
            If False, fit Gaussian parameters from the data.
    """

    use_known_mask: bool = True


def extract_pulse_gaussian(
    residual: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    artery_mask: np.ndarray,
    config: GaussianExtractorConfig | None = None,
) -> np.ndarray:
    """Extract the pulse component from residual using Gaussian spatial model.

    Projects the residual displacement field at each frame onto the known
    artery spatial pattern. This removes spatially inconsistent artifact
    residuals that the polynomial fit couldn't capture.

    The pulse displacement at each frame should be:
        d_pulse(x, y, t) = A(t) * mask(x, y) * direction

    where A(t) is the temporal amplitude and mask(x, y) is the spatial pattern.
    We estimate A(t) by weighted least squares projection of the residual
    onto the mask pattern.

    Args:
        residual: (T, R, C, 2) residual displacement after artifact subtraction.
        grid_x: (R, C) X coordinates.
        grid_y: (R, C) Y coordinates.
        artery_mask: (R, C) artery influence mask in [0, 1].
        config: Extraction configuration.

    Returns:
        (T, R, C, 2) pulse displacement estimate.
    """
    if config is None:
        config = GaussianExtractorConfig()

    T, R, C, _ = residual.shape
    mask = artery_mask  # (R, C)

    # Normalize mask for projection
    mask_norm = mask / (np.sum(mask ** 2) + 1e-12)

    result = np.empty_like(residual)

    for ax in range(2):
        # Residual for this axis: (T, R, C)
        res_ax = residual[:, :, :, ax]

        # Project each frame's spatial pattern onto the mask
        # A(t) = sum(residual * mask) / sum(mask^2) for each frame
        # This is the weighted average displacement at artery markers
        amplitude = np.einsum("trc,rc->t", res_ax, mask_norm)  # (T,)

        # Reconstruct: pulse = A(t) * mask(x,y)
        result[:, :, :, ax] = amplitude[:, None, None] * mask[None, :, :]

    return result
