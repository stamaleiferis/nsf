"""Fit spatial Gaussian parameters from marker data.

Estimates the artery's spatial parameters (center, width, angle, amplitude ratios)
from observed marker displacements using the known Gaussian pulse model.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class SpatialFitResult:
    """Result of fitting Gaussian spatial model to displacement data.

    Attributes:
        center_x_mm: Estimated artery center X in mm.
        center_y_mm: Estimated artery center Y in mm.
        sigma_mm: Estimated Gaussian width in mm.
        angle_deg: Estimated artery angle from Y-axis in degrees.
        amplitude_y: Peak Y-displacement amplitude in mm.
        amplitude_x: Peak X-displacement amplitude in mm.
        lateral_shear_ratio: Ratio of X to Y displacement.
        residual: Fit residual (lower is better).
        fitted_mask: (R, C) fitted artery mask.
    """

    center_x_mm: float
    center_y_mm: float
    sigma_mm: float
    angle_deg: float
    amplitude_y: float
    amplitude_x: float
    lateral_shear_ratio: float
    residual: float
    fitted_mask: np.ndarray


def _gaussian_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cx: float, cy: float,
    sigma: float,
    angle_deg: float,
) -> np.ndarray:
    """Compute Gaussian mask for given parameters."""
    angle_rad = np.radians(angle_deg)
    dx = grid_x - cx
    dy = grid_y - cy
    cross_dist = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    return np.exp(-(cross_dist ** 2) / (2 * sigma ** 2))


def fit_spatial_gaussian(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    rms_displacement: np.ndarray,
    initial_guess: dict | None = None,
) -> SpatialFitResult:
    """Fit Gaussian spatial model to the RMS displacement pattern.

    The RMS displacement over time should show the artery as a Gaussian ridge.

    Args:
        grid_x: (R, C) X coordinates in mm.
        grid_y: (R, C) Y coordinates in mm.
        rms_displacement: (R, C, 2) RMS displacement [dx, dy] in mm.
        initial_guess: Optional dict with keys 'cx', 'cy', 'sigma', 'angle'.

    Returns:
        SpatialFitResult with estimated parameters.
    """
    rms_y = rms_displacement[..., 1]  # Y is dominant
    rms_x = rms_displacement[..., 0]

    # Default initial guess: peak of RMS map
    if initial_guess is None:
        peak_idx = np.unravel_index(np.argmax(rms_y), rms_y.shape)
        cx0 = grid_x[peak_idx]
        cy0 = grid_y[peak_idx]
        sigma0 = 3.0
        angle0 = 0.0
    else:
        cx0 = initial_guess.get("cx", 0.0)
        cy0 = initial_guess.get("cy", 0.0)
        sigma0 = initial_guess.get("sigma", 3.0)
        angle0 = initial_guess.get("angle", 0.0)

    def objective(params):
        cx, cy, sigma, angle, amp_y = params
        sigma = max(sigma, 0.5)  # prevent collapse
        mask = _gaussian_mask(grid_x, grid_y, cx, cy, sigma, angle)
        predicted_y = amp_y * mask
        return np.sum((rms_y - predicted_y) ** 2)

    amp_y0 = float(np.max(rms_y))
    result = minimize(
        objective,
        x0=[cx0, cy0, sigma0, angle0, amp_y0],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 0.01, "fatol": 1e-10},
    )

    cx, cy, sigma, angle, amp_y = result.x
    sigma = max(abs(sigma), 0.5)

    # Estimate X amplitude from the derivative-of-Gaussian pattern
    mask = _gaussian_mask(grid_x, grid_y, cx, cy, sigma, angle)
    amp_x = float(np.max(rms_x))
    lateral_ratio = amp_x / (amp_y + 1e-12)

    return SpatialFitResult(
        center_x_mm=float(cx),
        center_y_mm=float(cy),
        sigma_mm=float(sigma),
        angle_deg=float(angle),
        amplitude_y=float(amp_y),
        amplitude_x=float(amp_x),
        lateral_shear_ratio=float(lateral_ratio),
        residual=float(result.fun),
        fitted_mask=mask,
    )
