"""Artifact model: time-varying 2D polynomial deformation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from itertools import product


@dataclass
class ArtifactConfig:
    """Configuration for polynomial artifact model.

    Attributes:
        degree: Maximum polynomial degree (1=affine, 2=quadratic, 3=cubic).
        amplitude_mm: RMS amplitude of artifact displacements in mm.
        max_freq_hz: Maximum temporal frequency for coefficient variation.
        seed: Random seed for reproducible generation.
    """

    degree: int = 2
    amplitude_mm: float = 1.0
    max_freq_hz: float = 2.0
    seed: int | None = None


def _poly_terms(degree: int) -> list[tuple[int, int]]:
    """Return list of (j, k) exponent pairs for total degree <= degree."""
    return [(j, k) for j in range(degree + 1) for k in range(degree + 1) if j + k <= degree]


def _poly_basis(
    grid_x: np.ndarray, grid_y: np.ndarray, degree: int
) -> np.ndarray:
    """Evaluate polynomial basis at grid points.

    Args:
        grid_x: (R, C) normalized X coordinates.
        grid_y: (R, C) normalized Y coordinates.
        degree: Max polynomial degree.

    Returns:
        (R, C, N_terms) polynomial basis matrix.
    """
    terms = _poly_terms(degree)
    R, C = grid_x.shape
    basis = np.empty((R, C, len(terms)))
    for i, (j, k) in enumerate(terms):
        basis[:, :, i] = (grid_x ** j) * (grid_y ** k)
    return basis


def generate_artifact_coefficients(
    num_frames: int,
    fps: float,
    config: ArtifactConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate time-varying polynomial coefficients.

    Returns:
        coeffs_x: (T, N_terms) coefficients for X displacement.
        coeffs_y: (T, N_terms) coefficients for Y displacement.
    """
    rng = np.random.default_rng(config.seed)
    terms = _poly_terms(config.degree)
    n_terms = len(terms)
    T = num_frames
    t = np.arange(T) / fps

    # Generate coefficients as sum of random sinusoids
    n_components = 5  # number of frequency components per coefficient
    coeffs_x = np.zeros((T, n_terms))
    coeffs_y = np.zeros((T, n_terms))

    for i, (j, k) in enumerate(terms):
        # Higher-order terms get smaller amplitude
        order = j + k
        order_scale = 1.0 / (1.0 + order) ** 2

        for _ in range(n_components):
            freq = rng.uniform(0.05, config.max_freq_hz)
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            amp = rng.exponential(config.amplitude_mm * order_scale / n_components)

            coeffs_x[:, i] += amp * np.sin(2 * np.pi * freq * t + phase_x)
            coeffs_y[:, i] += amp * np.sin(2 * np.pi * freq * t + phase_y)

    return coeffs_x, coeffs_y


def artifact_displacement_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    num_frames: int,
    fps: float,
    config: ArtifactConfig,
) -> np.ndarray:
    """Compute artifact displacement field (T, R, C, 2).

    Args:
        grid_x: (R, C) X coordinates of markers in mm.
        grid_y: (R, C) Y coordinates of markers in mm.
        num_frames: Number of frames.
        fps: Frame rate.
        config: Artifact configuration.

    Returns:
        (T, R, C, 2) displacement array [dx, dy] in mm.
    """
    # Normalize coordinates to [-1, 1] for numerical stability
    x_range = grid_x.max() - grid_x.min() or 1.0
    y_range = grid_y.max() - grid_y.min() or 1.0
    x_norm = 2.0 * (grid_x - grid_x.mean()) / x_range
    y_norm = 2.0 * (grid_y - grid_y.mean()) / y_range

    basis = _poly_basis(x_norm, y_norm, config.degree)  # (R, C, N)
    coeffs_x, coeffs_y = generate_artifact_coefficients(num_frames, fps, config)

    R, C = grid_x.shape
    result = np.empty((num_frames, R, C, 2))
    # (T, N) @ (R, C, N).T  -> broadcast via einsum
    result[..., 0] = np.einsum("tn,rcn->trc", coeffs_x, basis)
    result[..., 1] = np.einsum("tn,rcn->trc", coeffs_y, basis)

    return result
