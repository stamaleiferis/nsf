"""Weighted 2D polynomial fitting for artifact estimation.

Fits a low-order 2D polynomial to marker displacements at each frame,
using artery-map-derived weights so that markers near the artery
(contaminated by pulse) contribute less to the artifact fit.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt


@dataclass
class PolyFitConfig:
    """Configuration for polynomial artifact fitting.

    Attributes:
        degree: Maximum polynomial degree (1=affine, 2=quadratic, 3=cubic).
        regularization: Tikhonov regularization parameter (0 = none).
    """

    degree: int = 2
    regularization: float = 1e-6


def _build_design_matrix(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Build the polynomial design matrix.

    Args:
        x: (N,) normalized X coordinates.
        y: (N,) normalized Y coordinates.
        degree: Max polynomial degree.

    Returns:
        (N, M) design matrix where M = number of polynomial terms.
    """
    terms = []
    for j in range(degree + 1):
        for k in range(degree + 1):
            if j + k <= degree:
                terms.append((x ** j) * (y ** k))
    return np.column_stack(terms)


def fit_polynomial(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    displacements: np.ndarray,
    weights: np.ndarray,
    config: PolyFitConfig,
) -> np.ndarray:
    """Fit a 2D polynomial to a single frame's displacements.

    Args:
        grid_x: (R, C) X coordinates of markers.
        grid_y: (R, C) Y coordinates of markers.
        displacements: (R, C, 2) displacement at this frame.
        weights: (R, C) fitting weights (high = trusted artifact-only marker).
        config: Polynomial fit configuration.

    Returns:
        (R, C, 2) polynomial-predicted artifact displacement.
    """
    R, C = grid_x.shape
    N = R * C

    # Normalize coordinates to [-1, 1]
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()
    x_range = x_flat.max() - x_flat.min()
    y_range = y_flat.max() - y_flat.min()
    x_norm = 2.0 * (x_flat - x_flat.mean()) / (x_range if x_range > 0 else 1.0)
    y_norm = 2.0 * (y_flat - y_flat.mean()) / (y_range if y_range > 0 else 1.0)

    A = _build_design_matrix(x_norm, y_norm, config.degree)  # (N, M)
    M = A.shape[1]
    w = weights.ravel()  # (N,)

    # Weighted least squares: minimize ||W(A c - d)||^2 + λ||c||^2
    W = np.diag(w)
    WA = W @ A
    reg = config.regularization * np.eye(M)

    result = np.empty((R, C, 2))
    for ax in range(2):
        d = displacements[:, :, ax].ravel()
        Wd = W @ d
        # Normal equations: (A^T W^T W A + λI) c = A^T W^T W d
        c = np.linalg.solve(WA.T @ WA + reg, WA.T @ Wd)
        result[:, :, ax] = (A @ c).reshape(R, C)

    return result


def fit_polynomial_all_frames(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    displacements: np.ndarray,
    weights: np.ndarray,
    config: PolyFitConfig,
) -> np.ndarray:
    """Fit a 2D polynomial to each frame's displacements.

    Args:
        grid_x: (R, C) X coordinates.
        grid_y: (R, C) Y coordinates.
        displacements: (T, R, C, 2) displacement time series.
        weights: (R, C) fitting weights.
        config: Polynomial fit configuration.

    Returns:
        (T, R, C, 2) polynomial-predicted artifact displacement for all frames.
    """
    T = displacements.shape[0]
    R, C = grid_x.shape

    # Pre-compute design matrix (same for all frames)
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()
    x_range = x_flat.max() - x_flat.min()
    y_range = y_flat.max() - y_flat.min()
    x_norm = 2.0 * (x_flat - x_flat.mean()) / (x_range if x_range > 0 else 1.0)
    y_norm = 2.0 * (y_flat - y_flat.mean()) / (y_range if y_range > 0 else 1.0)

    A = _build_design_matrix(x_norm, y_norm, config.degree)  # (N, M)
    M = A.shape[1]
    w = weights.ravel()  # (N,)

    # Pre-compute (A^T W^2 A + λI)^{-1} A^T W^2
    W = np.diag(w)
    WA = W @ A
    reg = config.regularization * np.eye(M)
    solve_matrix = np.linalg.solve(WA.T @ WA + reg, WA.T @ W)  # (M, N)

    result = np.empty((T, R, C, 2))
    for ax in range(2):
        D = displacements[:, :, :, ax].reshape(T, -1)  # (T, N)
        C_coeffs = (solve_matrix @ D.T).T  # (T, M)
        result[:, :, :, ax] = (C_coeffs @ A.T).reshape(T, R, C)

    return result


def fit_polynomial_smooth_coeffs(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    displacements: np.ndarray,
    weights: np.ndarray,
    config: PolyFitConfig,
    fps: float,
    smooth_cutoff_hz: float = 0.5,
) -> np.ndarray:
    """Fit polynomial per frame, lowpass the coefficients, then reconstruct.

    This approach captures the full-bandwidth spatial structure of artifacts
    while removing pulsatile contamination from the polynomial coefficients.

    Args:
        grid_x: (R, C) X coordinates.
        grid_y: (R, C) Y coordinates.
        displacements: (T, R, C, 2) displacement time series.
        weights: (R, C) fitting weights.
        config: Polynomial fit configuration.
        fps: Sampling rate in Hz.
        smooth_cutoff_hz: Cutoff for lowpassing the polynomial coefficients.

    Returns:
        (T, R, C, 2) polynomial-predicted artifact displacement.
    """
    T = displacements.shape[0]
    R, C = grid_x.shape

    # Build design matrix
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()
    x_range = x_flat.max() - x_flat.min()
    y_range = y_flat.max() - y_flat.min()
    x_norm = 2.0 * (x_flat - x_flat.mean()) / (x_range if x_range > 0 else 1.0)
    y_norm = 2.0 * (y_flat - y_flat.mean()) / (y_range if y_range > 0 else 1.0)

    A = _build_design_matrix(x_norm, y_norm, config.degree)  # (N, M)
    M = A.shape[1]
    w = weights.ravel()

    W = np.diag(w)
    WA = W @ A
    reg = config.regularization * np.eye(M)
    solve_matrix = np.linalg.solve(WA.T @ WA + reg, WA.T @ W)  # (M, N)

    # Step 1: Get polynomial coefficients at each frame
    coeffs = np.empty((T, M, 2))
    for ax in range(2):
        D = displacements[:, :, :, ax].reshape(T, -1)  # (T, N)
        coeffs[:, :, ax] = (solve_matrix @ D.T).T  # (T, M)

    # Step 2: Lowpass the coefficients to remove pulse contamination
    nyquist = fps / 2.0
    if smooth_cutoff_hz < nyquist:
        sos = butter(4, smooth_cutoff_hz / nyquist, btype="low", output="sos")
        for m in range(M):
            for ax in range(2):
                coeffs[:, m, ax] = sosfiltfilt(sos, coeffs[:, m, ax])

    # Step 3: Reconstruct artifact from smoothed coefficients
    result = np.empty((T, R, C, 2))
    for ax in range(2):
        result[:, :, :, ax] = (coeffs[:, :, ax] @ A.T).reshape(T, R, C)

    return result
