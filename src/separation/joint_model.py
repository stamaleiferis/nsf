"""Joint model-based separation: fit pulse Gaussian + polynomial artifact simultaneously.

Instead of sequential polynomial fit → subtraction, this jointly optimizes
both the polynomial artifact coefficients and the Gaussian pulse parameters
at each frame, resolving ambiguity where a Gaussian partially projects
onto the polynomial basis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.separation.polynomial_fit import _build_design_matrix


@dataclass
class JointModelConfig:
    """Configuration for joint model separation.

    Attributes:
        poly_degree: Polynomial degree for artifact model.
        regularization: Tikhonov regularization.
    """

    poly_degree: int = 2
    regularization: float = 1e-6


def _build_joint_design_matrix(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    artery_mask_flat: np.ndarray,
    poly_degree: int,
) -> np.ndarray:
    """Build joint design matrix: polynomial terms + artery mask column.

    The model is: displacement = polynomial(x,y) + amplitude * mask(x,y)
    where polynomial captures artifact and mask captures pulse.

    Args:
        x_norm: (N,) normalized X coordinates.
        y_norm: (N,) normalized Y coordinates.
        artery_mask_flat: (N,) artery mask values.
        poly_degree: Max polynomial degree.

    Returns:
        (N, M+1) design matrix where M = number of poly terms, +1 for pulse.
    """
    poly_basis = _build_design_matrix(x_norm, y_norm, poly_degree)  # (N, M)
    # Add artery mask as an extra column for the pulse component
    return np.column_stack([poly_basis, artery_mask_flat])


def joint_separate(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    displacements: np.ndarray,
    artery_mask: np.ndarray,
    config: JointModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly fit polynomial artifact + Gaussian pulse at each frame.

    The model: d(x,y,t) = Σ c_jk(t) x^j y^k + a(t) * mask(x,y)

    This resolves the ambiguity where the Gaussian pulse partially projects
    onto the polynomial basis, which causes the sequential approach to
    either absorb pulse into the artifact estimate or leave artifact residual.

    Args:
        grid_x: (R, C) X coordinates in mm.
        grid_y: (R, C) Y coordinates in mm.
        displacements: (T, R, C, 2) displacement time series.
        artery_mask: (R, C) artery influence mask [0, 1].
        config: Joint model configuration.

    Returns:
        (recovered_pulse, estimated_artifact) each (T, R, C, 2).
    """
    if config is None:
        config = JointModelConfig()

    T, R, C, _ = displacements.shape
    N = R * C

    # Normalize coordinates
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()
    x_range = x_flat.max() - x_flat.min() or 1.0
    y_range = y_flat.max() - y_flat.min() or 1.0
    x_norm = 2.0 * (x_flat - x_flat.mean()) / x_range
    y_norm = 2.0 * (y_flat - y_flat.mean()) / y_range

    mask_flat = artery_mask.ravel()

    # Build joint design matrix
    A = _build_joint_design_matrix(x_norm, y_norm, mask_flat, config.poly_degree)
    M_total = A.shape[1]
    M_poly = M_total - 1  # last column is pulse

    # Solve: (A^T A + λI) c = A^T d for each frame and axis
    reg = config.regularization * np.eye(M_total)
    solve_matrix = np.linalg.solve(A.T @ A + reg, A.T)  # (M_total, N)

    recovered_pulse = np.empty((T, R, C, 2))
    estimated_artifact = np.empty((T, R, C, 2))

    for ax in range(2):
        D = displacements[:, :, :, ax].reshape(T, -1)  # (T, N)
        coeffs = (solve_matrix @ D.T).T  # (T, M_total)

        # Pulse component: last coefficient * mask
        pulse_amplitudes = coeffs[:, -1]  # (T,)
        recovered_pulse[:, :, :, ax] = (
            pulse_amplitudes[:, None, None] * artery_mask[None, :, :]
        )

        # Artifact component: polynomial terms only
        poly_coeffs = coeffs[:, :M_poly]  # (T, M_poly)
        A_poly = A[:, :M_poly]  # (N, M_poly)
        estimated_artifact[:, :, :, ax] = (
            (poly_coeffs @ A_poly.T).reshape(T, R, C)
        )

    return recovered_pulse, estimated_artifact
