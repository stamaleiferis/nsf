"""Subspace-based artifact estimation using off-artery reference markers.

Instead of assuming artifacts are polynomial, this approach learns the
artifact spatial pattern from off-artery markers (which are pure artifact)
and uses this to estimate and remove artifacts at artery markers.

The key insight: off-artery markers see only artifact + noise. We can
build a low-rank model of the artifact spatial structure from these markers,
then predict what the artifact looks like at artery markers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SubspaceConfig:
    """Configuration for subspace-based separation.

    Attributes:
        n_components: Number of artifact subspace components.
        mask_threshold: Artery mask threshold for reference marker selection.
            Markers with mask < threshold are considered artifact-only references.
    """

    n_components: int = 15
    mask_threshold: float = 0.3


def subspace_separate(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    displacements: np.ndarray,
    artery_mask: np.ndarray,
    config: SubspaceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate pulse from artifact using learned artifact subspace.

    Steps:
    1. Select off-artery reference markers (artery_mask < threshold).
    2. Build low-rank SVD model of reference marker displacements.
    3. For each artery marker, estimate artifact by projecting its
       displacement onto the subspace spanned by reference markers.
    4. Subtract artifact estimate to recover pulse.

    The method works because artifacts are spatially smooth and correlated
    across all markers, while pulse is localized to artery markers.
    The reference markers provide a clean view of the artifact, and spatial
    interpolation (via the SVD basis) predicts the artifact at artery locations.

    Args:
        grid_x: (R, C) X coordinates in mm.
        grid_y: (R, C) Y coordinates in mm.
        displacements: (T, R, C, 2) displacement time series.
        artery_mask: (R, C) artery influence mask [0, 1].
        config: Subspace configuration.

    Returns:
        (recovered_pulse, estimated_artifact) each (T, R, C, 2).
    """
    if config is None:
        config = SubspaceConfig()

    T, R, C, _ = displacements.shape
    N = R * C

    # Select reference markers (off-artery)
    ref_mask = artery_mask.ravel() < config.mask_threshold
    n_ref = np.sum(ref_mask)

    if n_ref < config.n_components + 1:
        # Not enough reference markers, fall back to using all low-mask markers
        threshold = np.percentile(artery_mask, 50)
        ref_mask = artery_mask.ravel() <= threshold
        n_ref = np.sum(ref_mask)

    K = min(config.n_components, n_ref - 1)

    recovered_pulse = np.zeros_like(displacements)
    estimated_artifact = np.zeros_like(displacements)

    # Estimate pulse temporal pattern from artery-weighted markers
    # (used to protect pulse from being absorbed by artifact subspace)
    mask_flat = artery_mask.ravel()
    mask_weights = mask_flat / (mask_flat.sum() + 1e-12)

    for ax in range(2):
        D = displacements[:, :, :, ax].reshape(T, N)  # (T, N)

        # Reference marker displacements
        D_ref = D[:, ref_mask]  # (T, n_ref)

        # Estimate pulse temporal waveform from artery markers
        pulse_temporal = D @ mask_weights  # (T,) — weighted average at artery
        pulse_temporal = pulse_temporal - pulse_temporal.mean()
        pulse_norm = np.linalg.norm(pulse_temporal)
        if pulse_norm > 1e-10:
            pulse_unit = pulse_temporal / pulse_norm
        else:
            pulse_unit = np.zeros(T)

        # SVD of reference markers → artifact temporal basis
        D_ref_centered = D_ref - D_ref.mean(axis=0)
        U, S, Vt = np.linalg.svd(D_ref_centered, full_matrices=False)
        U_k = U[:, :K]  # (T, K) — temporal basis of artifact

        # Orthogonalize artifact basis against pulse temporal pattern
        # Remove the pulse component from each artifact basis vector
        for j in range(K):
            overlap = np.dot(U_k[:, j], pulse_unit)
            U_k[:, j] -= overlap * pulse_unit
        # Re-orthogonalize
        Q, R_qr = np.linalg.qr(U_k)
        # Keep only columns with significant norm (some may collapse)
        norms = np.abs(np.diag(R_qr))
        valid = norms > 1e-8
        U_k = Q[:, valid]

        # Project onto protected artifact subspace
        D_centered = D - D.mean(axis=0)
        coeffs = U_k.T @ D_centered  # (K', N)
        D_artifact = U_k @ coeffs  # (T, N) — artifact estimate
        D_artifact += D.mean(axis=0)

        estimated_artifact[:, :, :, ax] = D_artifact.reshape(T, R, C)
        recovered_pulse[:, :, :, ax] = (D - D_artifact).reshape(T, R, C)

    return recovered_pulse, estimated_artifact
