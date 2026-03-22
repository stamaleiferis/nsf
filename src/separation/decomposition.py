"""ICA and PCA-based separation approaches for comparison.

Provides baseline separation using spatial ICA and PCA on
marker displacement time series.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DecompositionConfig:
    """Configuration for decomposition-based separation.

    Attributes:
        n_components: Number of components to extract.
        method: 'pca' or 'ica'.
        max_iter: Maximum iterations for ICA.
    """

    n_components: int = 10
    method: str = "pca"
    max_iter: int = 200


def _pca_decompose(
    data: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA decomposition of (T, N) data.

    Returns:
        scores: (T, K) temporal scores.
        components: (K, N) spatial components.
        explained_variance: (K,) variance per component.
    """
    # Center
    mean = data.mean(axis=0)
    centered = data - mean

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    K = min(n_components, len(S))

    scores = U[:, :K] * S[:K]
    components = Vt[:K]
    explained_variance = S[:K] ** 2 / (data.shape[0] - 1)

    return scores, components, explained_variance


def _ica_decompose(
    data: np.ndarray, n_components: int, max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Simple FastICA decomposition of (T, N) data.

    Uses a basic fixed-point ICA implementation with tanh nonlinearity.

    Returns:
        sources: (T, K) independent source signals.
        mixing: (K, N) mixing matrix (spatial patterns).
    """
    # First reduce dimensionality with PCA
    scores, components, _ = _pca_decompose(data, n_components)
    K = scores.shape[1]

    # Whiten
    std = scores.std(axis=0) + 1e-10
    whitened = scores / std

    # FastICA with tanh nonlinearity
    rng = np.random.default_rng(42)
    W = rng.standard_normal((K, K))
    W, _ = np.linalg.qr(W)

    for _ in range(max_iter):
        # g(x) = tanh(x), g'(x) = 1 - tanh(x)^2
        WX = W @ whitened.T  # (K, T)
        gWX = np.tanh(WX)
        g_prime = 1 - gWX ** 2

        W_new = (gWX @ whitened) / whitened.shape[0] - g_prime.mean(axis=1, keepdims=True) * W
        # Symmetric decorrelation
        U_w, S_w, Vt_w = np.linalg.svd(W_new)
        W_new = U_w @ Vt_w

        # Check convergence
        if np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1)) < 1e-6:
            W = W_new
            break
        W = W_new

    sources = (W @ whitened.T).T  # (T, K)
    # Recover mixing matrix: data ≈ sources @ mixing
    mixing = np.linalg.lstsq(sources, data - data.mean(axis=0), rcond=None)[0]  # (K, N)

    return sources, mixing


def decomposition_separate(
    displacements: np.ndarray,
    artery_mask: np.ndarray,
    config: DecompositionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate pulse from artifact using PCA or ICA.

    Identifies the component whose spatial pattern best matches the artery mask,
    assigns it as pulse, and treats the rest as artifact.

    Args:
        displacements: (T, R, C, 2) displacement time series.
        artery_mask: (R, C) artery influence mask.
        config: Decomposition configuration.

    Returns:
        (recovered_pulse, estimated_artifact) each (T, R, C, 2).
    """
    if config is None:
        config = DecompositionConfig()

    T, R, C, _ = displacements.shape
    N = R * C
    mask_flat = artery_mask.ravel()
    mask_norm = mask_flat / (np.linalg.norm(mask_flat) + 1e-12)

    recovered_pulse = np.zeros_like(displacements)
    estimated_artifact = np.zeros_like(displacements)

    for ax in range(2):
        data = displacements[:, :, :, ax].reshape(T, N)

        if config.method == "ica":
            sources, mixing = _ica_decompose(data, config.n_components, config.max_iter)
        else:
            sources, mixing, _ = _pca_decompose(data, config.n_components)

        K = sources.shape[1]

        # Find component most correlated with artery mask
        correlations = np.array([
            abs(np.corrcoef(mixing[k], mask_flat)[0, 1])
            for k in range(K)
        ])
        pulse_idx = np.argmax(correlations)

        # Pulse: the selected component
        pulse_component = np.outer(sources[:, pulse_idx], mixing[pulse_idx])
        recovered_pulse[:, :, :, ax] = pulse_component.reshape(T, R, C)

        # Artifact: everything else
        artifact = data - data.mean(axis=0) - pulse_component
        estimated_artifact[:, :, :, ax] = artifact.reshape(T, R, C)

    return recovered_pulse, estimated_artifact
