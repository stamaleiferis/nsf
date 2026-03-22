"""Estimate artifact statistics from off-artery markers.

Markers far from the artery contain only artifact + noise signals.
This module characterizes those statistics for use in synthetic generation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ArtifactStats:
    """Statistical characterization of motion artifacts.

    Attributes:
        rms_amplitude_mm: RMS artifact displacement in mm.
        max_amplitude_mm: Maximum artifact displacement in mm.
        spectral_centroid_hz: Power-weighted mean frequency.
        spectral_bandwidth_hz: Standard deviation of power spectrum.
        x_y_ratio: Ratio of X to Y artifact displacement RMS.
        spatial_correlation: Mean correlation between adjacent markers.
        temporal_autocorrelation: Autocorrelation at 1-frame lag.
    """

    rms_amplitude_mm: float
    max_amplitude_mm: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    x_y_ratio: float
    spatial_correlation: float
    temporal_autocorrelation: float


def estimate_artifact_stats(
    displacements: np.ndarray,
    fps: float,
    artery_mask: np.ndarray,
    mask_threshold: float = 0.1,
) -> ArtifactStats:
    """Estimate artifact statistics from off-artery markers.

    Args:
        displacements: (T, R, C, 2) displacement time series.
        fps: Frame rate.
        artery_mask: (R, C) artery influence mask [0, 1].
        mask_threshold: Markers with mask < threshold are considered artifact-only.

    Returns:
        ArtifactStats with estimated statistics.
    """
    T, R, C, _ = displacements.shape

    # Select artifact-only markers
    off_artery = artery_mask < mask_threshold
    if not np.any(off_artery):
        # Fall back to lowest-mask markers
        threshold = np.percentile(artery_mask, 25)
        off_artery = artery_mask <= threshold

    # Extract displacements at off-artery markers
    # Shape: (T, N_off, 2) where N_off is number of off-artery markers
    off_disp = displacements[:, off_artery, :]

    # RMS and max amplitude
    rms_per_marker = np.sqrt(np.mean(off_disp ** 2, axis=0))  # (N_off, 2)
    rms_amplitude = float(np.mean(rms_per_marker))
    max_amplitude = float(np.max(np.abs(off_disp)))

    # X/Y ratio
    rms_x = np.sqrt(np.mean(off_disp[..., 0] ** 2))
    rms_y = np.sqrt(np.mean(off_disp[..., 1] ** 2))
    x_y_ratio = float(rms_x / (rms_y + 1e-12))

    # Spectral analysis (average over off-artery markers, Y component)
    y_signals = off_disp[:, :, 1]  # (T, N_off)
    mean_spec = np.mean(
        np.abs(np.fft.rfft(y_signals, axis=0)) ** 2, axis=1
    )
    freqs = np.fft.rfftfreq(T, 1.0 / fps)

    # Spectral centroid and bandwidth
    total_power = np.sum(mean_spec[1:]) + 1e-20
    spectral_centroid = float(np.sum(freqs[1:] * mean_spec[1:]) / total_power)
    spectral_bandwidth = float(
        np.sqrt(np.sum((freqs[1:] - spectral_centroid) ** 2 * mean_spec[1:]) / total_power)
    )

    # Spatial correlation: mean correlation between adjacent markers
    rows, cols = np.where(off_artery)
    corrs = []
    for i in range(len(rows)):
        for dr, dc in [(0, 1), (1, 0)]:
            nr, nc = rows[i] + dr, cols[i] + dc
            if 0 <= nr < R and 0 <= nc < C and off_artery[nr, nc]:
                c = np.corrcoef(
                    displacements[:, rows[i], cols[i], 1],
                    displacements[:, nr, nc, 1]
                )[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
    spatial_corr = float(np.mean(corrs)) if corrs else 0.0

    # Temporal autocorrelation at lag 1
    y_all = off_disp[:, :, 1]
    autocorrs = []
    for j in range(y_all.shape[1]):
        sig = y_all[:, j]
        sig_centered = sig - sig.mean()
        var = np.sum(sig_centered ** 2)
        if var > 0:
            ac = np.sum(sig_centered[:-1] * sig_centered[1:]) / var
            autocorrs.append(ac)
    temporal_ac = float(np.mean(autocorrs)) if autocorrs else 0.0

    return ArtifactStats(
        rms_amplitude_mm=rms_amplitude,
        max_amplitude_mm=max_amplitude,
        spectral_centroid_hz=spectral_centroid,
        spectral_bandwidth_hz=spectral_bandwidth,
        x_y_ratio=x_y_ratio,
        spatial_correlation=spatial_corr,
        temporal_autocorrelation=temporal_ac,
    )
