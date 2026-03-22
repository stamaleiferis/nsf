"""Evaluation metrics for signal separation quality."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SeparationMetrics:
    """Results from evaluating separation quality.

    Attributes:
        separation_snr_db: SNR improvement in dB (recovered pulse power / residual artifact).
        waveform_correlation: Pearson correlation between recovered and true pulse waveform.
        spatial_correlation: Correlation between recovered and true spatial pulse pattern.
        artifact_residual_fraction: Fraction of artifact power remaining in residual.
    """

    separation_snr_db: float
    waveform_correlation: float
    spatial_correlation: float
    artifact_residual_fraction: float


def separation_snr(
    recovered_pulse: np.ndarray,
    true_pulse: np.ndarray,
    true_artifact: np.ndarray,
) -> float:
    """Compute separation SNR in dB.

    SNR = 10 * log10(pulse_power / residual_artifact_power)
    where residual_artifact = recovered_pulse - true_pulse (the artifact leakage).

    Args:
        recovered_pulse: (T, R, C, 2) recovered pulse displacement.
        true_pulse: (T, R, C, 2) ground truth pulse displacement.
        true_artifact: (T, R, C, 2) ground truth artifact displacement.

    Returns:
        SNR in dB.
    """
    pulse_power = np.mean(true_pulse ** 2)
    residual = recovered_pulse - true_pulse
    residual_power = np.mean(residual ** 2)
    if residual_power == 0:
        return float("inf")
    return 10.0 * np.log10(pulse_power / residual_power)


def waveform_correlation(
    recovered_pulse: np.ndarray,
    true_pulse: np.ndarray,
    artery_mask: np.ndarray,
) -> float:
    """Correlation between recovered and true pulse at artery-center markers.

    Extracts a scalar waveform by averaging Y-displacement over markers
    with high artery mask values, then computes Pearson correlation.

    Args:
        recovered_pulse: (T, R, C, 2) recovered pulse displacement.
        true_pulse: (T, R, C, 2) ground truth pulse displacement.
        artery_mask: (R, C) artery influence mask.

    Returns:
        Pearson correlation coefficient.
    """
    # Weight by artery mask to focus on artery-center markers
    w = artery_mask / (artery_mask.sum() + 1e-12)

    # Extract Y-component (axis=1, dominant due to camera tilt)
    recovered_wf = np.einsum("trc,rc->t", recovered_pulse[:, :, :, 1], w)
    true_wf = np.einsum("trc,rc->t", true_pulse[:, :, :, 1], w)

    # Pearson correlation
    r_mean = recovered_wf - recovered_wf.mean()
    t_mean = true_wf - true_wf.mean()
    num = np.sum(r_mean * t_mean)
    den = np.sqrt(np.sum(r_mean ** 2) * np.sum(t_mean ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


def spatial_correlation(
    recovered_pulse: np.ndarray,
    true_pulse: np.ndarray,
) -> float:
    """Correlation between recovered and true spatial pulse pattern.

    Computes the time-averaged spatial pattern and correlates.

    Args:
        recovered_pulse: (T, R, C, 2) recovered pulse displacement.
        true_pulse: (T, R, C, 2) ground truth pulse displacement.

    Returns:
        Pearson correlation of flattened spatial patterns.
    """
    # RMS spatial pattern over time
    rec_spatial = np.sqrt(np.mean(recovered_pulse ** 2, axis=0))  # (R, C, 2)
    true_spatial = np.sqrt(np.mean(true_pulse ** 2, axis=0))

    r = rec_spatial.ravel()
    t = true_spatial.ravel()

    r_mean = r - r.mean()
    t_mean = t - t.mean()
    num = np.sum(r_mean * t_mean)
    den = np.sqrt(np.sum(r_mean ** 2) * np.sum(t_mean ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


def artifact_residual_fraction(
    estimated_artifact: np.ndarray,
    true_artifact: np.ndarray,
) -> float:
    """Fraction of true artifact power remaining after subtraction.

    Args:
        estimated_artifact: (T, R, C, 2) estimated artifact displacement.
        true_artifact: (T, R, C, 2) ground truth artifact displacement.

    Returns:
        Fraction in [0, 1]. Lower is better.
    """
    residual = true_artifact - estimated_artifact
    artifact_power = np.mean(true_artifact ** 2)
    if artifact_power == 0:
        return 0.0
    return float(np.mean(residual ** 2) / artifact_power)


def evaluate(
    recovered_pulse: np.ndarray,
    estimated_artifact: np.ndarray,
    true_pulse: np.ndarray,
    true_artifact: np.ndarray,
    artery_mask: np.ndarray,
) -> SeparationMetrics:
    """Compute all separation quality metrics.

    Args:
        recovered_pulse: (T, R, C, 2) recovered pulse displacement.
        estimated_artifact: (T, R, C, 2) estimated artifact displacement.
        true_pulse: (T, R, C, 2) ground truth pulse displacement.
        true_artifact: (T, R, C, 2) ground truth artifact displacement.
        artery_mask: (R, C) artery influence mask.

    Returns:
        SeparationMetrics with all quality metrics.
    """
    return SeparationMetrics(
        separation_snr_db=separation_snr(recovered_pulse, true_pulse, true_artifact),
        waveform_correlation=waveform_correlation(recovered_pulse, true_pulse, artery_mask),
        spatial_correlation=spatial_correlation(recovered_pulse, true_pulse),
        artifact_residual_fraction=artifact_residual_fraction(estimated_artifact, true_artifact),
    )
