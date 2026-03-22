"""Extract and characterize pulse waveforms from marker data.

Identifies the highest-SNR markers (near the artery) and extracts
clean pulse waveforms for parameter estimation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt, find_peaks


@dataclass
class PulseExtractionResult:
    """Result of pulse waveform extraction.

    Attributes:
        waveform: (T,) extracted pulse waveform (bandpass-filtered, artery-weighted).
        heart_rate_bpm: Estimated heart rate in BPM.
        heart_rate_confidence: Confidence of HR estimate [0, 1].
        snr_map: (R, C) per-marker pulse SNR estimate.
        best_markers: List of (row, col) tuples for highest-SNR markers.
    """

    waveform: np.ndarray
    heart_rate_bpm: float
    heart_rate_confidence: float
    snr_map: np.ndarray
    best_markers: list[tuple[int, int]]


def estimate_pulse_snr_map(
    displacements: np.ndarray,
    fps: float,
    hr_range_bpm: tuple[float, float] = (40.0, 140.0),
) -> np.ndarray:
    """Estimate per-marker pulse SNR from displacement power spectrum.

    Pulse SNR = power in HR band / power outside HR band.

    Args:
        displacements: (T, R, C, 2) displacement time series.
        fps: Frame rate.
        hr_range_bpm: Heart rate search range in BPM.

    Returns:
        (R, C) SNR estimate per marker.
    """
    T, R, C, _ = displacements.shape

    # Use Y-component (dominant due to camera tilt)
    y_disp = displacements[:, :, :, 1]

    # Frequency resolution
    freqs = np.fft.rfftfreq(T, 1.0 / fps)
    hr_low = hr_range_bpm[0] / 60.0
    hr_high = hr_range_bpm[1] / 60.0

    # Compute power spectrum per marker
    snr_map = np.empty((R, C))
    for r in range(R):
        for c in range(C):
            spec = np.abs(np.fft.rfft(y_disp[:, r, c])) ** 2
            # Pulse band: HR fundamental + first 3 harmonics
            pulse_mask = np.zeros_like(freqs, dtype=bool)
            for harmonic in range(1, 4):
                pulse_mask |= (freqs >= hr_low * harmonic) & (freqs <= hr_high * harmonic)
            pulse_power = np.sum(spec[pulse_mask])
            noise_power = np.sum(spec[~pulse_mask]) + 1e-20
            snr_map[r, c] = pulse_power / noise_power

    return snr_map


def estimate_heart_rate(
    signal: np.ndarray,
    fps: float,
    hr_range_bpm: tuple[float, float] = (40.0, 140.0),
) -> tuple[float, float]:
    """Estimate heart rate from a 1D signal using FFT peak detection.

    Args:
        signal: (T,) time series.
        fps: Frame rate.
        hr_range_bpm: Search range in BPM.

    Returns:
        (heart_rate_bpm, confidence) where confidence is peak prominence ratio.
    """
    T = len(signal)
    freqs = np.fft.rfftfreq(T, 1.0 / fps)
    spec = np.abs(np.fft.rfft(signal)) ** 2

    # Search in HR range
    hr_low = hr_range_bpm[0] / 60.0
    hr_high = hr_range_bpm[1] / 60.0
    mask = (freqs >= hr_low) & (freqs <= hr_high)

    if not np.any(mask):
        return 0.0, 0.0

    hr_spec = spec.copy()
    hr_spec[~mask] = 0.0

    peak_idx = np.argmax(hr_spec)
    peak_freq = freqs[peak_idx]
    peak_power = spec[peak_idx]
    total_power = np.sum(spec[1:]) + 1e-20  # exclude DC

    hr_bpm = peak_freq * 60.0
    confidence = float(peak_power / total_power)

    return hr_bpm, confidence


def extract_pulse(
    displacements: np.ndarray,
    fps: float,
    artery_mask: np.ndarray | None = None,
    bandpass: tuple[float, float] = (0.5, 15.0),
    n_best: int = 10,
) -> PulseExtractionResult:
    """Extract the pulse waveform from marker displacements.

    Steps:
    1. Estimate per-marker pulse SNR.
    2. Select highest-SNR markers (or use artery mask).
    3. Bandpass filter and average to get clean waveform.
    4. Estimate heart rate.

    Args:
        displacements: (T, R, C, 2) displacement time series.
        fps: Frame rate.
        artery_mask: (R, C) optional artery mask for weighting.
        bandpass: (low, high) Hz for bandpass filter.
        n_best: Number of best markers to average.

    Returns:
        PulseExtractionResult.
    """
    T, R, C, _ = displacements.shape

    # Step 1: SNR map
    snr_map = estimate_pulse_snr_map(displacements, fps)

    # Step 2: Select best markers
    if artery_mask is not None:
        # Weight SNR by artery mask
        weighted_snr = snr_map * artery_mask
    else:
        weighted_snr = snr_map

    # Find top-N markers
    flat_idx = np.argsort(weighted_snr.ravel())[::-1][:n_best]
    best_markers = [divmod(int(i), C) for i in flat_idx]

    # Step 3: Bandpass filter Y-displacement at best markers and average
    nyquist = fps / 2.0
    low = bandpass[0] / nyquist
    high = min(bandpass[1] / nyquist, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")

    waveforms = []
    for r, c in best_markers:
        filtered = sosfiltfilt(sos, displacements[:, r, c, 1])
        waveforms.append(filtered)

    # Weighted average (weight by SNR)
    weights = np.array([weighted_snr[r, c] for r, c in best_markers])
    weights = weights / (weights.sum() + 1e-12)
    waveform = np.sum(
        [w * wf for w, wf in zip(weights, waveforms)], axis=0
    )

    # Step 4: Heart rate estimation
    hr_bpm, hr_conf = estimate_heart_rate(waveform, fps)

    return PulseExtractionResult(
        waveform=waveform,
        heart_rate_bpm=hr_bpm,
        heart_rate_confidence=hr_conf,
        snr_map=snr_map,
        best_markers=best_markers,
    )
