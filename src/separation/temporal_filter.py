"""Temporal filtering for marker displacement time series.

Provides bandpass and lowpass filtering to separate pulsatile components
from slow artifact/baseline components in the temporal domain.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt


@dataclass
class FilterConfig:
    """Configuration for temporal filtering.

    Attributes:
        lowpass_cutoff_hz: Cutoff frequency for lowpass filter (artifact extraction).
            Pulse fundamental is typically 0.67-2.33 Hz; set below pulse range.
        bandpass_low_hz: Lower cutoff for bandpass (pulse extraction).
        bandpass_high_hz: Upper cutoff for bandpass (pulse extraction).
        filter_order: Butterworth filter order.
    """

    lowpass_cutoff_hz: float = 0.5
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 15.0
    filter_order: int = 4


def lowpass_positions(
    positions: np.ndarray,
    fps: float,
    config: FilterConfig,
) -> np.ndarray:
    """Lowpass filter marker positions to extract artifact-only component.

    Removes pulsatile content (heart rate + harmonics), leaving only
    slow baseline changes and artifact deformations.

    Args:
        positions: (T, R, C, 2) marker positions.
        fps: Sampling rate in Hz.
        config: Filter configuration.

    Returns:
        (T, R, C, 2) lowpass-filtered positions.
    """
    nyquist = fps / 2.0
    if config.lowpass_cutoff_hz >= nyquist:
        return positions.copy()

    sos = butter(
        config.filter_order,
        config.lowpass_cutoff_hz / nyquist,
        btype="low",
        output="sos",
    )

    T, R, C, _ = positions.shape
    result = np.empty_like(positions)

    # Filter each marker's time series independently
    for r in range(R):
        for c in range(C):
            for ax in range(2):
                result[:, r, c, ax] = sosfiltfilt(sos, positions[:, r, c, ax])

    return result


def bandpass_positions(
    positions: np.ndarray,
    fps: float,
    config: FilterConfig,
) -> np.ndarray:
    """Bandpass filter to isolate pulse-band content.

    Args:
        positions: (T, R, C, 2) marker positions.
        fps: Sampling rate in Hz.
        config: Filter configuration.

    Returns:
        (T, R, C, 2) bandpass-filtered positions.
    """
    nyquist = fps / 2.0
    low = config.bandpass_low_hz / nyquist
    high = min(config.bandpass_high_hz / nyquist, 0.999)

    sos = butter(
        config.filter_order,
        [low, high],
        btype="band",
        output="sos",
    )

    T, R, C, _ = positions.shape
    result = np.empty_like(positions)

    for r in range(R):
        for c in range(C):
            for ax in range(2):
                result[:, r, c, ax] = sosfiltfilt(sos, positions[:, r, c, ax])

    return result
