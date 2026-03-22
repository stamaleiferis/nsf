"""Pulse model: Gaussian spatial profile x parametric temporal waveform."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PulseConfig:
    """Configuration for the arterial pulse model.

    Attributes:
        heart_rate_bpm: Heart rate in beats per minute.
        amplitude_mm: Peak pulse displacement amplitude in mm.
        artery_center_x_mm: Artery center X coordinate in mm (cross-artery).
        artery_center_y_mm: Artery center Y coordinate in mm (along-artery).
        artery_angle_deg: Artery angle from Y axis in degrees.
        sigma_mm: Gaussian width of pulse spatial profile in mm.
        lateral_shear_ratio: Ratio of X-displacement to Z-displacement.
        camera_tilt_deg: Camera tilt from surface normal in degrees.
    """

    heart_rate_bpm: float = 72.0
    amplitude_mm: float = 0.15
    artery_center_x_mm: float = 0.0
    artery_center_y_mm: float = 0.0
    artery_angle_deg: float = 0.0
    sigma_mm: float = 3.0
    lateral_shear_ratio: float = 0.3
    camera_tilt_deg: float = 40.0


def pulse_waveform(t: np.ndarray, config: PulseConfig) -> np.ndarray:
    """Generate a physiological pulse waveform A(t) in [0, 1].

    Uses a simplified model with systolic peak and dicrotic notch,
    built from a sum of Gaussian bumps.
    """
    freq = config.heart_rate_bpm / 60.0
    phase = (t * freq) % 1.0  # normalized phase [0, 1) within each beat

    # Systolic peak (sharp rise, fast decay)
    systolic = np.exp(-((phase - 0.15) ** 2) / (2 * 0.01))
    # Dicrotic notch and diastolic wave
    dicrotic = 0.35 * np.exp(-((phase - 0.45) ** 2) / (2 * 0.008))
    # Diastolic runoff
    diastolic = 0.15 * np.exp(-((phase - 0.60) ** 2) / (2 * 0.02))

    waveform = systolic + dicrotic + diastolic
    # Normalize to [0, 1]
    wmax = waveform.max()
    if wmax > 0:
        waveform /= wmax
    return waveform


def artery_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    config: PulseConfig,
) -> np.ndarray:
    """Compute soft artery influence mask (R, C) in [0, 1].

    The mask is a Gaussian centered on the artery, measuring the
    cross-artery distance (perpendicular to the artery direction).
    """
    angle_rad = np.radians(config.artery_angle_deg)
    dx = grid_x - config.artery_center_x_mm
    dy = grid_y - config.artery_center_y_mm

    # Cross-artery distance (perpendicular to artery direction)
    cross_dist = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)

    mask = np.exp(-(cross_dist ** 2) / (2 * config.sigma_mm ** 2))
    return mask


def pulse_displacement_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    waveform_t: np.ndarray,
    config: PulseConfig,
) -> np.ndarray:
    """Compute pulse displacement field (T, R, C, 2).

    Args:
        grid_x: (R, C) X coordinates of markers in mm.
        grid_y: (R, C) Y coordinates of markers in mm.
        waveform_t: (T,) scalar pulse waveform.
        config: Pulse configuration.

    Returns:
        (T, R, C, 2) displacement array [dx, dy] in mm.
    """
    angle_rad = np.radians(config.artery_angle_deg)
    dx = grid_x - config.artery_center_x_mm
    dy = grid_y - config.artery_center_y_mm

    # Cross-artery distance
    cross_dist = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)

    # Gaussian spatial profile (Z-displacement projected)
    gauss = np.exp(-(cross_dist ** 2) / (2 * config.sigma_mm ** 2))

    # Y-displacement (dominant): Z-bulge projected through camera tilt
    tilt_factor = np.sin(np.radians(config.camera_tilt_deg))
    dy_spatial = gauss * config.amplitude_mm * tilt_factor  # (R, C)

    # X-displacement: lateral stretch from bulge gradient (derivative of Gaussian)
    dx_spatial = (
        config.lateral_shear_ratio
        * config.amplitude_mm
        * (-cross_dist / config.sigma_mm ** 2)
        * np.exp(-(cross_dist ** 2) / (2 * config.sigma_mm ** 2))
    )

    # Rotate displacement components back to grid frame
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    # Cross-artery direction is (cos_a, -sin_a); along-artery is (sin_a, cos_a)
    # dx_spatial is in cross-artery direction, dy_spatial is in Z->Y projection
    disp_x = dx_spatial * cos_a  # (R, C)
    disp_y = dy_spatial + dx_spatial * (-sin_a)  # (R, C)

    # Scale by temporal waveform: (T, 1, 1) * (R, C)
    T = waveform_t.shape[0]
    result = np.empty((T, *grid_x.shape, 2))
    result[..., 0] = waveform_t[:, None, None] * disp_x[None, :, :]
    result[..., 1] = waveform_t[:, None, None] * disp_y[None, :, :]

    return result
