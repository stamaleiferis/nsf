"""Noise model: additive Gaussian measurement noise."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """Configuration for measurement noise model.

    Attributes:
        sigma_mm: Standard deviation of Gaussian noise in mm.
        seed: Random seed for reproducible generation.
    """

    sigma_mm: float = 0.02
    seed: int | None = None


def generate_noise(
    num_frames: int,
    num_rows: int,
    num_cols: int,
    config: NoiseConfig,
) -> np.ndarray:
    """Generate additive Gaussian noise (T, R, C, 2).

    Returns:
        (T, R, C, 2) noise array in mm.
    """
    rng = np.random.default_rng(config.seed)
    return rng.normal(0.0, config.sigma_mm, (num_frames, num_rows, num_cols, 2))
