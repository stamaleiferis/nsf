"""MarkerTimeSeries: core data structure for marker position time series."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarkerTimeSeries:
    """Time series of 2D marker positions on a deformable surface.

    Attributes:
        positions: (T, R, C, 2) XY position of each marker at each frame.
        visibility: (T, R, C) visibility/confidence score per marker [0, 1].
        fps: Frame rate in Hz.
        grid_spacing_mm: Physical spacing between markers in mm.
    """

    positions: np.ndarray  # (T, R, C, 2)
    visibility: np.ndarray  # (T, R, C)
    fps: float = 30.0
    grid_spacing_mm: float = 2.0

    def __post_init__(self) -> None:
        T, R, C, xy = self.positions.shape
        assert xy == 2, f"Last dim of positions must be 2, got {xy}"
        assert self.visibility.shape == (T, R, C), (
            f"visibility shape {self.visibility.shape} != positions shape ({T}, {R}, {C})"
        )

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def num_rows(self) -> int:
        return self.positions.shape[1]

    @property
    def num_cols(self) -> int:
        return self.positions.shape[2]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.fps

    @property
    def velocities(self) -> np.ndarray:
        """(T-1, R, C, 2) frame-to-frame displacement."""
        return np.diff(self.positions, axis=0)

    def displacements_from_rest(self, rest_frame: int = 0) -> np.ndarray:
        """(T, R, C, 2) displacement relative to a reference frame."""
        return self.positions - self.positions[rest_frame : rest_frame + 1]


@dataclass
class GroundTruth:
    """Ground truth decomposition of marker displacements (synthetic data only).

    All displacement arrays are relative to the rest (undeformed) grid positions.

    Attributes:
        pulse_displacement: (T, R, C, 2) pulse-only displacement.
        artifact_displacement: (T, R, C, 2) artifact-only displacement.
        noise: (T, R, C, 2) measurement noise component.
        artery_mask: (R, C) soft mask of artery spatial influence [0, 1].
        pulse_waveform: (T,) scalar pulse amplitude A(t).
        rest_positions: (R, C, 2) undeformed marker grid positions.
    """

    pulse_displacement: np.ndarray      # (T, R, C, 2)
    artifact_displacement: np.ndarray   # (T, R, C, 2)
    noise: np.ndarray                   # (T, R, C, 2)
    artery_mask: np.ndarray             # (R, C)
    pulse_waveform: np.ndarray          # (T,)
    rest_positions: np.ndarray          # (R, C, 2)


@dataclass
class SyntheticDataset:
    """Complete synthetic dataset: observed data + ground truth."""

    markers: MarkerTimeSeries
    ground_truth: GroundTruth

    def separation_snr(self) -> float:
        """Signal-to-noise ratio of pulse vs artifacts in dB."""
        pulse_power = np.mean(self.ground_truth.pulse_displacement ** 2)
        artifact_power = np.mean(self.ground_truth.artifact_displacement ** 2)
        if artifact_power == 0:
            return float("inf")
        return 10.0 * np.log10(pulse_power / artifact_power)
