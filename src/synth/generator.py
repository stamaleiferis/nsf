"""Compose pulse, artifact, and noise into a complete synthetic dataset."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.data.markers import GroundTruth, MarkerTimeSeries, SyntheticDataset
from src.synth.pulse import PulseConfig, pulse_waveform, pulse_displacement_field, artery_mask
from src.synth.artifact import ArtifactConfig, artifact_displacement_field
from src.synth.noise import NoiseConfig, generate_noise


@dataclass
class GeneratorConfig:
    """Top-level configuration for synthetic data generation.

    Attributes:
        num_rows: Number of marker rows.
        num_cols: Number of marker columns.
        grid_spacing_mm: Physical spacing between markers in mm.
        num_frames: Number of frames to generate.
        fps: Frame rate in Hz.
        pulse: Pulse model configuration.
        artifact: Artifact model configuration.
        noise: Noise model configuration.
    """

    num_rows: int = 19
    num_cols: int = 14
    grid_spacing_mm: float = 2.0
    num_frames: int = 300
    fps: float = 30.0
    pulse: PulseConfig | None = None
    artifact: ArtifactConfig | None = None
    noise: NoiseConfig | None = None

    def __post_init__(self) -> None:
        if self.pulse is None:
            self.pulse = PulseConfig()
        if self.artifact is None:
            self.artifact = ArtifactConfig()
        if self.noise is None:
            self.noise = NoiseConfig()


def make_grid(config: GeneratorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Create the rest-position marker grid.

    Returns:
        grid_x: (R, C) X coordinates in mm, centered at origin.
        grid_y: (R, C) Y coordinates in mm, centered at origin.
    """
    xs = np.arange(config.num_cols) * config.grid_spacing_mm
    ys = np.arange(config.num_rows) * config.grid_spacing_mm
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    grid_x, grid_y = np.meshgrid(xs, ys)  # (R, C) each
    return grid_x, grid_y


def generate(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Generate a complete synthetic dataset.

    Args:
        config: Generation configuration. Uses defaults if None.

    Returns:
        SyntheticDataset with observed markers and ground truth.
    """
    if config is None:
        config = GeneratorConfig()

    grid_x, grid_y = make_grid(config)
    R, C = grid_x.shape
    T = config.num_frames

    # Rest positions (R, C, 2)
    rest_positions = np.stack([grid_x, grid_y], axis=-1)

    # Time vector
    t = np.arange(T) / config.fps

    # Pulse component
    waveform = pulse_waveform(t, config.pulse)
    pulse_disp = pulse_displacement_field(grid_x, grid_y, waveform, config.pulse)
    mask = artery_mask(grid_x, grid_y, config.pulse)

    # Artifact component
    artifact_disp = artifact_displacement_field(
        grid_x, grid_y, T, config.fps, config.artifact
    )

    # Noise component
    noise_disp = generate_noise(T, R, C, config.noise)

    # Total positions = rest + pulse + artifact + noise
    positions = (
        rest_positions[None, :, :, :]
        + pulse_disp
        + artifact_disp
        + noise_disp
    )

    # Visibility: all markers visible (no occlusion model in lightweight generator)
    visibility = np.ones((T, R, C), dtype=np.float64)

    markers = MarkerTimeSeries(
        positions=positions,
        visibility=visibility,
        fps=config.fps,
        grid_spacing_mm=config.grid_spacing_mm,
    )

    ground_truth = GroundTruth(
        pulse_displacement=pulse_disp,
        artifact_displacement=artifact_disp,
        noise=noise_disp,
        artery_mask=mask,
        pulse_waveform=waveform,
        rest_positions=rest_positions,
    )

    return SyntheticDataset(markers=markers, ground_truth=ground_truth)
