"""Parameter library: store and retrieve realistic parameter sets.

Builds a library of estimated parameters from multiple recordings
for use in generating statistically realistic synthetic data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.estimation.spatial_fit import SpatialFitResult
from src.estimation.artifact_stats import ArtifactStats


@dataclass
class ParameterSet:
    """A complete set of estimated parameters from one recording.

    Attributes:
        name: Identifier for this parameter set.
        heart_rate_bpm: Estimated heart rate.
        sigma_mm: Gaussian width of artery influence.
        artery_center_x_mm: Artery X position.
        artery_center_y_mm: Artery Y position.
        artery_angle_deg: Artery angle from Y-axis.
        pulse_amplitude_y_mm: Peak Y-displacement from pulse.
        lateral_shear_ratio: X/Y displacement ratio for pulse.
        artifact_rms_mm: RMS artifact displacement.
        artifact_spectral_centroid_hz: Artifact frequency center.
        artifact_x_y_ratio: Artifact X/Y displacement ratio.
        source: Description of data source.
    """

    name: str
    heart_rate_bpm: float
    sigma_mm: float
    artery_center_x_mm: float
    artery_center_y_mm: float
    artery_angle_deg: float
    pulse_amplitude_y_mm: float
    lateral_shear_ratio: float
    artifact_rms_mm: float
    artifact_spectral_centroid_hz: float
    artifact_x_y_ratio: float
    source: str = ""


class ParameterLibrary:
    """Collection of parameter sets for synthetic generation."""

    def __init__(self) -> None:
        self.entries: list[ParameterSet] = []

    def add(self, params: ParameterSet) -> None:
        """Add a parameter set to the library."""
        self.entries.append(params)

    def add_from_estimates(
        self,
        name: str,
        spatial: SpatialFitResult,
        artifact: ArtifactStats,
        heart_rate_bpm: float,
        source: str = "",
    ) -> ParameterSet:
        """Create and add a parameter set from estimation results."""
        ps = ParameterSet(
            name=name,
            heart_rate_bpm=heart_rate_bpm,
            sigma_mm=spatial.sigma_mm,
            artery_center_x_mm=spatial.center_x_mm,
            artery_center_y_mm=spatial.center_y_mm,
            artery_angle_deg=spatial.angle_deg,
            pulse_amplitude_y_mm=spatial.amplitude_y,
            lateral_shear_ratio=spatial.lateral_shear_ratio,
            artifact_rms_mm=artifact.rms_amplitude_mm,
            artifact_spectral_centroid_hz=artifact.spectral_centroid_hz,
            artifact_x_y_ratio=artifact.x_y_ratio,
            source=source,
        )
        self.add(ps)
        return ps

    def save(self, path: str | Path) -> None:
        """Save library to JSON file."""
        path = Path(path)
        data = [asdict(e) for e in self.entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load library from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        self.entries = [ParameterSet(**d) for d in data]

    def summary(self) -> dict:
        """Statistical summary across all parameter sets."""
        if not self.entries:
            return {}

        fields = [
            "heart_rate_bpm", "sigma_mm", "pulse_amplitude_y_mm",
            "lateral_shear_ratio", "artifact_rms_mm",
            "artifact_spectral_centroid_hz",
        ]
        result = {}
        for field in fields:
            vals = [getattr(e, field) for e in self.entries]
            result[field] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return result

    def to_generator_configs(self) -> list[dict]:
        """Convert each parameter set to GeneratorConfig kwargs."""
        configs = []
        for ps in self.entries:
            configs.append({
                "pulse_heart_rate_bpm": ps.heart_rate_bpm,
                "pulse_sigma_mm": ps.sigma_mm,
                "pulse_artery_center_x_mm": ps.artery_center_x_mm,
                "pulse_artery_center_y_mm": ps.artery_center_y_mm,
                "pulse_artery_angle_deg": ps.artery_angle_deg,
                "pulse_amplitude_mm": ps.pulse_amplitude_y_mm,
                "pulse_lateral_shear_ratio": ps.lateral_shear_ratio,
                "artifact_amplitude_mm": ps.artifact_rms_mm,
            })
        return configs
