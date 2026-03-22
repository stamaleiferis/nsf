"""Edge case tests for robustness."""

import numpy as np
import pytest

from src.data.markers import MarkerTimeSeries, GroundTruth, SyntheticDataset
from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.metrics import evaluate, separation_snr


class TestShortSignals:
    def test_very_short_signal(self):
        """Pipeline should handle signals shorter than 2 beats."""
        cfg = GeneratorConfig(num_frames=10, fps=30.0)
        ds = generate(cfg)
        gt = ds.ground_truth
        sep_cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=1),
            use_temporal_prefilter=False,
        )
        result = separate(
            ds.markers, gt.rest_positions[..., 0],
            gt.rest_positions[..., 1], gt.artery_mask, sep_cfg,
        )
        assert result.recovered_pulse.shape == ds.markers.positions.shape


class TestZeroAmplitude:
    def test_zero_pulse(self):
        """Should handle zero-amplitude pulse (artifact only)."""
        cfg = GeneratorConfig(
            num_frames=60, fps=30.0,
            pulse=PulseConfig(amplitude_mm=0.0),
            artifact=ArtifactConfig(amplitude_mm=1.0, seed=42),
            noise=NoiseConfig(sigma_mm=0.01, seed=43),
        )
        ds = generate(cfg)
        gt = ds.ground_truth

        sep_cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=2),
            use_temporal_prefilter=False,
        )
        result = separate(
            ds.markers, gt.rest_positions[..., 0],
            gt.rest_positions[..., 1], gt.artery_mask, sep_cfg,
        )
        # Recovered pulse should be near zero
        assert np.sqrt(np.mean(result.recovered_pulse ** 2)) < 0.1

    def test_zero_artifact(self):
        """Should handle zero-amplitude artifact (pulse only)."""
        cfg = GeneratorConfig(
            num_frames=60, fps=30.0,
            pulse=PulseConfig(amplitude_mm=0.15),
            artifact=ArtifactConfig(amplitude_mm=0.0, seed=42),
            noise=NoiseConfig(sigma_mm=0.01, seed=43),
        )
        ds = generate(cfg)
        gt = ds.ground_truth

        sep_cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=2),
            use_temporal_prefilter=False,
        )
        result = separate(
            ds.markers, gt.rest_positions[..., 0],
            gt.rest_positions[..., 1], gt.artery_mask, sep_cfg,
        )
        assert result.recovered_pulse.shape == ds.markers.positions.shape


class TestSmallGrid:
    def test_3x3_grid(self):
        """Should work with minimal grid size."""
        cfg = GeneratorConfig(
            num_rows=3, num_cols=3, num_frames=60, fps=30.0,
            pulse=PulseConfig(amplitude_mm=0.15),
            artifact=ArtifactConfig(degree=1, amplitude_mm=0.5, seed=42),
            noise=NoiseConfig(sigma_mm=0.01, seed=43),
        )
        ds = generate(cfg)
        gt = ds.ground_truth

        sep_cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=1),
            use_temporal_prefilter=False,
        )
        result = separate(
            ds.markers, gt.rest_positions[..., 0],
            gt.rest_positions[..., 1], gt.artery_mask, sep_cfg,
        )
        assert result.recovered_pulse.shape == (60, 3, 3, 2)


class TestMethodComparison:
    """Verify all separation methods produce valid output on the same dataset."""

    def _make(self):
        cfg = GeneratorConfig(
            num_frames=100, fps=30.0,
            pulse=PulseConfig(amplitude_mm=0.15),
            artifact=ArtifactConfig(degree=2, amplitude_mm=1.0, seed=42),
            noise=NoiseConfig(sigma_mm=0.01, seed=43),
        )
        return generate(cfg)

    def test_all_methods_run(self):
        from src.separation.joint_model import JointModelConfig, joint_separate
        from src.separation.decomposition import DecompositionConfig, decomposition_separate
        from src.separation.subspace_separation import SubspaceConfig, subspace_separate

        ds = self._make()
        gt = ds.ground_truth
        gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]
        disp = ds.markers.displacements_from_rest()
        mask = gt.artery_mask

        # Sequential
        sep_cfg = SeparationConfig(use_temporal_prefilter=False)
        r = separate(ds.markers, gx, gy, mask, sep_cfg)
        assert r.recovered_pulse.shape == disp.shape

        # Joint
        p, a = joint_separate(gx, gy, disp, mask)
        assert p.shape == disp.shape

        # PCA
        p, a = decomposition_separate(disp, mask, DecompositionConfig(method="pca"))
        assert p.shape == disp.shape

        # Subspace
        p, a = subspace_separate(gx, gy, disp, mask)
        assert p.shape == disp.shape
