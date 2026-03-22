"""Tests for the lightweight synthetic generator."""

import numpy as np
import pytest

from src.synth.pulse import PulseConfig, pulse_waveform, artery_mask, pulse_displacement_field
from src.synth.artifact import ArtifactConfig, artifact_displacement_field, _poly_terms
from src.synth.noise import NoiseConfig, generate_noise
from src.synth.generator import GeneratorConfig, generate, make_grid


class TestPulse:
    def test_waveform_range(self):
        t = np.linspace(0, 5, 500)
        w = pulse_waveform(t, PulseConfig())
        assert w.min() >= 0.0
        assert w.max() <= 1.0 + 1e-10

    def test_waveform_shape(self):
        t = np.linspace(0, 1, 100)
        w = pulse_waveform(t, PulseConfig())
        assert w.shape == (100,)

    def test_artery_mask_peaked_at_center(self):
        cfg = PulseConfig(artery_center_x_mm=0, artery_center_y_mm=0)
        xs = np.linspace(-10, 10, 21)
        ys = np.linspace(-10, 10, 21)
        gx, gy = np.meshgrid(xs, ys)
        m = artery_mask(gx, gy, cfg)
        # Peak at center
        center = m[10, 10]
        assert center == pytest.approx(1.0)
        # Falls off away from center
        assert m[10, 0] < center
        assert m[10, 20] < center

    def test_artery_mask_shape(self):
        cfg = PulseConfig()
        gx = np.zeros((5, 4))
        gy = np.zeros((5, 4))
        m = artery_mask(gx, gy, cfg)
        assert m.shape == (5, 4)

    def test_displacement_field_shape(self):
        cfg = PulseConfig()
        gx, gy = np.meshgrid(np.arange(4) * 2.0, np.arange(5) * 2.0)
        t = np.linspace(0, 1, 30)
        w = pulse_waveform(t, cfg)
        d = pulse_displacement_field(gx, gy, w, cfg)
        assert d.shape == (30, 5, 4, 2)

    def test_displacement_zero_at_waveform_zero(self):
        cfg = PulseConfig()
        gx, gy = np.meshgrid(np.arange(3) * 2.0, np.arange(3) * 2.0)
        w = np.zeros(10)
        d = pulse_displacement_field(gx, gy, w, cfg)
        np.testing.assert_array_equal(d, 0.0)

    def test_displacement_scales_with_amplitude(self):
        gx, gy = np.meshgrid(np.arange(3) * 2.0, np.arange(3) * 2.0)
        w = np.ones(5)

        cfg1 = PulseConfig(amplitude_mm=0.1)
        cfg2 = PulseConfig(amplitude_mm=0.2)
        d1 = pulse_displacement_field(gx, gy, w, cfg1)
        d2 = pulse_displacement_field(gx, gy, w, cfg2)
        np.testing.assert_allclose(d2, d1 * 2.0, atol=1e-12)


class TestArtifact:
    def test_poly_terms_degree1(self):
        terms = _poly_terms(1)
        assert set(terms) == {(0, 0), (1, 0), (0, 1)}

    def test_poly_terms_degree2(self):
        terms = _poly_terms(2)
        assert len(terms) == 6  # 1 + 2 + 3

    def test_displacement_shape(self):
        gx, gy = np.meshgrid(np.arange(4) * 2.0, np.arange(5) * 2.0)
        cfg = ArtifactConfig(seed=42)
        d = artifact_displacement_field(gx, gy, 30, 30.0, cfg)
        assert d.shape == (30, 5, 4, 2)

    def test_reproducibility(self):
        gx, gy = np.meshgrid(np.arange(3) * 2.0, np.arange(3) * 2.0)
        cfg = ArtifactConfig(seed=123)
        d1 = artifact_displacement_field(gx, gy, 10, 30.0, cfg)
        d2 = artifact_displacement_field(gx, gy, 10, 30.0, cfg)
        np.testing.assert_array_equal(d1, d2)


class TestNoise:
    def test_shape(self):
        cfg = NoiseConfig(seed=0)
        n = generate_noise(10, 5, 4, cfg)
        assert n.shape == (10, 5, 4, 2)

    def test_statistics(self):
        cfg = NoiseConfig(sigma_mm=1.0, seed=42)
        n = generate_noise(10000, 5, 4, cfg)
        assert abs(n.mean()) < 0.05
        assert abs(n.std() - 1.0) < 0.05

    def test_reproducibility(self):
        cfg = NoiseConfig(seed=99)
        n1 = generate_noise(10, 3, 3, cfg)
        n2 = generate_noise(10, 3, 3, cfg)
        np.testing.assert_array_equal(n1, n2)


class TestGenerator:
    def test_default_generation(self):
        cfg = GeneratorConfig(num_frames=30)
        ds = generate(cfg)
        assert ds.markers.positions.shape == (30, 19, 14, 2)
        assert ds.markers.visibility.shape == (30, 19, 14)
        assert ds.ground_truth.pulse_displacement.shape == (30, 19, 14, 2)
        assert ds.ground_truth.artifact_displacement.shape == (30, 19, 14, 2)
        assert ds.ground_truth.noise.shape == (30, 19, 14, 2)
        assert ds.ground_truth.artery_mask.shape == (19, 14)
        assert ds.ground_truth.pulse_waveform.shape == (30,)
        assert ds.ground_truth.rest_positions.shape == (19, 14, 2)

    def test_positions_are_sum_of_components(self):
        cfg = GeneratorConfig(num_frames=10, num_rows=3, num_cols=3)
        ds = generate(cfg)
        gt = ds.ground_truth
        expected = (
            gt.rest_positions[None, :, :, :]
            + gt.pulse_displacement
            + gt.artifact_displacement
            + gt.noise
        )
        np.testing.assert_allclose(ds.markers.positions, expected, atol=1e-12)

    def test_make_grid_spacing(self):
        cfg = GeneratorConfig(num_rows=5, num_cols=4, grid_spacing_mm=2.0)
        gx, gy = make_grid(cfg)
        # Check spacing
        dx = np.diff(gx, axis=1)
        dy = np.diff(gy, axis=0)
        np.testing.assert_allclose(dx, 2.0)
        np.testing.assert_allclose(dy, 2.0)

    def test_make_grid_centered(self):
        cfg = GeneratorConfig(num_rows=5, num_cols=4, grid_spacing_mm=2.0)
        gx, gy = make_grid(cfg)
        assert abs(gx.mean()) < 1e-10
        assert abs(gy.mean()) < 1e-10

    def test_snr_is_finite(self):
        cfg = GeneratorConfig(num_frames=30)
        ds = generate(cfg)
        snr = ds.separation_snr()
        assert np.isfinite(snr)

    def test_none_config(self):
        ds = generate(None)
        assert ds.markers.num_frames == 300
