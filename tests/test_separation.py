"""Tests for the signal separation pipeline."""

import numpy as np
import pytest

from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.temporal_filter import FilterConfig, lowpass_positions, bandpass_positions
from src.separation.polynomial_fit import (
    PolyFitConfig, fit_polynomial, fit_polynomial_all_frames, _build_design_matrix,
)
from src.separation.separator import SeparationConfig, separate, weights_from_artery_mask
from src.separation.gaussian_extractor import GaussianExtractorConfig, extract_pulse_gaussian
from src.separation.metrics import (
    separation_snr, waveform_correlation, spatial_correlation,
    artifact_residual_fraction, evaluate,
)


# ---- Temporal Filter Tests ----

class TestTemporalFilter:
    def test_lowpass_shape(self):
        pos = np.random.randn(100, 5, 4, 2)
        cfg = FilterConfig(lowpass_cutoff_hz=2.0)
        lp = lowpass_positions(pos, 30.0, cfg)
        assert lp.shape == pos.shape

    def test_lowpass_removes_high_freq(self):
        """A pure high-frequency signal should be nearly zeroed by lowpass."""
        T = 300
        fps = 30.0
        t = np.arange(T) / fps
        # 5 Hz sine — well above 0.5 Hz cutoff
        signal = np.sin(2 * np.pi * 5.0 * t)
        pos = np.zeros((T, 1, 1, 2))
        pos[:, 0, 0, 1] = signal

        cfg = FilterConfig(lowpass_cutoff_hz=0.5, filter_order=4)
        lp = lowpass_positions(pos, fps, cfg)
        # Lowpassed signal power should be << original
        assert np.mean(lp[:, 0, 0, 1] ** 2) < 0.01 * np.mean(signal ** 2)

    def test_lowpass_preserves_low_freq(self):
        """A pure low-frequency signal should pass through lowpass."""
        T = 300
        fps = 30.0
        t = np.arange(T) / fps
        # 0.1 Hz sine — well below 0.5 Hz cutoff
        signal = np.sin(2 * np.pi * 0.1 * t)
        pos = np.zeros((T, 1, 1, 2))
        pos[:, 0, 0, 1] = signal

        cfg = FilterConfig(lowpass_cutoff_hz=0.5, filter_order=4)
        lp = lowpass_positions(pos, fps, cfg)
        corr = np.corrcoef(lp[:, 0, 0, 1], signal)[0, 1]
        assert corr > 0.99

    def test_bandpass_shape(self):
        pos = np.random.randn(100, 3, 3, 2)
        cfg = FilterConfig(bandpass_low_hz=0.5, bandpass_high_hz=10.0)
        bp = bandpass_positions(pos, 30.0, cfg)
        assert bp.shape == pos.shape


# ---- Polynomial Fit Tests ----

class TestPolynomialFit:
    def test_design_matrix_shape(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        A = _build_design_matrix(x, y, degree=2)
        assert A.shape == (10, 6)  # 6 terms for degree 2

    def test_fit_recovers_affine(self):
        """Polynomial fit should exactly recover an affine displacement field."""
        xs = np.linspace(-1, 1, 10)
        ys = np.linspace(-1, 1, 8)
        gx, gy = np.meshgrid(xs, ys)

        # True affine: dx = 0.5 + 0.3*x - 0.2*y, dy = -0.1 + 0.4*x + 0.1*y
        true_disp = np.empty((*gx.shape, 2))
        true_disp[..., 0] = 0.5 + 0.3 * gx - 0.2 * gy
        true_disp[..., 1] = -0.1 + 0.4 * gx + 0.1 * gy

        weights = np.ones(gx.shape)
        cfg = PolyFitConfig(degree=1, regularization=0.0)
        fit = fit_polynomial(gx, gy, true_disp, weights, cfg)
        np.testing.assert_allclose(fit, true_disp, atol=1e-10)

    def test_fit_recovers_quadratic(self):
        """Polynomial fit with degree 2 should recover quadratic field."""
        xs = np.linspace(-1, 1, 10)
        ys = np.linspace(-1, 1, 8)
        gx, gy = np.meshgrid(xs, ys)

        true_disp = np.empty((*gx.shape, 2))
        true_disp[..., 0] = 0.1 * gx ** 2 + 0.05 * gx * gy
        true_disp[..., 1] = -0.08 * gy ** 2 + 0.03 * gx

        weights = np.ones(gx.shape)
        cfg = PolyFitConfig(degree=2, regularization=0.0)
        fit = fit_polynomial(gx, gy, true_disp, weights, cfg)
        np.testing.assert_allclose(fit, true_disp, atol=1e-8)

    def test_weighted_fit_ignores_low_weight(self):
        """Markers with zero weight should not affect the fit."""
        xs = np.linspace(-1, 1, 10)
        ys = np.linspace(-1, 1, 8)
        gx, gy = np.meshgrid(xs, ys)

        # Affine artifact everywhere
        disp = np.empty((*gx.shape, 2))
        disp[..., 0] = 0.5 + 0.3 * gx
        disp[..., 1] = 0.2 * gy

        # Add large pulse-like signal at center markers
        center_mask = np.abs(gx) < 0.3
        disp[center_mask, 0] += 10.0  # big contamination

        # Give zero weight to contaminated markers
        weights = np.ones(gx.shape)
        weights[center_mask] = 0.0

        cfg = PolyFitConfig(degree=1, regularization=1e-10)
        fit = fit_polynomial(gx, gy, disp, weights, cfg)

        # The fit should recover the affine artifact, not the contamination
        expected = np.empty((*gx.shape, 2))
        expected[..., 0] = 0.5 + 0.3 * gx
        expected[..., 1] = 0.2 * gy
        np.testing.assert_allclose(fit, expected, atol=0.1)

    def test_all_frames_matches_per_frame(self):
        xs = np.linspace(-1, 1, 5)
        ys = np.linspace(-1, 1, 4)
        gx, gy = np.meshgrid(xs, ys)

        T = 5
        disp = np.random.randn(T, 4, 5, 2)
        weights = np.ones((4, 5))
        cfg = PolyFitConfig(degree=2, regularization=1e-6)

        all_fit = fit_polynomial_all_frames(gx, gy, disp, weights, cfg)
        for t in range(T):
            per_frame = fit_polynomial(gx, gy, disp[t], weights, cfg)
            np.testing.assert_allclose(all_fit[t], per_frame, atol=1e-8)


# ---- Metrics Tests ----

class TestMetrics:
    def test_perfect_separation_snr(self):
        true_pulse = np.random.randn(10, 3, 3, 2)
        snr = separation_snr(true_pulse, true_pulse, np.ones_like(true_pulse))
        assert snr == float("inf")

    def test_waveform_correlation_perfect(self):
        pulse = np.random.randn(50, 3, 3, 2)
        mask = np.ones((3, 3))
        corr = waveform_correlation(pulse, pulse, mask)
        assert corr == pytest.approx(1.0, abs=1e-10)

    def test_spatial_correlation_perfect(self):
        pulse = np.random.randn(50, 3, 3, 2)
        corr = spatial_correlation(pulse, pulse)
        assert corr == pytest.approx(1.0, abs=1e-10)

    def test_artifact_residual_fraction_perfect(self):
        artifact = np.random.randn(10, 3, 3, 2)
        frac = artifact_residual_fraction(artifact, artifact)
        assert frac == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_returns_metrics(self):
        T, R, C = 50, 3, 3
        pulse = np.random.randn(T, R, C, 2)
        artifact = np.random.randn(T, R, C, 2)
        mask = np.ones((R, C))
        m = evaluate(pulse, artifact, pulse, artifact, mask)
        assert m.separation_snr_db == float("inf")
        assert m.waveform_correlation == pytest.approx(1.0, abs=1e-10)
        assert m.artifact_residual_fraction == pytest.approx(0.0, abs=1e-10)


# ---- Gaussian Extractor Tests ----

class TestGaussianExtractor:
    def test_extracts_mask_shaped_signal(self):
        """Gaussian extractor should recover signal that matches the mask pattern."""
        R, C, T = 10, 8, 50
        mask = np.zeros((R, C))
        mask[4:6, 3:5] = 1.0  # concentrated artery region

        # Create residual that is mask-shaped pulse + random noise
        pulse_signal = np.sin(np.linspace(0, 10 * np.pi, T))
        residual = np.zeros((T, R, C, 2))
        residual[:, :, :, 1] = pulse_signal[:, None, None] * mask[None, :, :]
        residual += np.random.randn(T, R, C, 2) * 0.1  # add noise

        gx, gy = np.meshgrid(np.arange(C, dtype=float), np.arange(R, dtype=float))
        extracted = extract_pulse_gaussian(residual, gx, gy, mask)

        # At artery markers, extracted should match the pulse
        center_extracted = extracted[:, 4, 3, 1]
        corr = np.corrcoef(center_extracted, pulse_signal)[0, 1]
        assert corr > 0.95, f"Extraction correlation {corr:.3f} too low"

    def test_rejects_non_mask_signal(self):
        """Signals at non-artery locations should be rejected."""
        R, C, T = 10, 8, 50
        mask = np.zeros((R, C))
        mask[4:6, 3:5] = 1.0

        # Residual with signal ONLY at non-artery locations
        residual = np.zeros((T, R, C, 2))
        residual[:, 0, 0, 1] = np.sin(np.linspace(0, 10 * np.pi, T))

        gx, gy = np.meshgrid(np.arange(C, dtype=float), np.arange(R, dtype=float))
        extracted = extract_pulse_gaussian(residual, gx, gy, mask)

        # Extracted signal should be nearly zero (non-artery signal rejected)
        assert np.max(np.abs(extracted)) < 0.1

    def test_output_shape(self):
        T, R, C = 20, 5, 4
        residual = np.random.randn(T, R, C, 2)
        mask = np.ones((R, C))
        gx, gy = np.meshgrid(np.arange(C, dtype=float), np.arange(R, dtype=float))
        extracted = extract_pulse_gaussian(residual, gx, gy, mask)
        assert extracted.shape == (T, R, C, 2)


# ---- Separator Tests ----

class TestSeparatorWeights:
    def test_weights_invert_mask(self):
        mask = np.array([[0.0, 0.5, 1.0]])
        w = weights_from_artery_mask(mask)
        np.testing.assert_array_equal(w, [[1.0, 0.5, 0.0]])


# ---- Integration: End-to-End Separation ----

class TestEndToEnd:
    def _make_dataset(self, artifact_amp=1.0, pulse_amp=0.15, noise_sigma=0.01, seed=42):
        cfg = GeneratorConfig(
            num_rows=19,
            num_cols=14,
            num_frames=300,
            fps=30.0,
            pulse=PulseConfig(
                amplitude_mm=pulse_amp,
                heart_rate_bpm=72.0,
                sigma_mm=3.0,
            ),
            artifact=ArtifactConfig(
                degree=2,
                amplitude_mm=artifact_amp,
                max_freq_hz=2.0,
                seed=seed,
            ),
            noise=NoiseConfig(sigma_mm=noise_sigma, seed=seed + 1),
        )
        return generate(cfg)

    @staticmethod
    def _relative_gt(gt):
        """Make ground truth relative to frame 0 (matching separator output)."""
        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
        artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
        return pulse_rel, artifact_rel

    def test_separation_improves_snr(self):
        """Separation should improve SNR over raw signal."""
        ds = self._make_dataset()
        gt = ds.ground_truth
        pulse_rel, artifact_rel = self._relative_gt(gt)

        sep_cfg = SeparationConfig(
            polyfit=PolyFitConfig(degree=2),
            use_temporal_prefilter=False,
        )
        result = separate(
            ds.markers,
            gt.rest_positions[..., 0],
            gt.rest_positions[..., 1],
            gt.artery_mask,
            sep_cfg,
        )

        # SNR of raw signal (pulse vs artifact in raw data)
        raw_snr = separation_snr(
            ds.markers.displacements_from_rest(),
            pulse_rel,
            artifact_rel,
        )
        # SNR after separation
        sep_snr = separation_snr(
            result.recovered_pulse,
            pulse_rel,
            artifact_rel,
        )
        assert sep_snr > raw_snr, f"Separation SNR {sep_snr:.1f} <= raw SNR {raw_snr:.1f}"

    def test_waveform_fidelity(self):
        """Recovered pulse waveform should correlate well with ground truth."""
        ds = self._make_dataset()
        gt = ds.ground_truth
        pulse_rel, _ = self._relative_gt(gt)

        sep_cfg = SeparationConfig(use_temporal_prefilter=False)
        result = separate(
            ds.markers,
            gt.rest_positions[..., 0],
            gt.rest_positions[..., 1],
            gt.artery_mask,
            sep_cfg,
        )

        corr = waveform_correlation(
            result.recovered_pulse, pulse_rel, gt.artery_mask
        )
        assert corr > 0.8, f"Waveform correlation {corr:.3f} too low"

    def test_artifact_residual_low(self):
        """Polynomial artifact residual should be small for polynomial artifacts."""
        ds = self._make_dataset()
        gt = ds.ground_truth
        _, artifact_rel = self._relative_gt(gt)

        sep_cfg = SeparationConfig(use_temporal_prefilter=False)
        result = separate(
            ds.markers,
            gt.rest_positions[..., 0],
            gt.rest_positions[..., 1],
            gt.artery_mask,
            sep_cfg,
        )

        frac = artifact_residual_fraction(
            result.estimated_artifact, artifact_rel
        )
        assert frac < 0.05, f"Artifact residual fraction {frac:.3f} too high"

    def test_full_metrics(self):
        """Run full evaluation and check all metrics."""
        ds = self._make_dataset()
        gt = ds.ground_truth
        pulse_rel, artifact_rel = self._relative_gt(gt)

        sep_cfg = SeparationConfig(use_temporal_prefilter=False)
        result = separate(
            ds.markers,
            gt.rest_positions[..., 0],
            gt.rest_positions[..., 1],
            gt.artery_mask,
            sep_cfg,
        )

        m = evaluate(
            result.recovered_pulse,
            result.estimated_artifact,
            pulse_rel,
            artifact_rel,
            gt.artery_mask,
        )
        # SNR improvement over raw signal (spec target: >= 10 dB)
        raw_snr = separation_snr(
            ds.markers.displacements_from_rest(), pulse_rel, artifact_rel,
        )
        snr_improvement = m.separation_snr_db - raw_snr
        assert snr_improvement > 10, (
            f"SNR improvement {snr_improvement:.1f} dB should be > 10 dB"
        )
        assert m.waveform_correlation > 0.9
        assert m.spatial_correlation > 0.5
        assert m.artifact_residual_fraction < 0.001

    def test_gaussian_extraction_on_polynomial(self):
        """Gaussian extraction should preserve quality on polynomial artifacts."""
        ds = self._make_dataset()
        gt = ds.ground_truth
        pulse_rel, artifact_rel = self._relative_gt(gt)

        sep_cfg = SeparationConfig(
            use_temporal_prefilter=False,
            use_gaussian_extraction=True,
        )
        result = separate(
            ds.markers,
            gt.rest_positions[..., 0],
            gt.rest_positions[..., 1],
            gt.artery_mask,
            sep_cfg,
        )

        corr = waveform_correlation(
            result.recovered_pulse, pulse_rel, gt.artery_mask
        )
        assert corr > 0.9, f"Gaussian extraction waveform correlation {corr:.3f} too low"
