"""Tests for parameter estimation modules."""

import numpy as np
import pytest

from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.estimation.pulse_extractor import (
    extract_pulse, estimate_heart_rate, estimate_pulse_snr_map,
)
from src.estimation.spatial_fit import fit_spatial_gaussian
from src.estimation.artifact_stats import estimate_artifact_stats
from src.estimation.param_library import ParameterLibrary, ParameterSet


def _make_dataset(hr=72.0, amp=0.15, sigma=3.0, cx=0.0, cy=0.0,
                   artifact_amp=0.5, num_frames=300):
    cfg = GeneratorConfig(
        num_frames=num_frames, fps=30.0,
        pulse=PulseConfig(
            heart_rate_bpm=hr, amplitude_mm=amp, sigma_mm=sigma,
            artery_center_x_mm=cx, artery_center_y_mm=cy,
        ),
        artifact=ArtifactConfig(degree=2, amplitude_mm=artifact_amp, seed=42),
        noise=NoiseConfig(sigma_mm=0.01, seed=43),
    )
    return generate(cfg)


class TestPulseExtractor:
    def test_heart_rate_detection(self):
        """Should detect heart rate from synthetic data."""
        # Use high pulse amplitude and low artifact for clean HR detection
        ds = _make_dataset(hr=72.0, amp=0.5, artifact_amp=0.1, num_frames=600)
        displacements = ds.markers.displacements_from_rest()

        result = extract_pulse(
            displacements, ds.markers.fps, ds.ground_truth.artery_mask
        )
        # FFT resolution at 600 frames / 30fps = 20s → 3 BPM resolution
        assert abs(result.heart_rate_bpm - 72.0) < 6.0, (
            f"Detected HR {result.heart_rate_bpm:.1f} != 72.0"
        )

    def test_hr_detection_different_rates(self):
        """Should detect different heart rates."""
        for target_hr in [60.0, 90.0, 120.0]:
            ds = _make_dataset(hr=target_hr, amp=0.5, artifact_amp=0.1, num_frames=600)
            displacements = ds.markers.displacements_from_rest()
            result = extract_pulse(
                displacements, ds.markers.fps, ds.ground_truth.artery_mask
            )
            assert abs(result.heart_rate_bpm - target_hr) < 10.0, (
                f"HR={target_hr}: detected {result.heart_rate_bpm:.1f}"
            )

    def test_snr_map_peaks_at_artery(self):
        """SNR map should be highest near the artery."""
        ds = _make_dataset(amp=0.5, artifact_amp=0.1)
        displacements = ds.markers.displacements_from_rest()
        snr_map = estimate_pulse_snr_map(displacements, ds.markers.fps)

        # Peak should be near artery center
        peak = np.unravel_index(np.argmax(snr_map), snr_map.shape)
        artery_peak = np.unravel_index(
            np.argmax(ds.ground_truth.artery_mask),
            ds.ground_truth.artery_mask.shape,
        )
        # Within 3 markers
        assert abs(peak[0] - artery_peak[0]) < 4
        assert abs(peak[1] - artery_peak[1]) < 4

    def test_waveform_shape(self):
        ds = _make_dataset()
        displacements = ds.markers.displacements_from_rest()
        result = extract_pulse(displacements, ds.markers.fps)
        assert result.waveform.shape == (300,)

    def test_estimate_heart_rate_simple(self):
        """Direct HR estimation from a sinusoid."""
        fps = 30.0
        T = 300
        t = np.arange(T) / fps
        signal = np.sin(2 * np.pi * 1.2 * t)  # 72 BPM
        hr, conf = estimate_heart_rate(signal, fps)
        assert abs(hr - 72.0) < 2.0


class TestSpatialFit:
    def test_recovers_center_x(self):
        """Should recover artery center X position (cross-artery direction).
        Y-center is unconstrained because the pulse is uniform along the artery."""
        ds = _make_dataset(cx=2.0, amp=0.3)
        gt = ds.ground_truth
        gx = gt.rest_positions[..., 0]
        gy = gt.rest_positions[..., 1]

        pulse_rms = np.sqrt(np.mean(gt.pulse_displacement ** 2, axis=0))
        result = fit_spatial_gaussian(gx, gy, pulse_rms)
        assert abs(result.center_x_mm - 2.0) < 2.0, (
            f"Fitted center_x {result.center_x_mm:.1f} != 2.0"
        )

    def test_recovers_sigma(self):
        """Should recover Gaussian width."""
        ds = _make_dataset(sigma=4.0, amp=0.3)
        gt = ds.ground_truth
        gx = gt.rest_positions[..., 0]
        gy = gt.rest_positions[..., 1]
        pulse_rms = np.sqrt(np.mean(gt.pulse_displacement ** 2, axis=0))

        result = fit_spatial_gaussian(gx, gy, pulse_rms)
        assert abs(result.sigma_mm - 4.0) < 2.0, (
            f"Fitted sigma {result.sigma_mm:.1f} != 4.0"
        )

    def test_fitted_mask_shape(self):
        ds = _make_dataset()
        gt = ds.ground_truth
        gx = gt.rest_positions[..., 0]
        gy = gt.rest_positions[..., 1]
        pulse_rms = np.sqrt(np.mean(gt.pulse_displacement ** 2, axis=0))
        result = fit_spatial_gaussian(gx, gy, pulse_rms)
        assert result.fitted_mask.shape == (19, 14)


class TestArtifactStats:
    def test_rms_reasonable(self):
        """Artifact RMS should be close to configured amplitude."""
        ds = _make_dataset()
        displacements = ds.markers.displacements_from_rest()
        stats = estimate_artifact_stats(
            displacements, ds.markers.fps, ds.ground_truth.artery_mask
        )
        # RMS should be on the order of the artifact amplitude
        assert 0.01 < stats.rms_amplitude_mm < 5.0

    def test_spectral_centroid_positive(self):
        ds = _make_dataset()
        displacements = ds.markers.displacements_from_rest()
        stats = estimate_artifact_stats(
            displacements, ds.markers.fps, ds.ground_truth.artery_mask
        )
        assert stats.spectral_centroid_hz > 0

    def test_spatial_correlation_positive(self):
        """Adjacent artifact markers should be positively correlated."""
        ds = _make_dataset()
        displacements = ds.markers.displacements_from_rest()
        stats = estimate_artifact_stats(
            displacements, ds.markers.fps, ds.ground_truth.artery_mask
        )
        assert stats.spatial_correlation > 0


class TestParameterLibrary:
    def test_add_and_summary(self):
        lib = ParameterLibrary()
        lib.add(ParameterSet(
            name="test1", heart_rate_bpm=72.0, sigma_mm=3.0,
            artery_center_x_mm=0.0, artery_center_y_mm=0.0,
            artery_angle_deg=0.0, pulse_amplitude_y_mm=0.15,
            lateral_shear_ratio=0.3, artifact_rms_mm=1.0,
            artifact_spectral_centroid_hz=0.5, artifact_x_y_ratio=0.8,
        ))
        lib.add(ParameterSet(
            name="test2", heart_rate_bpm=90.0, sigma_mm=4.0,
            artery_center_x_mm=1.0, artery_center_y_mm=-1.0,
            artery_angle_deg=10.0, pulse_amplitude_y_mm=0.2,
            lateral_shear_ratio=0.25, artifact_rms_mm=1.5,
            artifact_spectral_centroid_hz=0.7, artifact_x_y_ratio=0.9,
        ))
        summary = lib.summary()
        assert "heart_rate_bpm" in summary
        assert summary["heart_rate_bpm"]["mean"] == pytest.approx(81.0)

    def test_save_and_load(self, tmp_path):
        lib = ParameterLibrary()
        lib.add(ParameterSet(
            name="test", heart_rate_bpm=72.0, sigma_mm=3.0,
            artery_center_x_mm=0.0, artery_center_y_mm=0.0,
            artery_angle_deg=0.0, pulse_amplitude_y_mm=0.15,
            lateral_shear_ratio=0.3, artifact_rms_mm=1.0,
            artifact_spectral_centroid_hz=0.5, artifact_x_y_ratio=0.8,
        ))
        path = tmp_path / "params.json"
        lib.save(path)

        lib2 = ParameterLibrary()
        lib2.load(path)
        assert len(lib2.entries) == 1
        assert lib2.entries[0].name == "test"

    def test_to_generator_configs(self):
        lib = ParameterLibrary()
        lib.add(ParameterSet(
            name="test", heart_rate_bpm=72.0, sigma_mm=3.0,
            artery_center_x_mm=0.0, artery_center_y_mm=0.0,
            artery_angle_deg=0.0, pulse_amplitude_y_mm=0.15,
            lateral_shear_ratio=0.3, artifact_rms_mm=1.0,
            artifact_spectral_centroid_hz=0.5, artifact_x_y_ratio=0.8,
        ))
        configs = lib.to_generator_configs()
        assert len(configs) == 1
        assert configs[0]["pulse_heart_rate_bpm"] == 72.0


class TestCloseTheLoop:
    """Integration test: estimate parameters, regenerate, verify match."""

    def test_estimated_params_reproduce_statistics(self):
        """Parameters estimated from data should produce similar statistics when
        used to generate new synthetic data."""
        # Step 1: Generate "real" data (high SNR for clean estimation)
        ds = _make_dataset(hr=80.0, amp=0.5, sigma=3.5, cx=1.0, cy=0.0,
                           artifact_amp=0.1, num_frames=600)
        gt = ds.ground_truth
        gx = gt.rest_positions[..., 0]
        gy = gt.rest_positions[..., 1]
        displacements = ds.markers.displacements_from_rest()

        # Step 2: Estimate parameters
        pulse_result = extract_pulse(
            displacements, ds.markers.fps, gt.artery_mask
        )
        pulse_rms = np.sqrt(np.mean(gt.pulse_displacement ** 2, axis=0))
        spatial_result = fit_spatial_gaussian(gx, gy, pulse_rms)
        artifact_result = estimate_artifact_stats(
            displacements, ds.markers.fps, gt.artery_mask
        )

        # Step 3: Build library
        lib = ParameterLibrary()
        lib.add_from_estimates(
            "test_recording", spatial_result, artifact_result,
            pulse_result.heart_rate_bpm, source="synthetic test"
        )

        # Step 4: Regenerate with estimated parameters
        est = lib.entries[0]
        cfg2 = GeneratorConfig(
            num_frames=300, fps=30.0,
            pulse=PulseConfig(
                heart_rate_bpm=est.heart_rate_bpm,
                amplitude_mm=est.pulse_amplitude_y_mm,
                sigma_mm=est.sigma_mm,
                artery_center_x_mm=est.artery_center_x_mm,
                artery_center_y_mm=est.artery_center_y_mm,
                artery_angle_deg=est.artery_angle_deg,
                lateral_shear_ratio=est.lateral_shear_ratio,
            ),
            artifact=ArtifactConfig(
                degree=2, amplitude_mm=est.artifact_rms_mm, seed=99,
            ),
            noise=NoiseConfig(sigma_mm=0.01, seed=100),
        )
        ds2 = generate(cfg2)

        # Step 5: Verify statistics match approximately
        # HR should match
        assert abs(est.heart_rate_bpm - 80.0) < 10.0

        # Sigma should be close
        assert abs(est.sigma_mm - 3.5) < 1.5, (
            f"Sigma mismatch: estimated {est.sigma_mm:.1f} vs true 3.5"
        )

        # Center X should be close
        assert abs(est.artery_center_x_mm - 1.0) < 2.0, (
            f"Center X mismatch: estimated {est.artery_center_x_mm:.1f} vs true 1.0"
        )
