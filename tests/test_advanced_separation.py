"""Tests for advanced separation methods and BP estimation."""

import numpy as np
import pytest

from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.joint_model import JointModelConfig, joint_separate
from src.separation.decomposition import DecompositionConfig, decomposition_separate
from src.separation.metrics import separation_snr, waveform_correlation
from src.estimation.bp_estimation import (
    segment_beats, extract_beat_morphology, extract_features,
    estimate_bp, calibrate, BPCalibration, FEATURE_NAMES,
)


def _make_dataset(seed=42, num_frames=300, artifact_amp=1.0, pulse_amp=0.15):
    cfg = GeneratorConfig(
        num_frames=num_frames, fps=30.0,
        pulse=PulseConfig(amplitude_mm=pulse_amp, heart_rate_bpm=72.0, sigma_mm=3.0),
        artifact=ArtifactConfig(degree=2, amplitude_mm=artifact_amp, seed=seed),
        noise=NoiseConfig(sigma_mm=0.01, seed=seed + 1),
    )
    return generate(cfg)


class TestJointModel:
    def test_output_shapes(self):
        ds = _make_dataset()
        gt = ds.ground_truth
        disp = ds.markers.displacements_from_rest()
        pulse, artifact = joint_separate(
            gt.rest_positions[..., 0], gt.rest_positions[..., 1],
            disp, gt.artery_mask,
        )
        assert pulse.shape == disp.shape
        assert artifact.shape == disp.shape

    def test_improves_snr(self):
        """Joint model should improve SNR over raw signal."""
        ds = _make_dataset()
        gt = ds.ground_truth
        disp = ds.markers.displacements_from_rest()
        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
        artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]

        pulse, artifact = joint_separate(
            gt.rest_positions[..., 0], gt.rest_positions[..., 1],
            disp, gt.artery_mask,
        )

        raw_snr = separation_snr(disp, pulse_rel, artifact_rel)
        joint_snr = separation_snr(pulse, pulse_rel, artifact_rel)
        assert joint_snr > raw_snr

    def test_waveform_correlation(self):
        """Joint model should recover pulse waveform."""
        ds = _make_dataset()
        gt = ds.ground_truth
        disp = ds.markers.displacements_from_rest()
        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]

        pulse, _ = joint_separate(
            gt.rest_positions[..., 0], gt.rest_positions[..., 1],
            disp, gt.artery_mask,
        )

        corr = waveform_correlation(pulse, pulse_rel, gt.artery_mask)
        assert corr > 0.8, f"Joint model waveform correlation {corr:.3f} too low"


class TestDecomposition:
    def test_pca_output_shapes(self):
        ds = _make_dataset()
        disp = ds.markers.displacements_from_rest()
        cfg = DecompositionConfig(n_components=5, method="pca")
        pulse, artifact = decomposition_separate(disp, ds.ground_truth.artery_mask, cfg)
        assert pulse.shape == disp.shape
        assert artifact.shape == disp.shape

    def test_ica_output_shapes(self):
        ds = _make_dataset()
        disp = ds.markers.displacements_from_rest()
        cfg = DecompositionConfig(n_components=5, method="ica")
        pulse, artifact = decomposition_separate(disp, ds.ground_truth.artery_mask, cfg)
        assert pulse.shape == disp.shape

    def test_pca_improves_snr(self):
        """PCA should improve SNR over raw."""
        ds = _make_dataset()
        gt = ds.ground_truth
        disp = ds.markers.displacements_from_rest()
        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
        artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]

        cfg = DecompositionConfig(n_components=10, method="pca")
        pulse, _ = decomposition_separate(disp, gt.artery_mask, cfg)

        raw_snr = separation_snr(disp, pulse_rel, artifact_rel)
        pca_snr = separation_snr(pulse, pulse_rel, artifact_rel)
        assert pca_snr > raw_snr


class TestBeatSegmentation:
    def test_segments_correct_number(self):
        """Should find ~8 beats in a 10-second, 72 BPM signal."""
        fps = 30.0
        T = 300  # 10 seconds
        t = np.arange(T) / fps
        # Simple sinusoidal pulse at 72 BPM = 1.2 Hz
        waveform = np.sin(2 * np.pi * 1.2 * t)
        beats = segment_beats(waveform, fps)
        # ~12 cycles in 10 seconds, expect 10-12 beats (between peaks)
        assert 8 <= len(beats) <= 13, f"Found {len(beats)} beats, expected ~11"

    def test_beat_duration(self):
        fps = 30.0
        T = 300
        t = np.arange(T) / fps
        waveform = np.sin(2 * np.pi * 1.2 * t)
        beats = segment_beats(waveform, fps)
        if beats:
            durations = [(e - s) / fps for s, e in beats]
            mean_dur = np.mean(durations)
            assert 0.7 < mean_dur < 1.0, f"Mean beat duration {mean_dur:.2f}s"


class TestBeatMorphology:
    def test_extracts_features(self):
        fps = 30.0
        T = 300
        t = np.arange(T) / fps
        waveform = np.sin(2 * np.pi * 1.2 * t)
        beats = segment_beats(waveform, fps)
        assert len(beats) > 0
        morph = extract_beat_morphology(waveform, beats[0][0], beats[0][1], fps)
        assert morph.systolic_peak_value > 0
        assert morph.beat_duration > 0

    def test_feature_vector_length(self):
        fps = 30.0
        T = 300
        t = np.arange(T) / fps
        waveform = np.sin(2 * np.pi * 1.2 * t)
        beats = segment_beats(waveform, fps)
        morph = extract_beat_morphology(waveform, beats[0][0], beats[0][1], fps)
        features = extract_features(morph)
        assert len(features) == len(FEATURE_NAMES)


class TestBPEstimation:
    def test_estimate_without_calibration(self):
        fps = 30.0
        T = 300
        t = np.arange(T) / fps
        waveform = np.sin(2 * np.pi * 1.2 * t)
        result = estimate_bp(waveform, fps)
        assert 60 < result.heart_rate_bpm < 80
        assert result.confidence > 0

    def test_calibration_and_prediction(self):
        """Calibrate on synthetic data and predict."""
        fps = 30.0
        T = 300
        t = np.arange(T) / fps
        waveform = np.sin(2 * np.pi * 1.2 * t)
        beats = segment_beats(waveform, fps)
        morphologies = [extract_beat_morphology(waveform, s, e, fps) for s, e in beats]
        features_list = [extract_features(m) for m in morphologies]

        # Create synthetic calibration data (5 "subjects")
        all_features = []
        sys_refs = []
        dia_refs = []
        for i in range(5):
            # Slightly perturb features
            f = features_list[0] * (1 + 0.1 * (i - 2))
            all_features.append(f)
            sys_refs.append(120.0 + 5 * (i - 2))  # 110-130 mmHg
            dia_refs.append(80.0 + 3 * (i - 2))   # 74-86 mmHg

        cal = calibrate(all_features, sys_refs, dia_refs)
        assert len(cal.systolic_weights) == len(FEATURE_NAMES)

        # Predict on the middle subject
        sys_pred, dia_pred = cal.predict(all_features[2])
        assert 100 < sys_pred < 140
        assert 60 < dia_pred < 100

    def test_empty_waveform(self):
        """Should handle waveform with no detectable beats."""
        waveform = np.zeros(100)
        result = estimate_bp(waveform, 30.0)
        assert result.confidence == 0.0
