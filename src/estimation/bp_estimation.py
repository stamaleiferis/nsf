"""Blood pressure estimation from separated pulse waveform.

Extracts morphology features from the pulse waveform and provides
a calibration framework for mapping to blood pressure values.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.signal import find_peaks, butter, sosfiltfilt


@dataclass
class BeatMorphology:
    """Morphological features of a single pulse beat.

    Attributes:
        systolic_peak_value: Amplitude of systolic peak.
        systolic_peak_time: Time of systolic peak within beat (seconds).
        dicrotic_notch_value: Amplitude at dicrotic notch.
        dicrotic_notch_time: Time of dicrotic notch within beat (seconds).
        diastolic_peak_value: Amplitude of diastolic peak.
        diastolic_peak_time: Time of diastolic peak (seconds).
        beat_duration: Total beat duration (seconds).
        systolic_diastolic_ratio: Ratio of systolic to diastolic amplitude.
        augmentation_index: (systolic - dicrotic) / systolic.
        rise_time: Time from onset to systolic peak (seconds).
        area_under_curve: Integral of the beat waveform.
    """

    systolic_peak_value: float
    systolic_peak_time: float
    dicrotic_notch_value: float
    dicrotic_notch_time: float
    diastolic_peak_value: float
    diastolic_peak_time: float
    beat_duration: float
    systolic_diastolic_ratio: float
    augmentation_index: float
    rise_time: float
    area_under_curve: float


@dataclass
class BPEstimate:
    """Blood pressure estimate from pulse waveform analysis.

    Attributes:
        systolic_mmhg: Estimated systolic blood pressure.
        diastolic_mmhg: Estimated diastolic blood pressure.
        mean_arterial_mmhg: Estimated mean arterial pressure.
        heart_rate_bpm: Heart rate derived from beat intervals.
        confidence: Confidence score [0, 1] based on waveform quality.
        beat_morphologies: Per-beat morphology features.
    """

    systolic_mmhg: float
    diastolic_mmhg: float
    mean_arterial_mmhg: float
    heart_rate_bpm: float
    confidence: float
    beat_morphologies: list[BeatMorphology] = field(default_factory=list)


def segment_beats(
    waveform: np.ndarray,
    fps: float,
    hr_range_bpm: tuple[float, float] = (40.0, 180.0),
) -> list[tuple[int, int]]:
    """Segment a pulse waveform into individual beats.

    Args:
        waveform: (T,) pulse waveform signal.
        fps: Sampling rate.
        hr_range_bpm: Expected heart rate range.

    Returns:
        List of (start_idx, end_idx) for each beat.
    """
    # Expected beat duration range
    min_period = 60.0 / hr_range_bpm[1]  # seconds
    max_period = 60.0 / hr_range_bpm[0]

    min_distance = int(min_period * fps * 0.8)
    max_distance = int(max_period * fps * 1.2)

    # Find systolic peaks
    peaks, properties = find_peaks(
        waveform,
        distance=min_distance,
        prominence=0.1 * (waveform.max() - waveform.min()),
    )

    if len(peaks) < 2:
        return []

    # Beats are from one peak to the next
    beats = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        duration = (end - start) / fps
        if min_period * 0.7 <= duration <= max_period * 1.3:
            beats.append((int(start), int(end)))

    return beats


def extract_beat_morphology(
    waveform: np.ndarray,
    start: int,
    end: int,
    fps: float,
) -> BeatMorphology:
    """Extract morphological features from a single beat.

    Args:
        waveform: (T,) full pulse waveform.
        start: Start index of beat.
        end: End index of beat.
        fps: Sampling rate.

    Returns:
        BeatMorphology with extracted features.
    """
    beat = waveform[start:end]
    T_beat = len(beat)
    t_beat = np.arange(T_beat) / fps
    duration = T_beat / fps

    # Systolic peak
    sys_idx = np.argmax(beat)
    systolic_peak_value = float(beat[sys_idx])
    systolic_peak_time = float(t_beat[sys_idx])

    # Dicrotic notch: minimum after systolic peak in first 2/3 of beat
    search_end = min(int(T_beat * 0.75), T_beat)
    if sys_idx + 1 < search_end:
        notch_region = beat[sys_idx + 1:search_end]
        notch_idx = sys_idx + 1 + np.argmin(notch_region)
    else:
        notch_idx = sys_idx

    dicrotic_notch_value = float(beat[notch_idx])
    dicrotic_notch_time = float(t_beat[notch_idx])

    # Diastolic peak: max after dicrotic notch
    if notch_idx + 1 < T_beat:
        dias_region = beat[notch_idx + 1:]
        dias_idx = notch_idx + 1 + np.argmax(dias_region)
    else:
        dias_idx = notch_idx

    diastolic_peak_value = float(beat[dias_idx])
    diastolic_peak_time = float(t_beat[dias_idx])

    # Derived features
    sys_dias_ratio = systolic_peak_value / (diastolic_peak_value + 1e-12)
    aug_index = (systolic_peak_value - dicrotic_notch_value) / (systolic_peak_value + 1e-12)
    rise_time = systolic_peak_time  # from beat onset to peak
    auc = float(np.trapezoid(beat, t_beat))

    return BeatMorphology(
        systolic_peak_value=systolic_peak_value,
        systolic_peak_time=systolic_peak_time,
        dicrotic_notch_value=dicrotic_notch_value,
        dicrotic_notch_time=dicrotic_notch_time,
        diastolic_peak_value=diastolic_peak_value,
        diastolic_peak_time=diastolic_peak_time,
        beat_duration=float(duration),
        systolic_diastolic_ratio=float(sys_dias_ratio),
        augmentation_index=float(aug_index),
        rise_time=float(rise_time),
        area_under_curve=auc,
    )


@dataclass
class BPCalibration:
    """Linear calibration model for BP estimation.

    Maps waveform features to BP using linear regression:
        BP = intercept + Σ(weight_i * feature_i)

    Attributes:
        systolic_weights: Feature weights for systolic BP.
        systolic_intercept: Intercept for systolic BP.
        diastolic_weights: Feature weights for diastolic BP.
        diastolic_intercept: Intercept for diastolic BP.
        feature_names: Names of features used.
    """

    systolic_weights: np.ndarray
    systolic_intercept: float
    diastolic_weights: np.ndarray
    diastolic_intercept: float
    feature_names: list[str]

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Predict systolic and diastolic BP from features.

        Args:
            features: (N_features,) feature vector.

        Returns:
            (systolic_mmhg, diastolic_mmhg).
        """
        sys = float(self.systolic_intercept + features @ self.systolic_weights)
        dia = float(self.diastolic_intercept + features @ self.diastolic_weights)
        return sys, dia


def extract_features(morphology: BeatMorphology) -> np.ndarray:
    """Convert beat morphology to feature vector for BP estimation."""
    return np.array([
        morphology.systolic_peak_value,
        morphology.dicrotic_notch_value,
        morphology.diastolic_peak_value,
        morphology.systolic_diastolic_ratio,
        morphology.augmentation_index,
        morphology.rise_time,
        morphology.beat_duration,
        morphology.area_under_curve,
    ])


FEATURE_NAMES = [
    "systolic_peak_value",
    "dicrotic_notch_value",
    "diastolic_peak_value",
    "systolic_diastolic_ratio",
    "augmentation_index",
    "rise_time",
    "beat_duration",
    "area_under_curve",
]


def calibrate(
    features_list: list[np.ndarray],
    systolic_refs: list[float],
    diastolic_refs: list[float],
) -> BPCalibration:
    """Fit calibration model from reference BP measurements.

    Args:
        features_list: List of feature vectors from different recordings.
        systolic_refs: Corresponding systolic BP references (mmHg).
        diastolic_refs: Corresponding diastolic BP references (mmHg).

    Returns:
        Fitted BPCalibration model.
    """
    X = np.array(features_list)
    y_sys = np.array(systolic_refs)
    y_dia = np.array(diastolic_refs)

    # Add bias column
    X_bias = np.column_stack([np.ones(len(X)), X])

    # Ridge regression
    lam = 1e-3
    reg = lam * np.eye(X_bias.shape[1])
    reg[0, 0] = 0  # don't regularize intercept

    w_sys = np.linalg.solve(X_bias.T @ X_bias + reg, X_bias.T @ y_sys)
    w_dia = np.linalg.solve(X_bias.T @ X_bias + reg, X_bias.T @ y_dia)

    return BPCalibration(
        systolic_weights=w_sys[1:],
        systolic_intercept=float(w_sys[0]),
        diastolic_weights=w_dia[1:],
        diastolic_intercept=float(w_dia[0]),
        feature_names=FEATURE_NAMES,
    )


def estimate_bp(
    waveform: np.ndarray,
    fps: float,
    calibration: BPCalibration | None = None,
) -> BPEstimate:
    """Estimate blood pressure from a pulse waveform.

    Args:
        waveform: (T,) pulse waveform.
        fps: Sampling rate.
        calibration: Optional calibration model. If None, returns morphology only.

    Returns:
        BPEstimate with BP values (if calibrated) and morphology features.
    """
    beats = segment_beats(waveform, fps)

    if not beats:
        return BPEstimate(
            systolic_mmhg=0.0, diastolic_mmhg=0.0, mean_arterial_mmhg=0.0,
            heart_rate_bpm=0.0, confidence=0.0,
        )

    morphologies = [extract_beat_morphology(waveform, s, e, fps) for s, e in beats]

    # Heart rate from beat intervals
    durations = [m.beat_duration for m in morphologies]
    mean_duration = np.mean(durations)
    hr_bpm = 60.0 / mean_duration if mean_duration > 0 else 0.0

    # Confidence based on beat-to-beat consistency
    if len(durations) > 1:
        cv = np.std(durations) / (mean_duration + 1e-12)
        confidence = float(max(0, 1 - cv * 5))  # penalize high variability
    else:
        confidence = 0.5

    # BP estimation if calibrated
    if calibration is not None:
        features_list = [extract_features(m) for m in morphologies]
        bp_predictions = [calibration.predict(f) for f in features_list]
        sys_mean = float(np.mean([p[0] for p in bp_predictions]))
        dia_mean = float(np.mean([p[1] for p in bp_predictions]))
    else:
        sys_mean = 0.0
        dia_mean = 0.0

    map_mmhg = dia_mean + (sys_mean - dia_mean) / 3.0

    return BPEstimate(
        systolic_mmhg=sys_mean,
        diastolic_mmhg=dia_mean,
        mean_arterial_mmhg=map_mmhg,
        heart_rate_bpm=float(hr_bpm),
        confidence=confidence,
        beat_morphologies=morphologies,
    )
