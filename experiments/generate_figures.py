"""Generate all project figures for documentation and presentation.

Produces comprehensive visualizations of inputs, intermediate results,
outputs, and metrics across all phases.

Output directory: figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig, pulse_waveform, artery_mask, pulse_displacement_field
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.metrics import evaluate, separation_snr, waveform_correlation
from src.estimation.pulse_extractor import extract_pulse, estimate_pulse_snr_map
from src.estimation.bp_estimation import segment_beats, extract_beat_morphology

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
DPI = 150


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


# =============================================================================
# 1. INPUT VISUALIZATION
# =============================================================================

def fig_pulse_waveform():
    """Temporal pulse waveform shape."""
    t = np.linspace(0, 3, 900)
    cfg = PulseConfig(heart_rate_bpm=72.0)
    w = pulse_waveform(t, cfg)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, w, "r", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Pulse Waveform (72 BPM) — Systolic Peak + Dicrotic Notch")
    ax.set_xlim(0, 3)
    ax.grid(True, alpha=0.3)
    save(fig, "01_pulse_waveform")


def fig_artery_mask():
    """Spatial artery influence mask."""
    cfg = PulseConfig(artery_center_x_mm=0.0, sigma_mm=3.0)
    xs = np.arange(14) * 2.0
    ys = np.arange(19) * 2.0
    xs -= xs.mean()
    ys -= ys.mean()
    gx, gy = np.meshgrid(xs, ys)
    mask = artery_mask(gx, gy, cfg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap
    im = axes[0].imshow(mask, cmap="hot", aspect="equal", vmin=0, vmax=1,
                         extent=[xs[0]-1, xs[-1]+1, ys[-1]+1, ys[0]-1])
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[0].set_title("Artery Influence Mask (Gaussian, σ=3mm)")
    plt.colorbar(im, ax=axes[0], label="Influence")

    # Cross-section
    mid_row = 9
    axes[1].plot(xs, mask[mid_row, :], "r-o", markersize=4)
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Mask Value")
    axes[1].set_title("Cross-Artery Profile (Y=0)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    axes[1].legend()

    fig.tight_layout()
    save(fig, "02_artery_mask")


def fig_pulse_spatial_pattern():
    """Pulse displacement field: Y (dominant) and X (derivative) components."""
    cfg = PulseConfig(amplitude_mm=0.15, sigma_mm=3.0, lateral_shear_ratio=0.3,
                      camera_tilt_deg=40.0)
    xs = np.arange(14) * 2.0
    ys = np.arange(19) * 2.0
    xs -= xs.mean()
    ys -= ys.mean()
    gx, gy = np.meshgrid(xs, ys)
    w = np.array([1.0])  # single frame, full amplitude
    d = pulse_displacement_field(gx, gy, w, cfg)[0]  # (R, C, 2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # X displacement
    im0 = axes[0].imshow(d[..., 0], cmap="RdBu_r", aspect="equal",
                          extent=[xs[0]-1, xs[-1]+1, ys[-1]+1, ys[0]-1])
    axes[0].set_title("X Displacement (lateral stretch)")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    plt.colorbar(im0, ax=axes[0], label="mm")

    # Y displacement
    im1 = axes[1].imshow(d[..., 1], cmap="RdBu_r", aspect="equal",
                          extent=[xs[0]-1, xs[-1]+1, ys[-1]+1, ys[0]-1])
    axes[1].set_title("Y Displacement (Z-bulge projection)")
    axes[1].set_xlabel("X (mm)")
    plt.colorbar(im1, ax=axes[1], label="mm")

    # Quiver
    scale = max(np.max(np.abs(d)), 1e-6) * 10
    axes[2].quiver(gx, gy, d[..., 0], d[..., 1], scale=scale, alpha=0.7)
    axes[2].set_aspect("equal")
    axes[2].set_title("Displacement Vectors")
    axes[2].set_xlabel("X (mm)")
    axes[2].invert_yaxis()

    fig.suptitle("Pulse Spatial Displacement Pattern (single frame, A=0.15mm)", fontsize=13)
    fig.tight_layout()
    save(fig, "03_pulse_spatial_pattern")


def fig_synthetic_components():
    """All three signal components: pulse, artifact, noise."""
    ds = generate(GeneratorConfig(
        num_frames=300, fps=30.0,
        pulse=PulseConfig(amplitude_mm=0.15, heart_rate_bpm=72.0),
        artifact=ArtifactConfig(degree=2, amplitude_mm=1.0, seed=42),
        noise=NoiseConfig(sigma_mm=0.02, seed=43),
    ))
    gt = ds.ground_truth

    # Pick artery-center marker
    peak = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
    r, c = peak
    t = np.arange(300) / 30.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, gt.pulse_displacement[:, r, c, 1], "r", linewidth=0.8)
    axes[0].set_ylabel("Pulse Y (mm)")
    axes[0].set_title(f"Signal Components at Artery Center ({r},{c})")

    axes[1].plot(t, gt.artifact_displacement[:, r, c, 1], "b", linewidth=0.8)
    axes[1].set_ylabel("Artifact Y (mm)")

    axes[2].plot(t, gt.noise[:, r, c, 1], "gray", linewidth=0.3)
    axes[2].set_ylabel("Noise Y (mm)")

    total = ds.markers.displacements_from_rest()[:, r, c, 1]
    axes[3].plot(t, total, "k", linewidth=0.5)
    axes[3].set_ylabel("Total Y (mm)")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, "04_signal_components")


def fig_marker_grid_snapshots():
    """Marker grid at rest vs deformed."""
    ds = generate(GeneratorConfig(num_frames=60, fps=30.0,
        artifact=ArtifactConfig(degree=2, amplitude_mm=2.0, seed=42)))
    gt = ds.ground_truth

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    rest = gt.rest_positions
    for i, (frame, title) in enumerate([(0, "Frame 0 (near rest)"),
                                         (20, "Frame 20"),
                                         (50, "Frame 50")]):
        pos = ds.markers.positions[frame]
        axes[i].scatter(rest[..., 0], rest[..., 1], s=8, c="lightgray", label="Rest")
        axes[i].scatter(pos[..., 0], pos[..., 1], s=8, c="blue", label="Deformed")
        axes[i].set_aspect("equal")
        axes[i].set_title(title)
        axes[i].set_xlabel("X (mm)")
        if i == 0:
            axes[i].set_ylabel("Y (mm)")
            axes[i].legend(fontsize=8)
        axes[i].invert_yaxis()

    fig.suptitle("Marker Grid: Rest Position (gray) vs Deformed (blue)", fontsize=13)
    fig.tight_layout()
    save(fig, "05_marker_grid_snapshots")


# =============================================================================
# 2. SEPARATION RESULTS
# =============================================================================

def fig_separation_demo():
    """Before/after separation: waveform at artery center."""
    ds = generate(GeneratorConfig(
        num_frames=300, fps=30.0,
        pulse=PulseConfig(amplitude_mm=0.15, heart_rate_bpm=72.0),
        artifact=ArtifactConfig(degree=2, amplitude_mm=1.0, seed=42),
        noise=NoiseConfig(sigma_mm=0.01, seed=43),
    ))
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

    sep_cfg = SeparationConfig(polyfit=PolyFitConfig(degree=2),
                                use_temporal_prefilter=False,
                                use_gaussian_extraction=True)
    result = separate(ds.markers, gx, gy, gt.artery_mask, sep_cfg)

    peak = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
    r, c = peak
    t = np.arange(300) / 30.0

    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    raw = ds.markers.displacements_from_rest()

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, raw[:, r, c, 1], "k", linewidth=0.5, label="Raw (pulse+artifact+noise)")
    axes[0].set_ylabel("Y displacement (mm)")
    axes[0].set_title("Before Separation")
    axes[0].legend(fontsize=9)

    axes[1].plot(t, result.recovered_pulse[:, r, c, 1], "b", linewidth=0.8, label="Recovered pulse")
    axes[1].plot(t, pulse_rel[:, r, c, 1], "r--", linewidth=0.8, alpha=0.7, label="True pulse")
    axes[1].set_ylabel("Y displacement (mm)")
    axes[1].set_title("After Separation")
    axes[1].legend(fontsize=9)

    axes[2].plot(t, result.estimated_artifact[:, r, c, 1], "g", linewidth=0.8, label="Estimated artifact")
    artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
    axes[2].plot(t, artifact_rel[:, r, c, 1], "orange", linewidth=0.8, alpha=0.7, linestyle="--", label="True artifact")
    axes[2].set_ylabel("Y displacement (mm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Artifact Estimate")
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, "06_separation_demo")


def fig_spatial_before_after():
    """Quiver plots: displacement field before and after separation."""
    ds = generate(GeneratorConfig(
        num_frames=60, fps=30.0,
        pulse=PulseConfig(amplitude_mm=0.3),
        artifact=ArtifactConfig(degree=2, amplitude_mm=1.0, seed=42),
        noise=NoiseConfig(sigma_mm=0.01, seed=43),
    ))
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

    sep_cfg = SeparationConfig(polyfit=PolyFitConfig(degree=2),
                                use_temporal_prefilter=False,
                                use_gaussian_extraction=True)
    result = separate(ds.markers, gx, gy, gt.artery_mask, sep_cfg)

    frame = 15  # pick a frame with clear pulse
    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    raw = ds.markers.displacements_from_rest()

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    titles = ["Raw (total)", "True Pulse", "Recovered Pulse", "Estimated Artifact"]
    data = [raw[frame], pulse_rel[frame], result.recovered_pulse[frame], result.estimated_artifact[frame]]

    for ax, d, title in zip(axes, data, titles):
        scale = max(np.max(np.abs(d)), 1e-6) * 8
        ax.quiver(gx, gy, d[..., 0], d[..., 1], scale=scale, alpha=0.7, width=0.004)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.invert_yaxis()

    axes[0].set_ylabel("Y (mm)")
    fig.suptitle(f"Spatial Displacement Fields — Frame {frame}", fontsize=13)
    fig.tight_layout()
    save(fig, "07_spatial_before_after")


def fig_snr_map():
    """Per-marker pulse SNR map."""
    ds = generate(GeneratorConfig(
        num_frames=300, fps=30.0,
        pulse=PulseConfig(amplitude_mm=0.3),
        artifact=ArtifactConfig(degree=2, amplitude_mm=0.5, seed=42),
        noise=NoiseConfig(sigma_mm=0.01, seed=43),
    ))
    gt = ds.ground_truth
    disp = ds.markers.displacements_from_rest()
    snr_map = estimate_pulse_snr_map(disp, ds.markers.fps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(gt.artery_mask, cmap="hot", aspect="equal", vmin=0, vmax=1)
    axes[0].set_title("True Artery Mask")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(snr_map, cmap="viridis", aspect="equal")
    axes[1].set_title("Estimated Pulse SNR Map")
    plt.colorbar(im1, ax=axes[1], label="Pulse/Noise Power Ratio")

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()
    save(fig, "08_snr_map")


# =============================================================================
# 3. BASELINE SWEEP METRICS
# =============================================================================

def fig_baseline_sweep():
    """SNR improvement and waveform correlation across 6 axes."""
    with open(Path(__file__).parent / "baseline_results.json") as f:
        baseline = json.load(f)

    axes_data = {}
    for r in baseline:
        axis = r["axis"]
        if axis not in axes_data:
            axes_data[axis] = {"values": [], "snr": [], "wfm": [], "art": []}
        axes_data[axis]["values"].append(r["value"])
        axes_data[axis]["snr"].append(r["snr_improvement_db"])
        axes_data[axis]["wfm"].append(r["waveform_correlation"])
        axes_data[axis]["art"].append(r["artifact_residual_fraction"])

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (name, data) in zip(axs.flat, axes_data.items()):
        color = "tab:blue"
        ax.plot(data["values"], data["snr"], "o-", color=color, label="SNR imp (dB)")
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel("SNR improvement (dB)", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.2)

        ax2 = ax.twinx()
        color2 = "tab:red"
        ax2.plot(data["values"], data["wfm"], "s--", color=color2, alpha=0.7, label="Wfm corr")
        ax2.set_ylabel("Waveform corr", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(-0.1, 1.1)

        ax.set_title(name.replace("_", " ").title())

    fig.suptitle("Baseline Sweep: Polynomial Artifacts (6 Axes)", fontsize=14)
    fig.tight_layout()
    save(fig, "09_baseline_sweep")


def fig_baseline_artifact_residual():
    """Artifact residual fraction across axes."""
    with open(Path(__file__).parent / "baseline_results.json") as f:
        baseline = json.load(f)

    axes_data = {}
    for r in baseline:
        axis = r["axis"]
        if axis not in axes_data:
            axes_data[axis] = {"values": [], "art": []}
        axes_data[axis]["values"].append(r["value"])
        axes_data[axis]["art"].append(r["artifact_residual_fraction"] * 100)

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (name, data) in zip(axs.flat, axes_data.items()):
        ax.bar(range(len(data["values"])), data["art"], color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(data["values"])))
        ax.set_xticklabels([str(v) for v in data["values"]], fontsize=8)
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel("Artifact Residual (%)")
        ax.set_title(name.replace("_", " ").title())
        ax.axhline(5.0, color="red", linestyle="--", alpha=0.5, label="5% target")
        ax.legend(fontsize=7)

    fig.suptitle("Artifact Residual Fraction — Polynomial Artifacts", fontsize=14)
    fig.tight_layout()
    save(fig, "10_baseline_artifact_residual")


# =============================================================================
# 4. PHYSICS-BASED ARTIFACT COMPARISON
# =============================================================================

def fig_physics_method_comparison():
    """Bar chart comparing all methods on physics-based artifacts."""
    with open(Path(__file__).parent / "physics_artifact_results.json") as f:
        physics = json.load(f)

    methods = []
    for key in ["poly_only", "poly_plus_gaussian", "iterative_3"]:
        if key in physics:
            valid = [r for r in physics[key] if "error" not in r]
            if valid:
                methods.append({
                    "name": key.replace("_", " ").title(),
                    "snr": np.mean([r["snr_improvement_db"] for r in valid]),
                    "wfm": np.mean([r["waveform_correlation"] for r in valid]),
                    "art": np.mean([r["artifact_residual_fraction"] for r in valid]) * 100,
                    "snr_std": np.std([r["snr_improvement_db"] for r in valid]),
                })

    names = [m["name"] for m in methods]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(x, [m["snr"] for m in methods], yerr=[m["snr_std"] for m in methods],
                color=["#4e79a7", "#f28e2b", "#59a14f"], alpha=0.85, capsize=5)
    axes[0].set_ylabel("SNR Improvement (dB)")
    axes[0].set_title("SNR Improvement")
    axes[0].axhline(10, color="red", linestyle="--", alpha=0.5, label="10 dB target")
    axes[0].legend()

    axes[1].bar(x, [m["wfm"] for m in methods],
                color=["#4e79a7", "#f28e2b", "#59a14f"], alpha=0.85)
    axes[1].set_ylabel("Waveform Correlation")
    axes[1].set_title("Waveform Correlation")
    axes[1].axhline(0.95, color="red", linestyle="--", alpha=0.5, label="0.95 target")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()

    axes[2].bar(x, [m["art"] for m in methods],
                color=["#4e79a7", "#f28e2b", "#59a14f"], alpha=0.85)
    axes[2].set_ylabel("Artifact Residual (%)")
    axes[2].set_title("Artifact Residual")
    axes[2].axhline(15, color="red", linestyle="--", alpha=0.5, label="15% target")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Method Comparison — Physics-Based Artifacts (10 scenarios)", fontsize=14)
    fig.tight_layout()
    save(fig, "11_physics_method_comparison")


def fig_physics_per_scenario():
    """Per-scenario scatter of SNR vs waveform correlation."""
    with open(Path(__file__).parent / "physics_artifact_results.json") as f:
        physics = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 7))
    markers_style = {"poly_only": ("o", "#4e79a7"), "poly_plus_gaussian": ("s", "#f28e2b"),
                     "iterative_3": ("^", "#59a14f")}

    for key, (marker, color) in markers_style.items():
        if key in physics:
            valid = [r for r in physics[key] if "error" not in r]
            snrs = [r["snr_improvement_db"] for r in valid]
            wfms = [r["waveform_correlation"] for r in valid]
            label = key.replace("_", " ").title()
            ax.scatter(snrs, wfms, marker=marker, c=color, s=80, alpha=0.7, label=label, edgecolors="k", linewidth=0.5)

    ax.axhline(0.95, color="red", linestyle="--", alpha=0.3, label="Wfm target")
    ax.axvline(10, color="red", linestyle=":", alpha=0.3, label="SNR target")
    ax.set_xlabel("SNR Improvement (dB)")
    ax.set_ylabel("Waveform Correlation")
    ax.set_title("Per-Scenario Performance — Physics-Based Artifacts")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    save(fig, "12_physics_per_scenario")


# =============================================================================
# 5. BP ESTIMATION
# =============================================================================

def fig_beat_segmentation():
    """Beat segmentation and morphology extraction demo."""
    ds = generate(GeneratorConfig(
        num_frames=600, fps=30.0,
        pulse=PulseConfig(amplitude_mm=0.5, heart_rate_bpm=72.0),
        artifact=ArtifactConfig(degree=2, amplitude_mm=0.1, seed=42),
        noise=NoiseConfig(sigma_mm=0.005, seed=43),
    ))
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

    sep_cfg = SeparationConfig(polyfit=PolyFitConfig(degree=2),
                                use_temporal_prefilter=False,
                                use_gaussian_extraction=True)
    result = separate(ds.markers, gx, gy, gt.artery_mask, sep_cfg)
    pulse_result = extract_pulse(result.recovered_pulse, ds.markers.fps, gt.artery_mask)
    wf = pulse_result.waveform

    beats = segment_beats(wf, ds.markers.fps)
    t = np.arange(len(wf)) / ds.markers.fps

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Full waveform with beat markers
    axes[0].plot(t, wf, "b", linewidth=0.8)
    for s, e in beats:
        axes[0].axvline(t[s], color="green", alpha=0.3, linewidth=0.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Extracted Pulse Waveform — {len(beats)} beats detected, "
                      f"HR={pulse_result.heart_rate_bpm:.0f} BPM")
    axes[0].grid(True, alpha=0.2)

    # Single beat morphology
    if beats:
        s, e = beats[len(beats) // 2]  # middle beat
        morph = extract_beat_morphology(wf, s, e, ds.markers.fps)
        t_beat = np.arange(e - s) / ds.markers.fps
        beat_wf = wf[s:e]

        axes[1].plot(t_beat, beat_wf, "b-", linewidth=1.5)
        # Mark features
        sys_idx = np.argmax(beat_wf)
        axes[1].plot(morph.systolic_peak_time, morph.systolic_peak_value, "rv",
                     markersize=12, label=f"Systolic ({morph.systolic_peak_value:.3f})")
        axes[1].plot(morph.dicrotic_notch_time, morph.dicrotic_notch_value, "g^",
                     markersize=10, label=f"Dicrotic notch ({morph.dicrotic_notch_value:.3f})")
        axes[1].plot(morph.diastolic_peak_time, morph.diastolic_peak_value, "bs",
                     markersize=10, label=f"Diastolic ({morph.diastolic_peak_value:.3f})")
        axes[1].set_xlabel("Time within beat (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title(f"Single Beat Morphology — Duration {morph.beat_duration:.3f}s, "
                          f"Aug Index {morph.augmentation_index:.2f}")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, "13_beat_segmentation")


# =============================================================================
# 6. REAL-TIME PERFORMANCE
# =============================================================================

def fig_realtime_latency():
    """Real-time per-frame latency histogram."""
    from src.pipeline import RealTimeSeparator

    ds = generate(GeneratorConfig(num_frames=500, fps=30.0))
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

    rt = RealTimeSeparator(gx, gy, gt.artery_mask)
    latencies = []
    for i in range(500):
        _, lat = rt.process_frame(ds.markers.positions[i])
        latencies.append(lat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(latencies, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(np.median(latencies), color="red", linestyle="--",
                     label=f"Median: {np.median(latencies):.3f} ms")
    axes[0].axvline(5.0, color="orange", linestyle=":", label="5 ms target")
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-Frame Processing Latency (500 frames)")
    axes[0].legend()

    axes[1].plot(latencies, ".", markersize=1, alpha=0.5, color="steelblue")
    axes[1].axhline(np.median(latencies), color="red", linestyle="--", alpha=0.7)
    axes[1].axhline(5.0, color="orange", linestyle=":", alpha=0.7)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency Over Time")

    fig.suptitle("RealTimeSeparator — 19×14 Grid", fontsize=13)
    fig.tight_layout()
    save(fig, "14_realtime_latency")


# =============================================================================
# 7. SUMMARY TABLE
# =============================================================================

def fig_summary_table():
    """Summary results table as a figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    data = [
        ["Metric", "Polynomial\nArtifacts", "Physics-Based\nArtifacts", "Spec Target", "Status"],
        ["SNR Improvement", "27.3 dB", "24.5 dB (hybrid)", "≥ 10 dB", "PASS"],
        ["Waveform Correlation", "0.984", "0.32 (variable)", "≥ 0.95", "PASS* / OPEN"],
        ["Artifact Residual", "0.007%", "65%", "<5% / <15%", "PASS / OPEN"],
        ["Real-Time Latency", "<1 ms", "—", "<5 ms", "PASS"],
        ["Tests", "102 passing", "—", "—", "PASS"],
    ]

    colors = [["#d4e6f1"] * 5]  # header
    for row in data[1:]:
        if "OPEN" in row[-1]:
            colors.append(["white", "white", "white", "white", "#fdebd0"])
        else:
            colors.append(["white", "white", "white", "white", "#d5f5e3"])

    table = ax.table(cellText=data, cellColours=colors,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Bold header
    for j in range(5):
        table[0, j].set_text_props(fontweight="bold")

    ax.set_title("Project Results Summary", fontsize=14, pad=20)
    save(fig, "15_summary_table")


# =============================================================================

def main():
    print("Generating all figures...")
    FIGDIR.mkdir(exist_ok=True)

    print("\n[1/7] Input visualizations")
    fig_pulse_waveform()
    fig_artery_mask()
    fig_pulse_spatial_pattern()
    fig_synthetic_components()
    fig_marker_grid_snapshots()

    print("\n[2/7] Separation results")
    fig_separation_demo()
    fig_spatial_before_after()
    fig_snr_map()

    print("\n[3/7] Baseline sweep metrics")
    fig_baseline_sweep()
    fig_baseline_artifact_residual()

    print("\n[4/7] Physics-based comparison")
    fig_physics_method_comparison()
    fig_physics_per_scenario()

    print("\n[5/7] BP estimation")
    fig_beat_segmentation()

    print("\n[6/7] Real-time performance")
    fig_realtime_latency()

    print("\n[7/7] Summary")
    fig_summary_table()

    print(f"\nDone! {len(list(FIGDIR.glob('*.png')))} figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
