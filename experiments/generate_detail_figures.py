"""Generate detailed component-level figures showing ground truth vs estimated.

For representative markers: all GT components and all estimated components,
as 1D time series and 2D spatial snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
DPI = 150


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


def make_data():
    """Generate dataset and run separation."""
    ds = generate(GeneratorConfig(
        num_frames=300, fps=30.0,
        pulse=PulseConfig(amplitude_mm=0.15, heart_rate_bpm=72.0, sigma_mm=3.0,
                          sigma_along_mm=8.0),
        artifact=ArtifactConfig(degree=2, amplitude_mm=1.0, seed=42),
        noise=NoiseConfig(sigma_mm=0.02, seed=43),
    ))
    gt = ds.ground_truth
    gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

    sep_cfg = SeparationConfig(
        polyfit=PolyFitConfig(degree=2),
        use_temporal_prefilter=False,
        use_gaussian_extraction=True,
    )
    result = separate(ds.markers, gx, gy, gt.artery_mask, sep_cfg)

    return ds, result


def pick_markers(ds):
    """Pick representative markers: artery center, artery edge, off-artery."""
    mask = ds.ground_truth.artery_mask
    # Artery center: highest mask value
    center = np.unravel_index(np.argmax(mask), mask.shape)

    # Artery edge: mask ~ 0.3-0.5
    edge_candidates = np.argwhere((mask > 0.2) & (mask < 0.5))
    if len(edge_candidates) > 0:
        # Pick one near center row
        dists = np.abs(edge_candidates[:, 0] - center[0])
        edge = tuple(edge_candidates[np.argmin(dists)])
    else:
        edge = (center[0], min(center[1] + 3, mask.shape[1] - 1))

    # Off-artery: lowest mask value, same row as center
    row = center[0]
    off_col = np.argmin(mask[row, :])
    off = (row, off_col)

    # Second off-artery: corner
    corner = (0, 0)

    return {
        "Artery center": center,
        "Artery edge": edge,
        "Off-artery": off,
        "Corner": corner,
    }


# =============================================================================
# FIGURE: 1D time series — GT components for representative markers
# =============================================================================

def fig_gt_components_1d(ds, markers_dict):
    """Ground truth components as time series for each marker."""
    gt = ds.ground_truth
    t = np.arange(ds.markers.num_frames) / ds.markers.fps

    n_markers = len(markers_dict)
    fig, axes = plt.subplots(n_markers, 4, figsize=(20, 3.5 * n_markers), sharex=True)
    if n_markers == 1:
        axes = axes.reshape(1, -1)

    components = [
        ("Pulse", gt.pulse_displacement, "tab:red"),
        ("Artifact", gt.artifact_displacement, "tab:blue"),
        ("Noise", gt.noise, "tab:gray"),
        ("Total", gt.pulse_displacement + gt.artifact_displacement + gt.noise, "black"),
    ]

    for row, (label, (r, c)) in enumerate(markers_dict.items()):
        mask_val = gt.artery_mask[r, c]
        for col, (comp_name, comp_data, color) in enumerate(components):
            ax = axes[row, col]
            # Plot both X and Y
            ax.plot(t, comp_data[:, r, c, 0], color=color, alpha=0.5, linewidth=0.6, label="X")
            ax.plot(t, comp_data[:, r, c, 1], color=color, linewidth=0.8, label="Y")
            ax.grid(True, alpha=0.15)
            if row == 0:
                ax.set_title(comp_name, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\n({r},{c}) mask={mask_val:.2f}\n\nDisp (mm)", fontsize=9)
            if row == n_markers - 1:
                ax.set_xlabel("Time (s)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Ground Truth Components — 1D Time Series (X and Y displacement)", fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "20_gt_components_1d")


# =============================================================================
# FIGURE: 1D time series — Estimated components for representative markers
# =============================================================================

def fig_estimated_components_1d(ds, result, markers_dict):
    """Estimated vs GT components as time series for each marker."""
    gt = ds.ground_truth
    t = np.arange(ds.markers.num_frames) / ds.markers.fps

    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
    raw = ds.markers.displacements_from_rest()

    n_markers = len(markers_dict)
    fig, axes = plt.subplots(n_markers, 3, figsize=(18, 3.5 * n_markers), sharex=True)
    if n_markers == 1:
        axes = axes.reshape(1, -1)

    for row, (label, (r, c)) in enumerate(markers_dict.items()):
        mask_val = gt.artery_mask[r, c]

        # Column 1: Raw signal
        ax = axes[row, 0]
        ax.plot(t, raw[:, r, c, 1], "k", linewidth=0.5, alpha=0.7)
        ax.set_ylabel(f"{label}\n({r},{c}) mask={mask_val:.2f}\n\nY disp (mm)", fontsize=9)
        if row == 0:
            ax.set_title("Raw (observed)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.15)

        # Column 2: Estimated artifact vs GT artifact
        ax = axes[row, 1]
        ax.plot(t, artifact_rel[:, r, c, 1], "tab:blue", linewidth=0.8, alpha=0.6, label="GT artifact")
        ax.plot(t, result.estimated_artifact[:, r, c, 1], "tab:orange", linewidth=0.8,
                linestyle="--", label="Estimated artifact")
        if row == 0:
            ax.set_title("Artifact: GT vs Estimated", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

        # Column 3: Recovered pulse vs GT pulse
        ax = axes[row, 2]
        ax.plot(t, pulse_rel[:, r, c, 1], "tab:red", linewidth=0.8, alpha=0.6, label="GT pulse")
        ax.plot(t, result.recovered_pulse[:, r, c, 1], "tab:green", linewidth=0.8,
                linestyle="--", label="Recovered pulse")
        if row == 0:
            ax.set_title("Pulse: GT vs Recovered", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

        if row == n_markers - 1:
            for ax in axes[row]:
                ax.set_xlabel("Time (s)")

    fig.suptitle("Separation Results — GT vs Estimated (Y displacement)", fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "21_estimated_vs_gt_1d")


# =============================================================================
# FIGURE: 1D — Zoomed pulse comparison (2 beats)
# =============================================================================

def fig_pulse_zoom_1d(ds, result, markers_dict):
    """Zoomed view of recovered vs GT pulse for ~2 beats."""
    gt = ds.ground_truth
    fps = ds.markers.fps
    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]

    # Zoom to frames 60-110 (~2 beats at 72 BPM)
    f_start, f_end = 60, 110
    t = np.arange(f_start, f_end) / fps

    n_markers = len(markers_dict)
    fig, axes = plt.subplots(1, n_markers, figsize=(5 * n_markers, 4), sharey=False)
    if n_markers == 1:
        axes = [axes]

    for i, (label, (r, c)) in enumerate(markers_dict.items()):
        ax = axes[i]
        ax.plot(t, pulse_rel[f_start:f_end, r, c, 1], "tab:red", linewidth=1.5,
                label="GT pulse", alpha=0.7)
        ax.plot(t, result.recovered_pulse[f_start:f_end, r, c, 1], "tab:green",
                linewidth=1.5, linestyle="--", label="Recovered")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{label} ({r},{c})\nmask={gt.artery_mask[r, c]:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.set_ylabel("Y displacement (mm)")

    fig.suptitle("Pulse Waveform Zoom (~2 beats) — GT vs Recovered", fontsize=13)
    fig.tight_layout()
    save(fig, "22_pulse_zoom_1d")


# =============================================================================
# FIGURE: 2D spatial snapshots — GT components at one frame
# =============================================================================

def fig_gt_components_2d(ds, frame=50):
    """2D spatial snapshots of each GT component at a single frame."""
    gt = ds.ground_truth
    rest = gt.rest_positions
    gx, gy = rest[..., 0], rest[..., 1]

    components = [
        ("Pulse", gt.pulse_displacement[frame]),
        ("Artifact", gt.artifact_displacement[frame]),
        ("Noise", gt.noise[frame]),
        ("Total", gt.pulse_displacement[frame] + gt.artifact_displacement[frame] + gt.noise[frame]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, (name, disp) in enumerate(components):
        # Top row: Y displacement heatmap
        ax = axes[0, col]
        im = ax.imshow(disp[..., 1], cmap="RdBu_r", aspect="equal",
                        extent=[gx[0, 0] - 1, gx[0, -1] + 1, gy[-1, 0] + 1, gy[0, 0] - 1])
        ax.set_title(f"{name} — Y displacement", fontsize=11)
        plt.colorbar(im, ax=ax, label="mm", shrink=0.8)
        if col == 0:
            ax.set_ylabel("Y (mm)")

        # Bottom row: quiver plot
        ax = axes[1, col]
        scale = max(np.max(np.abs(disp)), 1e-6) * 8
        ax.quiver(gx, gy, disp[..., 0], disp[..., 1], scale=scale, alpha=0.7, width=0.004)
        ax.set_aspect("equal")
        ax.set_title(f"{name} — vectors", fontsize=11)
        ax.set_xlabel("X (mm)")
        ax.invert_yaxis()
        if col == 0:
            ax.set_ylabel("Y (mm)")

    fig.suptitle(f"Ground Truth Components — 2D Spatial (frame {frame})", fontsize=14)
    fig.tight_layout()
    save(fig, "23_gt_components_2d")


# =============================================================================
# FIGURE: 2D spatial snapshots — Estimated vs GT at one frame
# =============================================================================

def fig_estimated_components_2d(ds, result, frame=50):
    """2D spatial comparison: GT vs estimated for pulse and artifact."""
    gt = ds.ground_truth
    rest = gt.rest_positions
    gx, gy = rest[..., 0], rest[..., 1]

    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]

    panels = [
        ("GT Pulse", pulse_rel[frame]),
        ("Recovered Pulse", result.recovered_pulse[frame]),
        ("Pulse Error", result.recovered_pulse[frame] - pulse_rel[frame]),
        ("GT Artifact", artifact_rel[frame]),
        ("Estimated Artifact", result.estimated_artifact[frame]),
        ("Artifact Error", result.estimated_artifact[frame] - artifact_rel[frame]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (name, disp) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        # Use consistent color scale per row
        if "Error" in name:
            vmax = max(np.max(np.abs(disp[..., 1])), 1e-6)
            im = ax.imshow(disp[..., 1], cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax,
                            extent=[gx[0, 0] - 1, gx[0, -1] + 1, gy[-1, 0] + 1, gy[0, 0] - 1])
        else:
            im = ax.imshow(disp[..., 1], cmap="RdBu_r", aspect="equal",
                            extent=[gx[0, 0] - 1, gx[0, -1] + 1, gy[-1, 0] + 1, gy[0, 0] - 1])

        ax.set_title(name, fontsize=12, fontweight="bold" if "Error" not in name else "normal")
        plt.colorbar(im, ax=ax, label="Y disp (mm)", shrink=0.8)
        if col == 0:
            ax.set_ylabel("Y (mm)")
        ax.set_xlabel("X (mm)")

    fig.suptitle(f"Separation Results — 2D Spatial (frame {frame}, Y displacement)", fontsize=14)
    fig.tight_layout()
    save(fig, "24_estimated_vs_gt_2d")


# =============================================================================
# FIGURE: 2D quiver — Estimated vs GT at one frame
# =============================================================================

def fig_estimated_quiver_2d(ds, result, frame=50):
    """2D quiver comparison: GT vs estimated vectors."""
    gt = ds.ground_truth
    rest = gt.rest_positions
    gx, gy = rest[..., 0], rest[..., 1]

    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]

    panels = [
        ("Raw (total)", ds.markers.displacements_from_rest()[frame]),
        ("GT Pulse", pulse_rel[frame]),
        ("Recovered Pulse", result.recovered_pulse[frame]),
        ("GT Artifact", artifact_rel[frame]),
        ("Estimated Artifact", result.estimated_artifact[frame]),
        ("Artery Mask", None),  # special case
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (name, disp) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        if name == "Artery Mask":
            im = ax.imshow(gt.artery_mask, cmap="hot", aspect="equal", vmin=0, vmax=1,
                            extent=[gx[0, 0] - 1, gx[0, -1] + 1, gy[-1, 0] + 1, gy[0, 0] - 1])
            ax.set_title("Artery Mask", fontsize=12)
            plt.colorbar(im, ax=ax, label="Influence", shrink=0.8)
        else:
            scale = max(np.max(np.abs(disp)), 1e-6) * 8
            ax.quiver(gx, gy, disp[..., 0], disp[..., 1], scale=scale, alpha=0.7, width=0.004)
            ax.set_aspect("equal")
            ax.set_title(name, fontsize=12)
            ax.invert_yaxis()

        ax.set_xlabel("X (mm)")
        if col == 0:
            ax.set_ylabel("Y (mm)")

    fig.suptitle(f"Vector Fields — GT vs Estimated (frame {frame})", fontsize=14)
    fig.tight_layout()
    save(fig, "25_estimated_quiver_2d")


# =============================================================================
# FIGURE: Multi-frame 2D evolution
# =============================================================================

def fig_temporal_evolution_2d(ds, result):
    """Show pulse spatial pattern evolving over 1 cardiac cycle."""
    gt = ds.ground_truth
    rest = gt.rest_positions
    gx, gy = rest[..., 0], rest[..., 1]
    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]

    # Pick 6 frames spanning ~1 beat (72 BPM = 0.83s = 25 frames)
    beat_start = 60
    frames = [beat_start + i * 5 for i in range(6)]

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    for col, frame in enumerate(frames):
        t_sec = frame / ds.markers.fps

        # Top: GT pulse
        ax = axes[0, col]
        im = ax.imshow(pulse_rel[frame, ..., 1], cmap="RdBu_r", aspect="equal",
                        extent=[gx[0, 0] - 1, gx[0, -1] + 1, gy[-1, 0] + 1, gy[0, 0] - 1],
                        vmin=-0.1, vmax=0.1)
        ax.set_title(f"t={t_sec:.2f}s", fontsize=10)
        if col == 0:
            ax.set_ylabel("GT Pulse\nY (mm)")

        # Bottom: Recovered pulse
        ax = axes[1, col]
        im = ax.imshow(result.recovered_pulse[frame, ..., 1], cmap="RdBu_r", aspect="equal",
                        extent=[gx[0, 0] - 1, gx[0, -1] + 1, gy[-1, 0] + 1, gy[0, 0] - 1],
                        vmin=-0.1, vmax=0.1)
        if col == 0:
            ax.set_ylabel("Recovered\nY (mm)")
        ax.set_xlabel("X (mm)")

    fig.suptitle("Pulse Spatial Evolution Over 1 Cardiac Cycle — GT (top) vs Recovered (bottom)", fontsize=13)
    fig.tight_layout()
    save(fig, "26_temporal_evolution_2d")


def main():
    print("Generating detailed component figures...")
    FIGDIR.mkdir(exist_ok=True)

    print("\nGenerating data and running separation...")
    ds, result = make_data()
    markers_dict = pick_markers(ds)

    print(f"\nSelected markers:")
    for label, (r, c) in markers_dict.items():
        print(f"  {label}: ({r},{c}) mask={ds.ground_truth.artery_mask[r, c]:.3f}")

    print("\n[1/6] GT components — 1D time series")
    fig_gt_components_1d(ds, markers_dict)

    print("[2/6] Estimated vs GT — 1D time series")
    fig_estimated_components_1d(ds, result, markers_dict)

    print("[3/6] Pulse zoom — 1D")
    fig_pulse_zoom_1d(ds, result, markers_dict)

    print("[4/6] GT components — 2D spatial")
    fig_gt_components_2d(ds, frame=50)

    print("[5/6] Estimated vs GT — 2D heatmap")
    fig_estimated_components_2d(ds, result, frame=50)

    print("[6/6] Estimated vs GT — 2D quiver + temporal evolution")
    fig_estimated_quiver_2d(ds, result, frame=50)
    fig_temporal_evolution_2d(ds, result)

    print(f"\nDone! Detail figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
