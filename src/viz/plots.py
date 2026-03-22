"""Static diagnostic plots for marker data and signal separation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.data.markers import MarkerTimeSeries, SyntheticDataset


def plot_marker_grid(
    markers: MarkerTimeSeries,
    frame: int = 0,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot marker positions at a single frame."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    pos = markers.positions[frame]  # (R, C, 2)
    vis = markers.visibility[frame]  # (R, C)

    # Visible markers
    mask = vis > 0.5
    ax.scatter(pos[mask, 0], pos[mask, 1], s=10, c="blue", alpha=0.7)

    # Invisible markers
    if (~mask).any():
        inv_pos = pos[~mask]
        valid = ~np.isnan(inv_pos[:, 0])
        if valid.any():
            ax.scatter(
                inv_pos[valid, 0], inv_pos[valid, 1],
                s=10, c="red", alpha=0.3, marker="x",
            )

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Marker grid — frame {frame}")
    ax.invert_yaxis()
    return fig


def plot_displacement_timeseries(
    dataset: SyntheticDataset,
    row: int | None = None,
    col: int | None = None,
    axis: int = 1,
) -> Figure:
    """Plot displacement time series for a single marker, showing components.

    If row/col not specified, picks the marker closest to artery center.
    axis: 0=X, 1=Y.
    """
    gt = dataset.ground_truth
    if row is None or col is None:
        # Pick marker with highest artery mask value
        idx = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
        row, col = idx

    T = dataset.markers.num_frames
    t = np.arange(T) / dataset.markers.fps
    axis_label = "X" if axis == 0 else "Y"

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Total displacement
    total = dataset.markers.displacements_from_rest()[: , row, col, axis]
    axes[0].plot(t, total, "k", linewidth=0.5)
    axes[0].set_ylabel(f"Total {axis_label} (mm)")

    # Pulse
    pulse = gt.pulse_displacement[:, row, col, axis]
    axes[1].plot(t, pulse, "r", linewidth=0.8)
    axes[1].set_ylabel(f"Pulse {axis_label} (mm)")

    # Artifact
    artifact = gt.artifact_displacement[:, row, col, axis]
    axes[2].plot(t, artifact, "b", linewidth=0.8)
    axes[2].set_ylabel(f"Artifact {axis_label} (mm)")

    # Noise
    noise = gt.noise[:, row, col, axis]
    axes[3].plot(t, noise, "gray", linewidth=0.3)
    axes[3].set_ylabel(f"Noise {axis_label} (mm)")
    axes[3].set_xlabel("Time (s)")

    fig.suptitle(f"Marker ({row}, {col}) — {axis_label} displacement components")
    fig.tight_layout()
    return fig


def plot_artery_mask(dataset: SyntheticDataset, ax: plt.Axes | None = None) -> Figure:
    """Plot the artery influence mask as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    mask = dataset.ground_truth.artery_mask
    im = ax.imshow(mask, cmap="hot", aspect="equal", vmin=0, vmax=1)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Artery influence mask")
    plt.colorbar(im, ax=ax)
    return fig


def plot_spatial_snapshot(
    dataset: SyntheticDataset,
    frame: int = 0,
    component: str = "pulse",
) -> Figure:
    """Plot displacement field at a single frame as quiver plot."""
    gt = dataset.ground_truth
    rest = gt.rest_positions  # (R, C, 2)

    if component == "pulse":
        disp = gt.pulse_displacement[frame]
        title = f"Pulse displacement — frame {frame}"
    elif component == "artifact":
        disp = gt.artifact_displacement[frame]
        title = f"Artifact displacement — frame {frame}"
    else:
        disp = (
            dataset.markers.positions[frame]
            - gt.rest_positions
        )
        title = f"Total displacement — frame {frame}"

    fig, ax = plt.subplots(figsize=(8, 10))
    scale = np.max(np.abs(disp)) or 1.0
    ax.quiver(
        rest[..., 0], rest[..., 1],
        disp[..., 0], disp[..., 1],
        scale=scale * 10, alpha=0.7,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    ax.invert_yaxis()
    return fig
