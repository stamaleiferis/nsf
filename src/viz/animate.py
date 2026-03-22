"""Marker displacement animation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from src.data.markers import MarkerTimeSeries, SyntheticDataset


def animate_markers(
    markers: MarkerTimeSeries,
    interval_ms: int = 33,
    trail_frames: int = 0,
) -> FuncAnimation:
    """Animate marker positions over time.

    Args:
        markers: Marker time series to animate.
        interval_ms: Milliseconds between animation frames.
        trail_frames: Number of trailing ghost frames to show.

    Returns:
        matplotlib FuncAnimation object.
    """
    fig, ax = plt.subplots(figsize=(8, 10))

    pos = markers.positions
    vis = markers.visibility

    # Set axis limits from data range
    valid = vis > 0.5
    x_all = pos[valid][..., 0]
    y_all = pos[valid][..., 1]
    margin = 1.0
    ax.set_xlim(np.nanmin(x_all) - margin, np.nanmax(x_all) + margin)
    ax.set_ylim(np.nanmax(y_all) + margin, np.nanmin(y_all) - margin)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    scatter = ax.scatter([], [], s=10, c="blue")
    title = ax.set_title("")

    def update(frame: int):
        mask = vis[frame] > 0.5
        p = pos[frame]
        scatter.set_offsets(p[mask].reshape(-1, 2))
        title.set_text(f"Frame {frame}/{markers.num_frames}  t={frame/markers.fps:.2f}s")
        return scatter, title

    anim = FuncAnimation(
        fig, update, frames=markers.num_frames,
        interval=interval_ms, blit=True,
    )
    return anim


def animate_components(
    dataset: SyntheticDataset,
    interval_ms: int = 33,
) -> FuncAnimation:
    """Animate total, pulse, and artifact displacements side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    gt = dataset.ground_truth
    rest = gt.rest_positions  # (R, C, 2)

    # Compute displacement ranges for consistent scaling
    max_disp = max(
        np.max(np.abs(gt.pulse_displacement)),
        np.max(np.abs(gt.artifact_displacement)),
        1e-6,
    )

    scatters = []
    for ax, label in zip(axes, ["Total", "Pulse", "Artifact"]):
        ax.set_xlim(rest[..., 0].min() - 2, rest[..., 0].max() + 2)
        ax.set_ylim(rest[..., 1].max() + 2, rest[..., 1].min() - 2)
        ax.set_aspect("equal")
        ax.set_title(label)
        s = ax.scatter([], [], s=8, c="blue")
        scatters.append(s)

    title = fig.suptitle("")

    def update(frame: int):
        total_pos = dataset.markers.positions[frame]
        pulse_pos = rest + gt.pulse_displacement[frame]
        artifact_pos = rest + gt.artifact_displacement[frame]

        for s, p in zip(scatters, [total_pos, pulse_pos, artifact_pos]):
            s.set_offsets(p.reshape(-1, 2))

        title.set_text(f"Frame {frame}  t={frame/dataset.markers.fps:.2f}s")
        return (*scatters, title)

    anim = FuncAnimation(
        fig, update, frames=dataset.markers.num_frames,
        interval=interval_ms, blit=True,
    )
    return anim
