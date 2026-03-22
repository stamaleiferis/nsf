"""Load marker data from the full simulator (synthetic-vitrack-videos) output."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from src.data.markers import MarkerTimeSeries


def load_from_simulator(clip_dir: str | Path) -> MarkerTimeSeries:
    """Load marker time series from a synthetic-vitrack-videos clip directory.

    The simulator outputs per-frame .npy files with shape (R, C, 3) where
    channels are [x_pixel_norm, y_pixel_norm, visibility]. Invisible markers
    have coordinate values of ~-1.5 (sentinel).

    Args:
        clip_dir: Path to a clip directory (e.g., dataset/train/clip_0001/).

    Returns:
        MarkerTimeSeries with positions in normalized pixel coordinates.
    """
    clip_dir = Path(clip_dir)
    metadata_path = clip_dir / "metadata.json"

    # Load metadata for fps and grid shape
    fps = 30.0
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        fps = meta.get("fps", 30.0)

    # Find and sort frame files
    frame_files = sorted(clip_dir.glob("frame_*.npy"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.npy files found in {clip_dir}")

    # Load all frames
    frames = [np.load(f) for f in frame_files]
    data = np.stack(frames, axis=0)  # (T, R, C, 3)

    positions = data[..., :2]      # (T, R, C, 2) — normalized pixel coords
    visibility = data[..., 2]      # (T, R, C) — 0 or 1

    # Replace sentinel coordinates with NaN for invisible markers
    invisible = visibility < 0.5
    positions[invisible] = np.nan

    return MarkerTimeSeries(
        positions=positions,
        visibility=visibility,
        fps=fps,
        grid_spacing_mm=2.0,
    )
