"""Bridge to the full simulator for generating realistic synthetic data.

Generates marker position time series with physics-based artifacts using
the full synthetic-vitrack-videos simulator, with component-wise ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Add upstream simulator to path
_SIM_PATH = Path("/home/stam/synthetic-vitrack-videos")
if str(_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_SIM_PATH))

import synthetic_video_gen as sim
import dataset_config as dc

from src.data.markers import MarkerTimeSeries, GroundTruth, SyntheticDataset


@dataclass
class SimulatorConfig:
    """Configuration for generating data via the full simulator.

    Attributes:
        num_frames: Number of frames to generate.
        fps: Frame rate.
        seed: Random seed for scenario sampling.
        scenario: Scenario tier ('golden', 'clinical', 'edge', 'broken', or None for random).
    """

    num_frames: int = 150
    fps: float = 30.0
    seed: int = 42
    scenario: str | None = None


def generate_from_simulator(config: SimulatorConfig | None = None) -> SyntheticDataset:
    """Generate a dataset using the full physics simulator.

    Uses the upstream synthetic-vitrack-videos simulator to generate
    marker positions with realistic physics-based artifacts, and
    returns component-wise ground truth for separation evaluation.

    Args:
        config: Simulator configuration.

    Returns:
        SyntheticDataset with realistic data and ground truth decomposition.
    """
    if config is None:
        config = SimulatorConfig()

    rng = np.random.default_rng(config.seed)

    # Sample a scenario configuration from the simulator
    configs = dc.sample_random_config(rng)
    configs.render_config.fps = config.fps
    configs.render_config.duration_sec = config.num_frames / config.fps

    # Get base grid
    base_points, grid_shape = sim.get_base_grid(configs.mech_config)
    rows, cols = grid_shape
    N = rows * cols

    T = config.num_frames
    positions_2d = np.empty((T, rows, cols, 2))
    pulse_disp_all = np.empty((T, rows, cols, 2))
    artifact_disp_all = np.empty((T, rows, cols, 2))
    mechanical_disp_all = np.empty((T, rows, cols, 2))
    visibility_all = np.ones((T, rows, cols))
    pulse_waveform = np.empty(T)

    for frame_idx in range(T):
        t = frame_idx / config.fps

        # Get instantaneous configs (handles time-varying deformations)
        curr_deform, curr_opt, curr_render, curr_sensor = sim.get_instantaneous_configs(
            configs.deformation_config, configs.opt_config,
            configs.render_config, configs.sensor_config, t
        )
        curr_phys = sim.apply_slippage(configs.phys_config, curr_deform, t)

        # Get deformed positions with component decomposition
        final_3d, components = sim.apply_mechanical_deformations(
            base_points, t,
            configs.mech_config, curr_phys, curr_deform,
            configs.render_config.duration_sec,
            return_components=True,
        )

        # Project total 3D positions to 2D
        pts_2d, depth_values, f_px = sim.project_points(
            final_3d, curr_opt, configs.sensor_config
        )

        # Store total 2D positions (in pixels)
        positions_2d[frame_idx] = pts_2d.reshape(rows, cols, 2)

        # For displacement ground truth, we need 2D projections of each component.
        # Project base + component to get the 2D effect of each.
        # Pulse displacement in 2D:
        pulse_3d = base_points + components['pulse']
        pulse_2d, _, _ = sim.project_points(pulse_3d, curr_opt, configs.sensor_config)
        base_2d, _, _ = sim.project_points(base_points, curr_opt, configs.sensor_config)
        pulse_disp_all[frame_idx] = (pulse_2d - base_2d).reshape(rows, cols, 2)

        # For artifact and mechanical, compute displacement from projected positions
        # Artifact: deformation of the grid (twist, shear, pinch, elastic, curl)
        artifact_3d = base_points + components['artifact']
        artifact_2d, _, _ = sim.project_points(artifact_3d, curr_opt, configs.sensor_config)
        artifact_disp_all[frame_idx] = (artifact_2d - base_2d).reshape(rows, cols, 2)

        # Mechanical: balloon inflation
        mech_3d = base_points + components['mechanical']
        mech_2d, _, _ = sim.project_points(mech_3d, curr_opt, configs.sensor_config)
        mechanical_disp_all[frame_idx] = (mech_2d - base_2d).reshape(rows, cols, 2)

        # Pulse waveform scalar
        pulse_waveform[frame_idx] = sim.get_pulse_waveform(t, curr_phys)

    # Rest positions in 2D (first frame base projection)
    curr_opt_0 = sim.get_instantaneous_configs(
        configs.deformation_config, configs.opt_config,
        configs.render_config, configs.sensor_config, 0.0
    )[1]
    base_2d_0, _, _ = sim.project_points(base_points, curr_opt_0, configs.sensor_config)
    rest_positions = base_2d_0.reshape(rows, cols, 2)

    # Artery mask from pulse spatial maps
    from src.synth.pulse import artery_mask as compute_artery_mask, PulseConfig
    # Use the rest grid coordinates (in mm) for artery mask
    grid_x_mm = base_points[:, 0].reshape(rows, cols)
    grid_y_mm = base_points[:, 1].reshape(rows, cols)

    pulse_cfg = PulseConfig(
        artery_center_x_mm=configs.phys_config.artery_offset_mm[0],
        artery_center_y_mm=configs.phys_config.artery_offset_mm[1],
        artery_angle_deg=configs.phys_config.artery_angle_deg,
        sigma_mm=configs.phys_config.pulse_width_sigma,
    )
    mask = compute_artery_mask(grid_x_mm, grid_y_mm, pulse_cfg)

    markers = MarkerTimeSeries(
        positions=positions_2d,
        visibility=visibility_all,
        fps=config.fps,
        grid_spacing_mm=configs.mech_config.grid_spacing,
    )

    ground_truth = GroundTruth(
        pulse_displacement=pulse_disp_all,
        artifact_displacement=artifact_disp_all + mechanical_disp_all,
        noise=np.zeros_like(pulse_disp_all),  # No separate noise in this mode
        artery_mask=mask,
        pulse_waveform=pulse_waveform,
        rest_positions=rest_positions,
    )

    return SyntheticDataset(markers=markers, ground_truth=ground_truth)
