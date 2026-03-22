"""Phase 3: Evaluate separation on physics-based artifacts from the full simulator.

Compares separation quality between polynomial (Phase 2) and physics-based artifacts.
Tests individual artifact types and combined scenarios.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Also need the upstream simulator
_SIM_PATH = Path("/home/stam/synthetic-vitrack-videos")
sys.path.insert(0, str(_SIM_PATH))

import synthetic_video_gen as sim
import dataset_config as dc

from src.data.markers import MarkerTimeSeries, GroundTruth, SyntheticDataset
from src.data.simulator_bridge import SimulatorConfig, generate_from_simulator
from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.metrics import evaluate, separation_snr


def run_separation_on_simulator_data(
    seed: int = 42,
    num_frames: int = 150,
    poly_degree: int = 2,
    use_gaussian: bool = False,
) -> dict:
    """Generate data from the simulator and run separation."""
    cfg = SimulatorConfig(num_frames=num_frames, fps=30.0, seed=seed)
    ds = generate_from_simulator(cfg)
    gt = ds.ground_truth

    # Use rest positions in mm for the polynomial fit grid
    # Since simulator gives us pixel coords, we need the mm grid
    base_points, grid_shape = sim.get_base_grid(sim.MechanicalConfig())
    grid_x_mm = base_points[:, 0].reshape(grid_shape)
    grid_y_mm = base_points[:, 1].reshape(grid_shape)

    sep_cfg = SeparationConfig(
        polyfit=PolyFitConfig(degree=poly_degree),
        use_temporal_prefilter=False,
        use_gaussian_extraction=use_gaussian,
    )

    result = separate(ds.markers, grid_x_mm, grid_y_mm, gt.artery_mask, sep_cfg)

    # Relative ground truth
    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]

    m = evaluate(
        result.recovered_pulse,
        result.estimated_artifact,
        pulse_rel,
        artifact_rel,
        gt.artery_mask,
    )

    raw_snr = separation_snr(
        ds.markers.displacements_from_rest(), pulse_rel, artifact_rel,
    )

    return {
        "separation_snr_db": float(m.separation_snr_db),
        "snr_improvement_db": float(m.separation_snr_db - raw_snr),
        "waveform_correlation": float(m.waveform_correlation),
        "spatial_correlation": float(m.spatial_correlation),
        "artifact_residual_fraction": float(m.artifact_residual_fraction),
        "raw_snr_db": float(raw_snr),
        "pulse_rms": float(np.sqrt(np.mean(pulse_rel ** 2))),
        "artifact_rms": float(np.sqrt(np.mean(artifact_rel ** 2))),
    }


def main():
    results = {}

    # 1. Polynomial-only baseline
    print("=== Polynomial Only (no Gaussian extraction) ===")
    poly_results = []
    for seed in range(10):
        print(f"  Seed {seed}...", end=" ", flush=True)
        try:
            r = run_separation_on_simulator_data(seed=seed, num_frames=150, use_gaussian=False)
            r["seed"] = seed
            poly_results.append(r)
            print(f"SNR_imp={r['snr_improvement_db']:.1f}dB  "
                  f"wfm={r['waveform_correlation']:.3f}  "
                  f"art_res={r['artifact_residual_fraction']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            poly_results.append({"error": str(e), "seed": seed})
    results["poly_only"] = poly_results

    # 2. Polynomial + Gaussian extraction
    print("\n=== Polynomial + Gaussian Extraction ===")
    gauss_results = []
    for seed in range(10):
        print(f"  Seed {seed}...", end=" ", flush=True)
        try:
            r = run_separation_on_simulator_data(seed=seed, num_frames=150, use_gaussian=True)
            r["seed"] = seed
            gauss_results.append(r)
            print(f"SNR_imp={r['snr_improvement_db']:.1f}dB  "
                  f"wfm={r['waveform_correlation']:.3f}  "
                  f"art_res={r['artifact_residual_fraction']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            gauss_results.append({"error": str(e), "seed": seed})
    results["poly_plus_gaussian"] = gauss_results

    # Summary comparison
    for label, res_list in [("Poly only", poly_results), ("Poly+Gaussian", gauss_results)]:
        valid = [r for r in res_list if "error" not in r]
        if valid:
            print(f"\n  {label} Summary ({len(valid)} successful):")
            for key in ["snr_improvement_db", "waveform_correlation", "artifact_residual_fraction"]:
                vals = [r[key] for r in valid]
                print(f"    {key}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
                      f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    # 3. Polynomial degree sweep with Gaussian extraction
    print("\n=== Degree Sweep (with Gaussian extraction) ===")
    degree_results = []
    for degree in [1, 2, 3]:
        print(f"  Degree {degree}...", end=" ", flush=True)
        try:
            r = run_separation_on_simulator_data(seed=0, num_frames=150, poly_degree=degree, use_gaussian=True)
            r["degree"] = degree
            degree_results.append(r)
            print(f"SNR_imp={r['snr_improvement_db']:.1f}dB  "
                  f"wfm={r['waveform_correlation']:.3f}  "
                  f"art_res={r['artifact_residual_fraction']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            degree_results.append({"error": str(e), "degree": degree})
    results["degree_sweep_gaussian"] = degree_results

    # Save results
    out_path = Path(__file__).parent / "physics_artifact_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
