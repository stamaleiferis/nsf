"""Baseline validation: sweep separation performance across key axes.

Generates synthetic data with controlled variation and evaluates
the polynomial separation algorithm across each axis.

Axes from SPEC.md §5.3:
- Pulse SNR (amplitude ratio)
- Artifact magnitude
- Polynomial degree (artifact vs fit)
- Heart rate
- Noise level
- Artery map error (offset)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synth.generator import GeneratorConfig, generate
from src.synth.pulse import PulseConfig
from src.synth.artifact import ArtifactConfig
from src.synth.noise import NoiseConfig
from src.separation.separator import SeparationConfig, separate
from src.separation.polynomial_fit import PolyFitConfig
from src.separation.metrics import evaluate


def run_single(
    gen_cfg: GeneratorConfig,
    sep_cfg: SeparationConfig,
    artery_offset_mm: float = 0.0,
) -> dict:
    """Run one experiment: generate data, separate, evaluate."""
    ds = generate(gen_cfg)
    gt = ds.ground_truth

    # Optionally shift the artery mask to simulate artery map error
    if artery_offset_mm != 0.0:
        from src.synth.pulse import artery_mask, PulseConfig
        shifted_cfg = PulseConfig(
            artery_center_x_mm=gen_cfg.pulse.artery_center_x_mm + artery_offset_mm,
            artery_center_y_mm=gen_cfg.pulse.artery_center_y_mm,
            sigma_mm=gen_cfg.pulse.sigma_mm,
        )
        gx = gt.rest_positions[..., 0]
        gy = gt.rest_positions[..., 1]
        mask = artery_mask(gx, gy, shifted_cfg)
    else:
        mask = gt.artery_mask

    result = separate(
        ds.markers,
        gt.rest_positions[..., 0],
        gt.rest_positions[..., 1],
        mask,
        sep_cfg,
    )

    # Use relative ground truth (matching separator's frame-0 reference)
    pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
    artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]

    m = evaluate(
        result.recovered_pulse,
        result.estimated_artifact,
        pulse_rel,
        artifact_rel,
        gt.artery_mask,
    )

    # Also compute raw SNR for comparison
    from src.separation.metrics import separation_snr
    raw_snr = separation_snr(
        ds.markers.displacements_from_rest(), pulse_rel, artifact_rel,
    )

    return {
        "separation_snr_db": m.separation_snr_db,
        "snr_improvement_db": m.separation_snr_db - raw_snr,
        "waveform_correlation": m.waveform_correlation,
        "spatial_correlation": m.spatial_correlation,
        "artifact_residual_fraction": m.artifact_residual_fraction,
    }


def sweep_axis(name: str, values: list, make_configs) -> list[dict]:
    """Sweep one axis and collect metrics."""
    results = []
    for val in values:
        gen_cfg, sep_cfg, offset = make_configs(val)
        metrics = run_single(gen_cfg, sep_cfg, offset)
        metrics["axis"] = name
        metrics["value"] = val
        results.append(metrics)
        print(f"  {name}={val}: SNR_imp={metrics['snr_improvement_db']:.1f}dB  "
              f"wfm_corr={metrics['waveform_correlation']:.3f}  "
              f"art_resid={metrics['artifact_residual_fraction']:.5f}")
    return results


def base_gen_cfg(**overrides) -> GeneratorConfig:
    pulse_kw = {}
    artifact_kw = {"seed": 42}
    noise_kw = {"seed": 43}
    gen_kw = {"num_frames": 300, "fps": 30.0}

    for k, v in overrides.items():
        if k.startswith("pulse_"):
            pulse_kw[k[6:]] = v
        elif k.startswith("artifact_"):
            artifact_kw[k[9:]] = v
        elif k.startswith("noise_"):
            noise_kw[k[6:]] = v
        else:
            gen_kw[k] = v

    return GeneratorConfig(
        pulse=PulseConfig(**pulse_kw),
        artifact=ArtifactConfig(**artifact_kw),
        noise=NoiseConfig(**noise_kw),
        **gen_kw,
    )


def main():
    sep_cfg = SeparationConfig(
        polyfit=PolyFitConfig(degree=2),
        use_temporal_prefilter=False,
    )
    all_results = []

    # 1. Pulse amplitude sweep
    print("=== Pulse Amplitude ===")
    pulse_amps = [0.01, 0.05, 0.1, 0.15, 0.3, 0.5]
    all_results.extend(sweep_axis("pulse_amplitude_mm", pulse_amps, lambda v: (
        base_gen_cfg(pulse_amplitude_mm=v), sep_cfg, 0.0
    )))

    # 2. Artifact magnitude sweep
    print("\n=== Artifact Magnitude ===")
    art_amps = [0.1, 0.5, 1.0, 2.0, 5.0]
    all_results.extend(sweep_axis("artifact_amplitude_mm", art_amps, lambda v: (
        base_gen_cfg(artifact_amplitude_mm=v), sep_cfg, 0.0
    )))

    # 3. Polynomial degree mismatch (artifact degree vs fit degree)
    print("\n=== Artifact Degree (fit degree=2) ===")
    for deg in [1, 2, 3]:
        gen_cfg = base_gen_cfg(artifact_degree=deg)
        metrics = run_single(gen_cfg, sep_cfg)
        metrics["axis"] = "artifact_degree"
        metrics["value"] = deg
        all_results.append(metrics)
        print(f"  artifact_degree={deg}: SNR_imp={metrics['snr_improvement_db']:.1f}dB  "
              f"wfm_corr={metrics['waveform_correlation']:.3f}  "
              f"art_resid={metrics['artifact_residual_fraction']:.5f}")

    # 4. Heart rate sweep
    print("\n=== Heart Rate ===")
    heart_rates = [50, 72, 100, 130]
    all_results.extend(sweep_axis("heart_rate_bpm", heart_rates, lambda v: (
        base_gen_cfg(pulse_heart_rate_bpm=v), sep_cfg, 0.0
    )))

    # 5. Noise level sweep
    print("\n=== Noise Level ===")
    noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
    all_results.extend(sweep_axis("noise_sigma_mm", noise_levels, lambda v: (
        base_gen_cfg(noise_sigma_mm=v), sep_cfg, 0.0
    )))

    # 6. Artery map offset error
    print("\n=== Artery Map Offset ===")
    offsets = [0.0, 1.0, 2.0, 3.0, 5.0]
    all_results.extend(sweep_axis("artery_offset_mm", offsets, lambda v: (
        base_gen_cfg(), sep_cfg, v
    )))

    # Save results
    out_path = Path(__file__).parent / "baseline_results.json"
    # Convert numpy types to Python
    clean = []
    for r in all_results:
        clean.append({k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in r.items()})
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
