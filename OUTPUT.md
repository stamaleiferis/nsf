# NSF — Arterial Pulse Tonometry: Signal Separation

## Project Output Document

**Date:** 2026-03-22
**Repository:** github.com/stamaleiferis/nsf
**Status:** All phases complete. 102 tests passing. CI configured.

---

## 1. Problem Statement

Non-invasive blood pressure monitoring via optical tonometry. A camera observes a deformable elastomer surface with 266 printed fiducial markers (19×14 grid, 2mm spacing) placed over the radial artery at the wrist. The surface deforms from two sources:

1. **Arterial pulse** — spatially localized Gaussian profile (σ ≈ 3mm), quasi-periodic at heart rate (0.67–2.33 Hz)
2. **Motion artifacts** — spatially global, smooth deformations from wrist movement (curl, twist, pinch, shear, elastic noise)

**Goal:** Separate these two sources to extract clean pulse waveforms for blood pressure estimation.

**Key insight:** The pulse is spatially localized (~4–8mm wide) while artifacts are spatially global. With 266 markers, >80% are artifact-only references, enabling spatial separation even when signals overlap spectrally.

---

## 2. Architecture

```
MarkerTimeSeries (T, R, C, 2)        Input: 2D marker positions per frame
        │
        ▼
┌───────────────────────────────┐
│   Signal Separation Pipeline  │
│                               │
│   1. Displacements from rest  │     Subtract frame-0 positions
│   2. Polynomial artifact fit  │     Degree-2 weighted least squares
│   3. Artifact subtraction     │     Raw − polynomial estimate
│   4. Gaussian pulse extract   │     Project residual onto artery mask
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│   Pulse Extraction            │
│                               │
│   • SNR-weighted averaging    │     Best markers near artery
│   • Bandpass filtering        │     0.5–15 Hz
│   • HR detection (FFT)        │     Peak in pulse band
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│   BP Estimation               │
│                               │
│   • Beat segmentation         │     Peak detection
│   • Morphology extraction     │     Systolic, dicrotic, diastolic
│   • Calibrated regression     │     Ridge regression on 8 features
└───────────────────────────────┘
        │
        ▼
BPEstimate (systolic, diastolic, MAP, HR, confidence)
```

---

## 3. Repository Structure

```
nsf/                                    5,233 LOC total
├── SPEC.md                             Technical specification (429 lines)
├── CLAUDE.md                           Project context & decisions
├── requirements.txt                    numpy, scipy, matplotlib, pytest
├── .github/workflows/tests.yml         CI: Python 3.10–3.12
│
├── src/                                2,298 LOC
│   ├── data/
│   │   ├── markers.py                  MarkerTimeSeries, GroundTruth, SyntheticDataset
│   │   ├── loader.py                   Load from simulator .npy output
│   │   └── simulator_bridge.py         Generate datasets via full simulator
│   │
│   ├── synth/
│   │   ├── pulse.py                    Gaussian spatial × physiological temporal
│   │   ├── artifact.py                 Time-varying 2D polynomial (degree 1–3)
│   │   ├── noise.py                    Additive Gaussian measurement noise
│   │   └── generator.py               Compose all → SyntheticDataset
│   │
│   ├── separation/
│   │   ├── temporal_filter.py          Butterworth bandpass/lowpass
│   │   ├── polynomial_fit.py           Weighted 2D polynomial fitting
│   │   ├── gaussian_extractor.py       Project residual onto artery mask
│   │   ├── separator.py               Full pipeline (iterative, configurable)
│   │   ├── metrics.py                  SNR, waveform/spatial correlation, residual
│   │   ├── joint_model.py             Simultaneous polynomial + Gaussian fit
│   │   ├── decomposition.py           PCA/ICA baselines
│   │   └── subspace_separation.py     SVD-based artifact subspace estimation
│   │
│   ├── estimation/
│   │   ├── pulse_extractor.py          HR detection, SNR mapping, waveform extraction
│   │   ├── spatial_fit.py             Gaussian parameter fitting (center, σ, angle)
│   │   ├── artifact_stats.py          Off-artery marker statistics
│   │   ├── bp_estimation.py           Beat morphology, calibration, BP prediction
│   │   └── param_library.py           Parameter sets: save/load/regenerate
│   │
│   ├── viz/
│   │   ├── animate.py                 Marker displacement animation
│   │   └── plots.py                   Static diagnostic plots
│   │
│   └── pipeline.py                    End-to-end + RealTimeSeparator (<1ms/frame)
│
├── tests/                              1,478 LOC, 102 tests
│   ├── test_markers.py                 9 tests — data structures
│   ├── test_synth.py                   20 tests — synthetic generator
│   ├── test_separation.py              23 tests — separation pipeline + metrics
│   ├── test_advanced_separation.py     15 tests — joint, PCA, ICA, subspace
│   ├── test_edge_cases.py              5 tests — short signals, zero amp, 3×3 grid
│   ├── test_estimation.py              15 tests — parameter estimation + close-the-loop
│   ├── test_pipeline.py                9 tests — end-to-end + real-time latency
│   └── test_viz.py                     6 tests — visualization smoke tests
│
├── experiments/
│   ├── baseline_sweep.py               6-axis parameter sweep (polynomial artifacts)
│   ├── baseline_results.json           27 result sets
│   ├── physics_artifact_sweep.py       Full simulator benchmark (4 methods × 10 seeds)
│   └── physics_artifact_results.json   40 result sets
│
└── notebooks/
    └── project_summary.ipynb           Interactive project overview
```

---

## 4. Separation Methods Implemented

| Method | Module | Description |
|--------|--------|-------------|
| **Sequential polynomial** | `separator.py` | Weighted degree-2 polynomial fit → subtract → residual |
| **Polynomial + Gaussian** | `separator.py` + `gaussian_extractor.py` | Sequential + project residual onto artery mask |
| **Iterative** | `separator.py` (n_iterations) | Multi-pass: fit → extract pulse → subtract → re-fit |
| **Joint model** | `joint_model.py` | Simultaneous polynomial + Gaussian in one linear system |
| **PCA** | `decomposition.py` | Select component most correlated with artery mask |
| **ICA** | `decomposition.py` | FastICA with mask-guided component selection |
| **Subspace** | `subspace_separation.py` | Learn artifact temporal basis from off-artery markers (SVD) |
| **Hybrid** | poly → subspace → Gaussian | Polynomial removes global, subspace removes remainder |

---

## 5. Results

### 5.1 Polynomial Artifacts (Ideal Case)

Baseline sweep across 6 validation axes (SPEC §5.3):

| Metric | Nominal | Spec Target | Status |
|--------|---------|-------------|--------|
| SNR improvement | **27.3 dB** | ≥ 10 dB | Exceeded |
| Waveform correlation | **0.984** | ≥ 0.95 | Exceeded |
| Artifact residual | **0.007%** | < 5% | Exceeded |

**Robustness findings:**
- Heart rate: No effect on quality (50–130 BPM)
- Artery map offset: Graceful degradation (0.96 correlation at 5mm error)
- Noise: Main quality limiter at high levels (0.2mm → 8 dB improvement)
- Polynomial degree mismatch: Degree-3 artifact with degree-2 fit → 0.04% residual

### 5.2 Physics-Based Artifacts (Realistic Simulator)

Full simulator with curl, twist, pinch, shear, elastic noise:

| Method | SNR Improvement | Waveform Corr | Artifact Residual |
|--------|----------------|---------------|-------------------|
| Polynomial only | 13.7 dB | 0.32 | 71% |
| Poly + Gaussian | **22.0 dB** | 0.32 | 71% |
| Hybrid (poly→subspace→Gauss) | **24.5 dB** | 0.32 | 65% |
| Iterative (3×) | 18.1 dB | 0.32 | 77% |

**Key finding:** Waveform correlation is limited by artery mask alignment, not the separation method. Well-aligned scenarios (seeds 2, 3) achieve 0.96 correlation; poorly aligned → negative. The polynomial can't capture curl and elastic noise, explaining the 65–71% artifact residual.

### 5.3 Real-Time Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Per-frame latency | < 5 ms | **< 1 ms** |
| Approach | — | Precomputed solve matrix, per-frame matrix-vector multiply |

---

## 6. Phases Completed

| Phase | Title | Key Deliverables |
|-------|-------|-----------------|
| 1 | Data Interface & Synthetic Generator | MarkerTimeSeries, pulse/artifact/noise models, simulator adapter, visualization |
| 2 | Signal Separation Algorithm | Temporal filter, weighted polynomial fit, metrics, 6-axis baseline sweep |
| 3 | Realistic Synthetic Data | Simulator component-wise GT export, physics benchmark, Gaussian extractor |
| 4 | Parameter Estimation | Pulse/HR extraction, Gaussian spatial fit, artifact stats, parameter library |
| 5 | Advanced Separation & BP | Joint model, PCA/ICA, beat morphology, BP calibration framework |
| 6 | Integration & Real-Time | End-to-end pipeline, RealTimeSeparator, <1ms latency |
| 7 | Separation Tuning | Subspace separation, iterative refinement, hybrid approach |
| 8 | Test Coverage & CI | 102 tests, GitHub Actions (Python 3.10–3.12), edge cases |
| 9 | Project Summary | Jupyter notebook, this document |

---

## 7. Key Decisions

1. **Frame-relative displacements:** Separator works with displacements relative to frame 0. Ground truth comparisons must use relative GT.

2. **No temporal pre-filter for polynomial artifacts:** Artery-weighted polynomial fit on raw data avoids losing high-frequency artifact content to lowpass. The spatial structure (localized vs global) provides the discrimination.

3. **Gaussian extraction as post-processing:** Projecting the residual onto the artery mask rejects spatially inconsistent artifact residuals. Improved physics-artifact SNR from 14 → 22 dB.

4. **Hybrid poly → subspace:** Polynomial captures smooth global artifact component; SVD-based subspace captures remaining non-polynomial structure. Best overall at 24.5 dB.

---

## 8. Known Limitations

1. **Physics-based artifact residual (65–71%):** Curl and elastic noise spatial patterns aren't polynomial. The polynomial fit captures the global component but misses local nonlinear deformations.

2. **Artery mask alignment:** Waveform correlation varies from -0.45 to 0.96 across scenarios, dominated by how well the artery mask matches the true pulse location in the deformed grid. Adaptive artery localization would be the highest-impact improvement.

3. **Synthetic-only validation:** All parameter estimation and BP modules validated on synthetic data. Real marker recordings needed for ground-truth validation.

4. **BP calibration requires reference:** The ridge regression calibration framework is implemented but needs paired BP measurements for real calibration.

---

## 9. Dependencies

- Python 3.10+
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7
- Upstream simulator: `/home/stam/synthetic-vitrack-videos/` (modified with `return_components=True`)

---

## 10. How to Run

```bash
# Setup
cd /home/stam/nsf
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Generate synthetic data and run separation
python -c "
from src.synth.generator import generate
from src.separation.separator import SeparationConfig, separate
ds = generate()
gt = ds.ground_truth
result = separate(ds.markers, gt.rest_positions[...,0], gt.rest_positions[...,1], gt.artery_mask)
print('Recovered pulse shape:', result.recovered_pulse.shape)
"

# Run baseline sweep
python experiments/baseline_sweep.py

# Run physics artifact benchmark
python experiments/physics_artifact_sweep.py
```
