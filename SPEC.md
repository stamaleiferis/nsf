# Arterial Pulse Tonometry: Signal Separation via Spatial-Temporal Decomposition

## Project Overview

Non-invasive blood pressure monitoring via optical tonometry. A camera observes a
deformable elastomer surface with printed fiducial markers (19x14 grid, 2mm spacing)
placed over the radial artery. The surface deforms due to two sources: (1) the arterial
pulse beneath it, and (2) wrist motion artifacts. This project develops methods to
separate these two signal sources, validated on realistic synthetic data.

**Upstream dependency:** The synthetic data generator at
`/home/stam/synthetic-vitrack-videos/` provides the full rendering pipeline — physics
simulation, camera model, marker rendering, dataset generation. This project operates
on the **marker position time series** output by that pipeline (or by real marker
tracking), not on raw video.

---

## 1. Physical Setup

### 1.1 Sensor Surface
- Flexible elastomer membrane, approximately 38x26 mm (19x14 markers at 2mm spacing)
- Printed with circular fiducial markers in a grid pattern
- Placed on the skin over the radial artery at the wrist
- Sits atop a balloon that provides controlled contact pressure against the skin

### 1.2 Camera
- Views the surface at an oblique angle (~40° tilt from normal)
- Focal length ~3mm, distance ~43mm from surface
- 1920x1080 native resolution, rendered at 640x480
- Frame rate ≥ 30 fps (typically 30–60 fps)
- The oblique viewing angle causes the Z-axis bulge of the arterial pulse to project
  primarily onto the Y-axis in the image plane

### 1.3 Coordinate System
- **X-axis**: Perpendicular to the artery (lateral, across the wrist)
- **Y-axis**: Along the artery (proximal-distal direction along the wrist)
- **Z-axis**: Normal to the skin surface (toward the camera)
- The artery runs approximately along Y within the field of view, with orientation
  angle ±15° from Y

### 1.4 Known Hardware Parameters
From the existing simulator (`mechanical_optical_priors.json`):
- Radial artery diameter: 2.0–3.2 mm (luminal)
- Artery depth below skin: 1.5–4.0 mm
- Skin thickness: 1.0–2.5 mm
- Wrist curvature radius: 20–40 mm
- Contact preload: 0.2–2.0 N

---

## 2. Signal Sources

### 2.1 Physiological Signal (Arterial Pulse)

The radial artery lies beneath the surface. Each cardiac cycle produces a pressure wave
that mechanically deforms the overlying skin and sensor surface.

**Spatial characteristics:**
- Localized around the artery — a narrow band in X, extended along Y
- The deformation profile perpendicular to the artery is approximately Gaussian
  with σ ≈ 2–4 mm (related to artery depth and tissue diffusion)
- Within the ~38 mm FOV along Y, the profile is roughly uniform (the artery is
  much longer than the FOV)
- Asymmetry possible: bone side (stiffer) vs soft tissue side

**Temporal characteristics:**
- Quasi-periodic at heart rate (0.67–2.33 Hz fundamental, i.e., 40–140 BPM)
- Waveform shape encodes blood pressure information (systolic peak, dicrotic notch,
  diastolic runoff)
- Harmonics extend to ~10–15 Hz
- Slow baseline drift from respiration and vasomotor tone (~0.1–0.3 Hz)
- Beat-to-beat variability (heart rate variability)

**Displacement field from pulse:**
The scalar height map h(x,y,t) is a Gaussian centered on the artery. The observed 2D
displacement field has two components:

- **Y-displacement (dominant):** Projection of the Z-bulge through the camera angle:
  `dy_pulse ≈ β · h(x,y,t) · sin(θ_camera)`
  This is a Gaussian centered on the artery. This is the dominant component because
  the camera views at ~40° from normal.

- **X-displacement:** Lateral skin stretch from the bulge gradient:
  `dx_pulse ≈ α · ∂h/∂x`
  This is the derivative of a Gaussian — antisymmetric about the artery centerline,
  zero at center, peak at ±σ, opposite sign on each side.

The existing simulator already models this via coupled 2D Gaussian displacement maps
with configurable lateral_shear_ratio and longitudinal_shear_ratio parameters, plus
tissue_diffusion_factor and spatial_skew.

### 2.2 Motion Artifacts (Wrist Flexion/Extension)

Wrist movement and sensor mechanics cause global, smooth deformation of the surface.

**Spatial characteristics:**
- Spatially smooth — well-described by low-order 2D polynomials
- Affects the entire field of view (not localized like the pulse)
- Includes rigid translation, rotation, and low-order warping

**Types (from existing simulator):**
- **Curl:** Global bending of the surface (±20°)
- **Twist:** Local rotation (±45°)
- **Pinch:** Radial contraction/expansion (±0.7)
- **Shear:** Local displacement (±3.0 mm)
- **Slippage:** Sensor drift (±2.0 mm/s)
- **Elastic noise:** Organic texture displacement (amplitude ≤ 0.8 mm)

**Temporal characteristics:**
- Primarily low frequency (< 1 Hz for postural drift, up to ~5 Hz for tremor)
- Each artifact type has configurable rate (°/s or mm/s)
- Can overlap spectrally with the pulse signal, especially at low frequencies
- Episodic — may be absent during still periods, strong during movement

---

## 3. Data Abstraction Layer

This project operates at the **marker position** level, not the video level. The input
is a time series of 2D marker positions, regardless of whether they come from:
- The synthetic simulator (ground truth positions)
- A marker tracking algorithm running on real or synthetic video
- Dense optical flow sampled at marker locations

### 3.1 Primary Data Structure

For a grid of R×C markers over T frames:

```
positions:    (T, R, C, 2)    — XY position of each marker at each frame
velocities:   (T, R, C, 2)    — Frame-to-frame displacement (derived)
visibility:   (T, R, C)       — Visibility/confidence score per marker [0,1]
```

### 3.2 Ground Truth (synthetic only)

```
pulse_displacement:     (T, R, C, 2)  — Pulse-only component
artifact_displacement:  (T, R, C, 2)  — Artifact-only component
noise:                  (T, R, C, 2)  — Measurement noise component
artery_mask:            (R, C)        — Binary/soft mask of artery influence
pulse_waveform:         (T,)          — Scalar pulse amplitude A(t)
```

### 3.3 Auxiliary Inputs

- **Artery probability map:** (R, C) confidence map of artery location. Semi-robust;
  derived from pulse detection algorithms on the real data. Provides spatial prior.

---

## 4. Signal Separation Approach

### 4.1 Core Insight

The pulse signal is **spatially localized** (Gaussian in cross-artery direction, ~4–8 mm
wide) while artifacts are **spatially global** (smooth, affecting all markers similarly).
This spatial structure enables separation even when the signals overlap spectrally.

The key enabler: the field of view (~38x26 mm) is significantly larger than the artery
influence zone (~4–8 mm wide), guaranteeing that many markers capture **artifact-only**
signal. These markers serve as artifact references.

### 4.2 Method Outline

#### Step 1: Temporal Pre-filtering
Apply a frequency-selective filter to marker position time series to remove the
pulsatile components (heart rate fundamental and harmonics up to ~15 Hz). This yields
low-passed positions containing only:
- Slow baseline BP changes (< ~0.5 Hz)
- Low-frequency artifact deformations
- DC marker positions

**Why filter first:** The polynomial fit in Step 2 needs to capture only the artifact
field. If pulse components remain, they will distort the polynomial fit, especially
near the artery where pulse amplitude is highest.

#### Step 2: Spatial Polynomial Fit (Artifact Estimation)
For each frame, fit a low-order 2D polynomial to the low-passed marker displacements:

```
d_artifact_x(x, y) = Σ_{j+k ≤ N} c_xjk · x^j · y^k
d_artifact_y(x, y) = Σ_{j+k ≤ N} c_yjk · x^j · y^k
```

Polynomial degree N = 1 (affine) to N = 3 (mild nonlinear warping).

**Weighted fitting using artery map:** Markers far from the artery (low artery
probability) get high weight — they are pure artifact references. Markers near the
artery get low weight — they contain pulse signal that would bias the polynomial fit.

#### Step 3: Artifact Subtraction
Subtract the polynomial-predicted artifact displacement from the **raw (unfiltered)**
marker data at each frame:

```
d_residual_i(t) = d_raw_i(t) - d_artifact_i(t)
```

The residual contains: pulse signal + high-frequency noise + polynomial fit error.

#### Step 4: Pulse Extraction and Validation
- The residual displacement field should be spatially consistent with a Gaussian
  centered on the artery (from the artery probability map)
- Optionally fit a Gaussian spatial model to the residual to further denoise
- The temporal signal at the artery center gives the cleanest pulse waveform

### 4.3 Key Assumptions
- Artifact deformation is spatially smooth (polynomial of degree ≤ 2 or 3)
- Pulse deformation is spatially localized (Gaussian, σ ≈ 2–4 mm)
- The artery location is approximately known (from probability map or prior)
- Locally smooth deformation (no discontinuities)
- The FOV contains enough artifact-only markers to constrain the polynomial fit

### 4.4 Known Challenges
- **Spectral overlap:** Low-frequency BP changes look like slow artifacts. The spatial
  structure (localized vs global) is the only discriminator here.
- **Artery map uncertainty:** If the artery probability map is wrong, the polynomial
  weights are wrong, and separation degrades. Need graceful degradation.
- **Nonlinear artifacts:** If real artifacts are not well-captured by low-order
  polynomials (e.g., elastic noise), separation quality suffers.
- **Low SNR:** When pulse amplitude is small relative to artifact magnitude.

---

## 5. Synthetic Data for Validation

### 5.1 Leveraging the Existing Simulator

The synthetic-vitrack-videos codebase already provides:

| What we need | What exists | Gap |
|---|---|---|
| Marker grid geometry | 19x14, 2mm spacing, with jitter | None |
| Pulse spatial model | 2D Gaussian with tissue diffusion, shear ratios, skew | None |
| Pulse temporal waveform | Sinusoidal + "standard" physiological waveform | May need more waveform variety |
| Artifact deformations | Curl, twist, pinch, shear, elastic noise, slippage | Need polynomial decomposition view |
| Camera projection | Full perspective + lens distortion | None |
| Visibility scoring | Per-marker FOV + overlap + occlusion | None |
| Ground truth output | Per-marker XY + visibility as NPY | Need separate pulse/artifact GT |
| Temporal evolution | Dynamic rates for all parameters | None |
| Scenario sampling | 4-tier curriculum (golden/clinical/edge/broken) | None |

**Key gap:** The existing simulator outputs **total** marker positions. For signal
separation validation, we need the simulator to also output the **individual components**
(pulse-only displacement, artifact-only displacement) as separate ground truth channels.

### 5.2 Required Simulator Extensions

1. **Component-wise ground truth export:**
   Modify the rendering pipeline to output, for each frame:
   - `positions_total`: Final marker positions (existing)
   - `displacement_pulse`: Pulse-only displacement from rest position
   - `displacement_artifact`: Artifact-only displacement from rest position
   - `displacement_mechanical`: Balloon/contact-only displacement from rest position

2. **Simplified marker-position-only mode:**
   A lightweight mode that outputs just the marker position time series (no rendering),
   since signal separation operates on positions, not images. Much faster generation.

3. **Polynomial-compatible artifact parameterization:**
   Add an option to generate artifacts purely as time-varying polynomials (degree 1–3),
   in addition to the existing physics-based artifacts. This lets us control how well
   the polynomial separation assumption holds.

### 5.3 Validation Experiment Design

Generate synthetic datasets with controlled variation along these axes:

| Axis | Range | Purpose |
|------|-------|---------|
| Pulse SNR | 0.1x–10x nominal | Separation vs signal strength |
| Artifact magnitude | 0–100% of hard limits | Separation vs artifact strength |
| Artifact type | Polynomial-only, physics-based, mixed | Assumption violation |
| Artery position | Center, edge, off-nominal | Spatial robustness |
| Artery map error | 0–5 mm offset, wrong width | Prior robustness |
| Heart rate | 40–140 BPM | Temporal filter design |
| Noise level | 0–2x nominal | Noise floor |
| Polynomial degree | 1–3 for artifacts, 1–3 for fit | Model order selection |

---

## 6. Project Phases

### Phase 1: Data Interface and Lightweight Synthetic Generator
**Goal:** Build the data abstraction layer and a fast, lightweight synthetic data
generator for rapid algorithm iteration.

**Deliverables:**
- Data structures for marker position time series (§3.1)
- Lightweight Python module that generates marker positions directly (no rendering):
  - Pulse model: 1-Gaussian spatial × parametric temporal waveform
  - Artifact model: time-varying 2D polynomial
  - Noise model: additive Gaussian
  - Ground truth: separate pulse/artifact/noise components
- Adapter to import data from the full simulator (synthetic-vitrack-videos)
- Basic visualization: marker displacement animation, component overlay

**Why lightweight first:** The full simulator renders video frames at ~4K lines of code.
For signal separation development, we need fast iteration on marker positions only.
Rendering is irrelevant until we need to test end-to-end with marker tracking.

### Phase 2: Signal Separation Algorithm
**Goal:** Implement and validate the spatial-temporal separation method.

**Deliverables:**
- Temporal filtering module (configurable bandpass/notch for pulse harmonics)
- Spatial polynomial fitting with artery-map-weighted least squares
- Artifact subtraction and residual extraction
- Performance evaluation framework:
  - Separation SNR (recovered pulse power / residual artifact power)
  - Waveform fidelity (correlation with ground truth pulse waveform)
  - Spatial fidelity (correlation of recovered spatial pattern with true Gaussian)
- Baseline results on lightweight synthetic data across the validation axes (§5.3)

### Phase 3: Realistic Synthetic Data via Full Simulator
**Goal:** Validate separation on realistic synthetic data from the full simulator.

**Deliverables:**
- Extend synthetic-vitrack-videos to export component-wise ground truth
- Generate datasets with full physics-based artifacts (curl, twist, pinch, shear,
  elastic noise) — these may violate the polynomial assumption
- Quantify performance gap: polynomial artifacts vs physics-based artifacts
- Refine separation method based on findings (adaptive polynomial degree, robust
  fitting, etc.)

### Phase 4: Parameter Estimation from Real Data
**Goal:** Learn realistic model parameters from actual recorded data.

**Deliverables:**
- Fit spatial Gaussian parameters (σ, α, β) to real marker data
- Estimate artifact statistics from markers far from artery
- Extract and characterize pulse waveforms from highest-SNR markers
- Build library of realistic parameter sets for synthetic generation
- Close the loop: generate synthetic data that statistically matches real data

### Phase 5: Advanced Separation and BP Estimation
**Goal:** Improve robustness and connect to blood pressure estimation.

**Deliverables:**
- Joint model-based separation (fit pulse Gaussian + polynomial artifact simultaneously)
- Comparison with ICA/PCA-based approaches
- Handling of spectral overlap (low-frequency BP changes vs slow artifacts)
- Adaptive methods for non-stationary artifacts
- BP estimation from separated pulse waveform:
  - Pulse transit time features
  - Waveform morphology features (systolic/diastolic ratio, dicrotic notch timing)
  - Calibration against reference BP measurements

### Phase 6: Integration and Real-Time
**Goal:** End-to-end pipeline from video to BP estimate.

**Deliverables:**
- Integration with marker tracking (from synthetic-vitrack-videos neural network or
  classical tracking)
- Real-time separation algorithm (target: ≤ 5ms per frame for 19x14 grid)
- End-to-end validation: video → tracking → separation → BP estimate
- Robustness testing across subjects, wrist positions, skin types

---

## 7. Success Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Separation SNR improvement | ≥ 10 dB over unseparated signal | 2 |
| Waveform correlation (synthetic) | ≥ 0.95 with ground truth pulse | 2 |
| Polynomial artifact residual | < 5% of artifact power | 2 |
| Physics-based artifact residual | < 15% of artifact power | 3 |
| Real data waveform quality | Visually clean pulse, HR detection > 99% | 4 |
| BP estimation error | < 5 mmHg systolic, < 3 mmHg diastolic (AAMI) | 5 |
| Real-time latency | < 5 ms per frame (19x14 grid) | 6 |

---

## 8. Technical Constraints and Notes

- The camera viewing angle (~40° tilt) means Z-deformation projects primarily onto Y.
  The ratio of Y-to-X displacement from the pulse depends on camera angle and tissue
  mechanics. The existing simulator parameterizes this via lateral_shear_ratio and
  longitudinal_shear_ratio.
- The artery probability map is "kind of robust" — algorithms must degrade gracefully
  when the map is inaccurate. Phase 2 validation must include artery map error sweeps.
- Deformation is locally smooth — no discontinuities or sharp spatial features expected.
- The FOV (~38x26 mm) is larger than the artery influence zone (~4–8 mm wide),
  guaranteeing artifact-only reference markers. With 19x14 = 266 markers, the majority
  (>80%) are artifact-only, providing strong constraint for polynomial fitting.
- Spectral overlap between pulse and artifact is the fundamental hard case — pure
  frequency-domain separation is insufficient, which motivates the spatial approach.
- The existing simulator supports reproducible generation via seed-based RNG, enabling
  exact reproduction of validation experiments.

---

## 9. Repository Structure

```
nsf/                              ← This project (signal separation)
├── SPEC.md                       ← This document
├── src/
│   ├── data/                     ← Data structures and I/O
│   │   ├── markers.py            ← MarkerTimeSeries class
│   │   └── loader.py             ← Load from simulator / real data
│   ├── synth/                    ← Lightweight synthetic generator
│   │   ├── pulse.py              ← Pulse spatial + temporal model
│   │   ├── artifact.py           ← Polynomial artifact model
│   │   ├── noise.py              ← Noise model
│   │   └── generator.py          ← Compose all components
│   ├── separation/               ← Signal separation algorithms
│   │   ├── temporal_filter.py    ← Bandpass / notch filtering
│   │   ├── polynomial_fit.py     ← Weighted 2D polynomial fitting
│   │   ├── separator.py          ← Full separation pipeline
│   │   └── metrics.py            ← Evaluation metrics
│   └── viz/                      ← Visualization utilities
│       ├── animate.py            ← Marker displacement animation
│       └── plots.py              ← Static diagnostic plots
├── notebooks/                    ← Exploration and analysis
├── tests/                        ← Unit and integration tests
└── experiments/                  ← Reproducible experiment configs and results

synthetic-vitrack-videos/         ← Upstream simulator (separate repo)
├── synthetic_video_gen.py        ← Full physics simulation + rendering
├── simulation_api.py             ← Programmatic API
├── dataset_config.py             ← Scenario sampling
├── generate_dataset.py           ← Batch generation
└── ...
```
