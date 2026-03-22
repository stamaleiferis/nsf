# NSF — Arterial Pulse Tonometry Signal Separation

## Identity
You are stam. You have exchange tools connected to the exchange-server MCP.

## Project
Non-invasive BP monitoring via optical tonometry. Separate arterial pulse from
motion artifacts using spatial-temporal decomposition of marker position data.
Full spec: `SPEC.md`. Issues: `gh issue list`.

## Current Phase
All phases complete — project at maintenance/iteration stage.

## Progress Log
<!-- Update this section at the end of each session with what was accomplished -->
- 2026-03-22: Created repo, spec, and GitHub issues #1–#6 for all phases
- 2026-03-22: Completed Phase 1 — all deliverables implemented (markers.py, pulse.py, artifact.py, noise.py, generator.py, loader.py, animate.py, plots.py). 29 tests passing. Venv at `.venv/`.
- 2026-03-22: Completed Phase 2 — signal separation pipeline (temporal_filter.py, polynomial_fit.py, separator.py, metrics.py). 48 tests passing. Baseline sweep across 6 axes: 27 dB SNR improvement, 0.984 waveform correlation, 0.007% artifact residual at nominal settings.
- 2026-03-22: Completed Phase 3 — extended simulator with `return_components`, built simulator bridge, benchmarked on physics-based artifacts (22 dB SNR improvement with Gaussian extraction, 71% artifact residual). Added gaussian_extractor.py. 52 tests passing.
- 2026-03-22: Completed Phase 4 — parameter estimation infrastructure (pulse_extractor.py, spatial_fit.py, artifact_stats.py, param_library.py). 67 tests passing. Validated close-the-loop on synthetic data.
- 2026-03-22: Completed Phase 5 — joint model separation, ICA/PCA baselines, BP estimation with beat morphology and calibration framework. 80 tests passing.
- 2026-03-22: Completed Phase 6 — end-to-end pipeline (pipeline.py), real-time separator (<1ms per frame). All 6 phases done. 89 tests passing.

## Workflow
- **Start of session:** Run `gh issue list` and check this file to orient yourself
- **During work:** Track progress by updating the task checkboxes on the GitHub issue
- **End of session:** Update "Progress Log" above and the "Current Phase" if needed
- **New tasks discovered:** Create new GitHub issues with `gh issue create`
- **Phase complete:** Close the issue, update "Current Phase" to the next one

## Build & Test
- Python project, use pytest for tests
- Run: `cd /home/stam/nsf && python -m pytest tests/`
- No CI yet — run tests locally before committing

## Coding Standards
- Python 3.10+, type hints on public APIs
- NumPy for array operations — marker data is `(T, R, C, 2)` shaped
- Follow repo structure in SPEC.md §9
- Keep modules small and focused — one concern per file

## Upstream Dependency
The synthetic data generator lives at `/home/stam/synthetic-vitrack-videos/`.
Phase 3 extended it with `return_components=True` parameter in `apply_mechanical_deformations()`.

## Key Decisions
<!-- Record non-obvious architectural decisions here as they're made -->
- Separator works with displacements relative to frame 0 (not absolute). Ground truth comparisons must use relative GT.
- Artery-weighted polynomial fit on raw data (no temporal prefilter) works best for polynomial artifacts — avoids losing high-freq artifact content to lowpass. Temporal coefficient smoothing available as option for non-polynomial artifacts.
- Physics-based artifacts (curl, elastic noise) are poorly captured by polynomials (71% residual). Gaussian spatial extraction helps SNR (14→22 dB) but artifact residual stays high. Performance is highly scenario-dependent (artery alignment matters most).
