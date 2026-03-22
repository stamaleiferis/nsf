# NSF — Arterial Pulse Tonometry Signal Separation

## Identity
You are stam. You have exchange tools connected to the exchange-server MCP.

## Project
Non-invasive BP monitoring via optical tonometry. Separate arterial pulse from
motion artifacts using spatial-temporal decomposition of marker position data.
Full spec: `SPEC.md`. Issues: `gh issue list`.

## Current Phase
Phase 1 — Data Interface and Lightweight Synthetic Generator (GitHub issue #1)

## Progress Log
<!-- Update this section at the end of each session with what was accomplished -->
- 2026-03-22: Created repo, spec, and GitHub issues #1–#6 for all phases

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
Phase 3 will extend it. Do not modify it until Phase 3.

## Key Decisions
<!-- Record non-obvious architectural decisions here as they're made -->
