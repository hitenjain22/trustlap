# TrustLap: working notes for Claude Code

The source of truth is [`docs/SPEC.md`](docs/SPEC.md). If something in it is wrong or impossible, stop and say so with a proposed fix. Don't silently deviate.

## Workflow

- Build milestone by milestone (SPEC Section 5). Before coding a milestone, append a short plan to `docs/PLAN.md` (files, tests, risks).
- A milestone is done only when every acceptance check passes and the output has been shown. Then stop and wait for Hiten's go-ahead.
- Keep `docs/EXPLAINER.md` current as components land.
- New dependency → one-line justification in `docs/DECISIONS.md`.
- Load design/frontend skills before UI work (M2, M5, M7).

## Engineering principles (SPEC Section 2)

1. **Single source of truth for constants.** Physics constants, observation layout, action scaling and condition ranges live in `sim/trustlap/config.py` and are exported to `web/public/artifacts/<version>/config.json`. The TypeScript code reads them. Nothing is hand-copied between languages.
2. **Python and TypeScript physics must agree.** Parity is enforced by golden-file tests (SPEC 9.4). Any physics change regenerates the golden files and must pass both test suites.
3. **Pure functions, static shapes.** Simulation code is pure (state in, state out) and JIT-friendly: no Python control flow on traced values, no dynamic shapes, all randomness from explicitly split PRNG keys.
4. **Every number on the website comes from a file a script produced.** No hand-typed results. The site renders from `metrics.json`; if a metric is missing, the UI shows a clear "not computed yet" state, never a fake number.
5. **Ship lines over big bangs.** After each milestone the site builds, deploys and works.
6. **Measure, then claim.** Speedups, accuracies and crash rates are measured with fixed seeds and reported with confidence intervals.
7. **Readable over clever.** Interviewers will read this code. Type hints in Python, strict TypeScript, docstrings on every public function stating units (meters, seconds, radians, newtons).
8. **Units in names when ambiguous:** `speed_mps`, `yaw_rad`, `dt_s`, `force_n`.
9. **Deterministic by default.** Every script takes `--seed`. Same seed gives the same output (to floating-point tolerance on GPU).
10. **Fail loudly.** Validate artifact files with schemas at load time; assert shapes; check for NaNs during training and stop the run with a clear message instead of silently training on garbage.
11. **Commit after every green milestone** with a conventional-commit message (`feat(sim): ...`, `test(web): ...`). Never commit large binaries outside `web/public/artifacts/` (budget in SPEC 9.5).

**Never hand-type metrics.** Not in the site, the README, or the docs. Numbers come from `metrics.json` / `bench.json`.

## Running things

Local toolchain (Intel Mac): Node lives in `~/.local/share/node/bin` (add it to `PATH`); Python 3.12 via `uv`.

```bash
# Python (from repo root)
cd sim
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q

# Web (from repo root)
cd web
npm ci
npm run typecheck && npm run lint && npm test && npm run build
npm run test:e2e          # Playwright (needs `npx playwright install chromium` once)
npm run dev               # local dev server
```

## Code style

- **Python:** ruff (lint) + ruff format (black-compatible formatter), line length 100. Type hints on all public functions. Docstrings state units.
- **TypeScript:** strict mode with `noUncheckedIndexedAccess`; eslint + prettier. All user-facing strings in `web/src/copy/strings.ts`.
- Conventional commits: `feat(sim): ...`, `fix(web): ...`, `test(web): ...`, `docs: ...`, `chore: ...`.
