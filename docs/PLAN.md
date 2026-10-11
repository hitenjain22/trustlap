# Plans (append-only)

Each milestone gets a short plan before code is written: files, tests, what could go wrong. Later notes (changes of plan, debugging logs) are appended under the milestone, never rewritten.

---

## M0: Scaffold (2026-10-10)

**Files**
- Root: `CLAUDE.md`, `README.md` (placeholder), `LICENSE` (MIT), `.gitignore`, `.github/workflows/ci.yml`.
- `docs/`: `SPEC.md` (copy of the spec), `PLAN.md`, `DECISIONS.md`, `EXPLAINER.md` (stub), `RUNBOOK.md` (stub).
- `sim/`: `pyproject.toml` (package `trustlap`, deps jax/flax/optax/distrax/chex/numpy/matplotlib, dev extras pytest/ruff), `trustlap/__init__.py`, `tests/test_smoke.py`.
- `web/`: Vite + React + TS (strict + `noUncheckedIndexedAccess`), Tailwind, React Router, Zustand, Comlink, zod, d3-scale/shape/array/hierarchy, Vitest, Playwright, eslint, prettier. Placeholder home page using the design tokens from SPEC 11.2. `vercel.json` with the SPA rewrite, cache and security headers from SPEC 13.

**Tests**
- `sim/tests/test_smoke.py`: package imports, JAX runs a jitted op on CPU.
- `web/tests/unit/smoke.test.ts`: trivial Vitest test.
- `web/tests/e2e/smoke.spec.ts`: home page renders the wordmark.

**What could go wrong**
- Intel Mac: JAX wheels for macOS x86_64 stopped at 0.4.38, so local JAX lags CI/Colab. Keep `pyproject.toml` unpinned for now (spec: pin after first working install) and record versions in `RUNBOOK.md`.
- Vite base path: keep `base: '/'` for Vercel.
- Playwright in CI needs `npx playwright install --with-deps chromium`.
- Vercel deploy needs Hiten's account: Claude Code prepares `vercel.json`; Hiten links the project (instructions in `RUNBOOK.md`).
