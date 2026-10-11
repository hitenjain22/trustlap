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

---

## M1: The world: tracks, physics, benchmark (2026-10-11)

**Files**
- `sim/trustlap/config.py`: frozen dataclasses for every constant in SPEC 6–7 / Appendix A; `OBS_LAYOUT`; `to_dict()` for export.
- `sim/trustlap/rng.py`: PCG32 (O'Neill reference `pcg32_srandom_r` / `pcg32_random_r`, stream 54) + `splitmix32`. Pure Python ints, explicit masking. API: `next_u32`, `next_float01`, `next_range(lo, hi)` (float), `next_int(lo, hi)` (inclusive). The spec's `int_range`/`uniform` in 6.1 map to `next_int`/`next_range`.
- `sim/trustlap/track.py`: generator, validator, padded track bank, `.npz` save/load, CLI to build the train/val/test banks.
- `sim/trustlap/conditions.py`: condition struct-of-arrays, curriculum-aware sampling, puddle field.
- `sim/trustlap/dynamics.py`: one control step = 4 substeps. Written against an array module `xp` (`numpy` or `jax.numpy`) so the same source runs in JAX (GPU/CPU) and as the plain NumPy baseline for the benchmark. Gust noise is an explicit input so the step is deterministic (needed for golden files).
- `sim/trustlap/env.py`: progress tracking, observations (policy / privileged / history), reward, termination, auto-reset, vmapped batch API.
- `sim/trustlap/scripted.py`: pure pursuit + braking-aware curvature speed profile + P speed controller, using true μ.
- `sim/trustlap/bench.py`: steps/s for N ∈ {1, 64, 1k, 4k, 16k, 64k}: JAX on the default device, JAX CPU, NumPy loop (N ≤ 1k) → `bench.json`.
- `sim/scripts/debug_plot.py`: track + puddles + scripted lap coloured by speed → PNG.
- `sim/scripts/eval_scripted.py`: scripted lap completion on dry / wet.
- `sim/notebooks/bench_colab.ipynb`: thin wrapper: install repo, run bench on GPU, download `bench.json`.

**Tests**: `test_rng.py`, `test_track.py` (500 seeds + hand-made bad tracks), `test_dynamics.py` (all ten of SPEC 6.8), `test_env.py` (obs layout, reward terms, each termination rule, auto-reset, history shift, no progress jumps), `test_scripted.py` (completion on a handful of tracks).

**Spec issues found while planning (fixes proposed, see DECISIONS.md)**
1. *Curriculum interpolation (6.2)* — "nominal ± d·half-width, clipped" never reaches the full range when the nominal isn't the range centre (tire wear would top out at 0.125 instead of 0.25, wind at 5 m/s instead of 10, μ_dry never reaches 1.0). Fix: interpolate each bound, `lo(d) = nominal + d·(lo_full − nominal)`, `hi(d) = nominal + d·(hi_full − nominal)`. Same at d = 0 (a point at nominal) and d = 1 (the full range), which is clearly the intent.
2. *Uniform resampling on a closed loop (6.1.6)* — an exact 2.0 m spacing can't tile a loop of arbitrary length. Fix: `N = round(L / 2.0)`, spacing `ds = L / N` (within ±0.1% of 2 m). The track bank stores `length[T]` so `ds` is exact per track.
3. *Action buffer size (6.3.1)* — `[D_MAX=4, 2]` with D up to 4 works if the buffer holds the 4 *previous* actions: D = 0 executes the current action, D = k executes `buffer[k−1]`. No change, just pinning the meaning.
4. *Puddle placement (6.2)* — `s_j ~ U(100, L − 50)` is measured from the episode's start arclength, so cars never spawn inside a puddle.

**What could go wrong**: the risks in SPEC M1 (low-speed blow-up, pinched tracks, progress jumps, recompilation) plus: high track rejection rates (measure the attempt histogram over the banks); stiff tires at dt = 12.5 ms oscillating at 3–5 m/s (sanity test 9 + a jitter check); a 2,600 m track at wet grip may not fit the 120 s evaluation time limit (measure lap times; if needed, propose a length-scaled limit).

### M1 notes (append-only log)

- Track generator: 82% of first attempts failed the 18 m radius check with the spec parameters. Fixed by applying the drawn-track auto-fix (Laplacian smoothing) to generated tracks; see DECISIONS. Banks: 2,000 / 100 / 200 tracks, max 3 attempts, lengths 900–1,539 m.
- Dynamics tests: three first-draft tests were wrong about physics, not the code. Coasting from 20 m/s only slows to 5 m/s in 60 s. A crosswind turns the nose into the wind, so the car doesn't simply drift downwind. Heavy throttle saturates the rear tires (power oversteer). Fixed the tests. One real bug found: `np.sqrt(dt)` promoted the NumPy path to float64. Now `math.sqrt`, plus a dtype test.
- Scripted driver: 0% → 100% dry completion through four traced failure modes (power oversteer → friction-circle pedals; pure-pursuit oscillation → 1 s lookahead + yaw damping; late braking → conservative braking plan + firmer speed control; pedal chatter → acceleration-based speed controller). Tuned on the validation bank only.
- Local benchmark (Intel Mac CPU, 8 tracks): JAX CPU 131k steps/s at 4,096 cars; NumPy loop ~600 steps/s. GPU numbers pending the Colab run.
- GPU benchmark (Colab T4, JAX 0.11.1, `sim/results/bench_t4.json`): 26.0M env steps/s at 65,536 cars; 4.7M at 1,024. Same code on the Colab CPU: ~28–29k steps/s (plateaus from 1,024 cars) → 926× at 65,536 cars. NumPy loop: 554 steps/s at 1,024 cars → GPU 8,528× at that size. The run took ~25 min, mostly CPU JAX at 16k/65k cars; trimmed CPU sizes to ≤ 4,096 and made bench.json save after every row.
- **M1 complete** (2026-10-11): all acceptance checks pass.
