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
- GPU benchmark (Colab T4, JAX 0.11.1, `sim/results/bench_t4.json`): 26.0M env steps/s at 65,536 cars; 4.7M at 1,024. Same code on the Colab CPU: ~28–29k steps/s (plateaus from 1,024 cars) → 926× at 65,536 cars. NumPy loop: 554 steps/s at 1,024 cars → GPU 8,528× at that size. The run took ~25 min, mostly CPU JAX at 16k/65k cars; made bench.json save after every row and shortened the big CPU runs (fewer fused steps, 2 repeats) while keeping 65,536 so the GPU/CPU headline compares like for like.
- **M1 complete** (2026-10-11): all acceptance checks pass.

---

## M2: TS port + live web skeleton ⟶ Ship line 1 (2026-10-11)

**Python**
- `sim/trustlap/export.py`: `config.json` (from `config.to_dict()`), `tracks/showcase.json` (24 test-bank seeds chosen across the range of twistiness, with short names), `bench.json` (from `sim/results/`), `manifest.json` (sizes, sha256, git sha, config hash). `--golden` writes parity fixtures.
- Golden fixtures (JAX on CPU, highest precision): `rng`, `tracks`, `dynamics` (2,000 cases), `obs` (1,000), plus `env_step` (core step: progress, reward, flags), `scripted` (actions) and `scripted_closed_loop` (5 tracks × dry/wet × 400 steps). The last three aren't in SPEC 9.4 but the browser runs the env and the scripted driver from M2 on, so they get the same guarantee.

**TypeScript** (`web/src/sim/`): `rng.ts`, `track.ts`, `conditions.ts`, `dynamics.ts`, `env.ts` (progress, obs, reward, termination), `scripted.ts`, `config.ts` (zod schema for `config.json`). Same algorithms and operation order as Python; state rounded with `Math.fround` after every substep; track arrays are `Float32Array` like the Python bank. Hot path mutates preallocated state (no per-step allocation).

**Parity tests** (`web/tests/parity/`): one per fixture, tolerances from SPEC 9.4.

**Worker + render**: `sim.worker.ts` (Comlink): fixed 20 Hz × speed accumulator, ≤ 4 catch-up steps, transferable Float32Array snapshots, pause on hidden tab. Browser driving mode is continuous: a finished lap rolls over, a crashed car respawns on the centerline after a short fade. Puddles are a track feature shared by all cars (seeded positions; rain level decides how many are active). Renderer: Canvas 2D, static layers (runoff texture, track, kerbs, start line) cached offscreen per track/size; per frame: puddles, skid layer, cars (interpolated), rain streaks.

**Pages**: Home hero (live canvas, 48 scripted cars, rain slider, new-track button, readout, track-draw intro, reduced-motion path) + placeholder story; Playground skeleton (canvas + Weather / Car / Electronics sliders with training-range bands). Nav shows only pages that exist. Self-hosted Archivo variable font.

**Risks**: float32 vs float64 drift (fround + tolerances); tie-breaks in nearest-point search (continuous projection makes them harmless); canvas cost of 1,200 rain streaks + 48 cars (batch paths, cache static layers); Lighthouse on a canvas-heavy page (defer worker start until after first paint, no layout shift from fonts).

Golden fixtures live in `web/tests/parity/golden/` rather than `web/public/artifacts/v1/golden/` (SPEC 9.1): they're test data, so they shouldn't be deployed or count against the artifact budget.

### M2 notes (append-only log)

- Parity: all fixtures passed on the first run; checked the margin (worst dynamics case at 1.4% of tolerance, max abs diff 4e-6) and sensitivity (a planted 1% rolling-resistance change fails the test).
- Hero visibility: true-size cars were 3–5 px at overview zoom; drawn at ≥ 13 CSS px. The default track was tall and narrow; the hero now picks the showcase track that fits the screen.
- Rain e2e test first failed because the cars were still accelerating off the grid at 6 s; measured the series (settles ~11 s; −4.5% mean speed 1 s after rain 0 → 1, still falling) and moved the measurement to 12 s.
- Frame rate: 60 fps at DPR 1 but 36 at DPR 2. Not the rain (36 fps with no rain either), not canvas size (a bare 2880 px canvas does 60). Cause: `backdrop-filter` on the hero panel. Removed → 60 fps everywhere, no long tasks.
- Lighthouse desktop, home: 91/96/100/91 → 100/100/100/100 after batching kerbs, adding robots.txt and underlining in-text links.

---

## M7 follow-ups (2026-10-11 afternoon)

1. Playground telemetry (SPEC 10.2): speed (km/h ↔ mph), lap time, best lap, Δ vs first ghost, front/rear grip gauges (true tick vs estimate bar), crash-risk meter with the governor threshold, 10 s sparklines, steering and throttle/brake bars. Snapshot gains per-car inputs and a focused-car lap timer.
2. Presets, URL state (SPEC 10.5, zod-parsed and clamped, replaceState debounced 300 ms), Copy link, keyboard shortcuts (SPEC 10.4).
3. Draw a track (SPEC 6.1 drawn tracks + 10.2): pointer path → RDP → control points → scale → validate with specific messages → Auto-fix.
4. Load-time long tasks on retina.

Status 2026-10-11 14:30: items 1–3 done (draw-a-track in `web/src/components/TrackDrawer.tsx`, geometry in `web/src/sim/track.ts`); item 4 next.
