# TrustLap: build spec for Claude Code

> **Attach this whole file to Claude Code** and start with:
> *"Read TRUSTLAP_SPEC.md end to end. Then copy it into the repo as `docs/SPEC.md`, write `CLAUDE.md` from Section 2, and start Milestone M0. Work milestone by milestone. Stop at the end of each milestone, show me the acceptance checks passing, and wait for my go-ahead."*

---

## 0. How to use this document

**For Hiten (the owner):**
- Each milestone (Section 5) ends at a **ship line**, a point where the project is already deployable and resume-worthy. If time runs out, stop at the last ship line you reached, deploy, and apply. The NVIDIA Ignite deadline is Oct 18, 2026, so target applying by Oct 16 with whatever ship line you've reached. Keep building afterward and update the link.
- You need a free Google account (Colab) and/or a Kaggle account for the GPU work, a GitHub account, and a free Vercel account for hosting.
- After each milestone, read `docs/EXPLAINER.md` (Claude Code updates it as it goes). Interviewers will ask about every piece of this, so you need to be able to explain it in your own words.

**For Claude Code (the builder):**
- Treat this document as the source of truth. If something here is wrong or impossible, **stop and say so** with a proposed fix. Don't silently deviate.
- Before writing code for each milestone, write a short plan in `docs/PLAN.md` (what files, what tests, what could go wrong), then build.
- Never mark a milestone done unless every acceptance check in it passes and you've shown the output.
- If you have design- or frontend-related skills or plugins available (e.g. a frontend design skill), load them before any UI work in M2, M5 and M7.
- Prefer boring, well-known tools. No new dependency without a one-line justification in `docs/DECISIONS.md`.

---

## 1. What we're building

### 1.1 One-sentence pitch

**TrustLap** is a website where a population of AI drivers, trained by self-play reinforcement learning on tens of thousands of simulated cars at once on a GPU, races procedurally generated tracks through rain, puddles, wind, tire wear, laggy steering and noisy sensors. Each car carries a second neural network that infers how slippery the road is from how the car is sliding, live, and adapts. Visitors can drag the rain slider, draw their own track, and watch the AI cope in real time **in their own browser**.

### 1.2 The story the site tells (in order)

1. **The world:** a hand-built GPU physics simulator with tire slip, drag, weight, wind and puddles. It runs N cars in parallel, X times faster than CPU (real benchmark numbers).
2. **The swarm:** a population of drivers trains at once. Weak drivers copy strong ones and mutate their learning settings (population-based training). Shown as a family tree.
3. **The curriculum:** the world automatically gets harsher as the drivers improve (an automatic curriculum inspired by automatic domain randomization).
4. **The sense:** a "teacher" driver that is told the true conditions trains first. Then a "student" learns to infer those conditions from the last second of motion (rapid motor adaptation), and drives without being told anything. A live grip gauge shows the student's estimate against the truth.
5. **The doubt:** an ensemble of students measures its own disagreement. When the car is in conditions it doesn't understand (ice, extreme lag), it knows and slows down.
6. **The proof:** experiments on unseen tracks and conditions, with confidence intervals.

### 1.3 Success criteria ("done" means all of these)

- [ ] Public URL (Vercel) loads in under 3 s on a laptop over normal Wi-Fi and works on a phone.
- [ ] The home page hero shows **live AI cars driving in the browser** (the actual exported neural network, not a video) within 2 s of first paint, with a rain control the visitor can drag.
- [ ] Playground: pick conditions, regenerate tracks, draw a track, compare drivers as ghosts, read live telemetry including grip estimate vs truth.
- [ ] Training Lab: a timeline you can scrub to watch the driver at different points in training (real checkpoints running live), plus population curves and the family tree.
- [ ] Results: every headline number on the site traces to `metrics.json` produced by the evaluation script, with 95% confidence intervals.
- [ ] Repo: README with headline numbers in the first 5 lines, GIF, live link, honest attribution, reproduction instructions; CI green.
- [ ] `docs/EXPLAINER.md`: plain-English explanation of every component plus 20 likely interview questions with answers.

### 1.4 Explicit non-goals (do not build these)

- No user accounts, no database, no server-side compute. The site is static.
- No training in the browser and no GPU in the cloud backend. All training happens offline in Colab/Kaggle notebooks, and the browser loads exported weights.
- No 3D rendering. Top-down 2D is clearer, faster, and looks better when done well.
- No reinforcement learning library frameworks with big abstractions (no RLlib, no Stable-Baselines). Use a small, readable PureJaxRL-style PPO we fully own.

---

## 2. Engineering principles (copy into `CLAUDE.md`)

1. **Single source of truth for constants.** Physics constants, observation layout, action scaling and condition ranges live in one Python file (`sim/trustlap/config.py`) and are exported to `web/public/artifacts/<version>/config.json`. The TypeScript code reads them. Nothing is hand-copied between languages.
2. **Python and TypeScript physics must agree.** Parity is enforced by golden-file tests (Section 9.4). Any physics change regenerates the golden files and must pass both test suites.
3. **Pure functions, static shapes.** Simulation code is pure (state in, state out) and JIT-friendly: no Python control flow on traced values, no dynamic shapes, all randomness from explicitly split PRNG keys.
4. **Every number on the website comes from a file a script produced.** No hand-typed results. The site renders from `metrics.json`; if a metric is missing, the UI shows a clear "not computed yet" state, never a fake number.
5. **Ship lines over big bangs.** After each milestone the site builds, deploys and works.
6. **Measure, then claim.** Speedups, accuracies and crash rates are measured with fixed seeds and reported with confidence intervals.
7. **Readable over clever.** Interviewers will read this code. Type hints in Python, strict TypeScript, docstrings on every public function stating units (meters, seconds, radians, newtons).
8. **Units in names when ambiguous:** `speed_mps`, `yaw_rad`, `dt_s`, `force_n`.
9. **Deterministic by default.** Every script takes `--seed`. Same seed gives the same output (to floating-point tolerance on GPU).
10. **Fail loudly.** Validate artifact files with schemas at load time; assert shapes; check for NaNs during training and stop the run with a clear message instead of silently training on garbage.
11. **Commit after every green milestone** with a conventional-commit message (`feat(sim): ...`, `test(web): ...`). Never commit large binaries outside `web/public/artifacts/` (budget in Section 9.5).

---

## 3. Architecture

### 3.1 The big picture

```
                OFFLINE (Colab / Kaggle GPU)                          ONLINE (static website, Vercel)
 ┌───────────────────────────────────────────────────┐      ┌───────────────────────────────────────────────┐
 │ sim/  Python package "trustlap" (JAX)             │      │ web/  Vite + React + TypeScript                │
 │                                                   │      │                                               │
 │ config.py ─► track.py ─► dynamics.py ─► env.py    │      │ src/sim/   TS port of track + dynamics + env   │
 │                               │                   │      │ src/policy/ tiny MLP inference (Float32Array)  │
 │       scripted.py (baseline)  ▼                   │      │ src/worker/ sim + policy in a Web Worker       │
 │  ppo.py ◄── curriculum.py ◄── pbt.py              │      │ src/render/ Canvas 2D renderer (track, rain,   │
 │     │                                             │      │            puddles, cars, trails, ghosts)      │
 │     ├─► teacher (privileged)  ──► student.py      │      │ src/pages/ Home · Playground · Lab ·           │
 │     ├─► blind baseline              (adaptation)  │      │            Results · Method                    │
 │     └─► ensemble/confidence.py                    │      │ src/charts/ SVG charts (d3-scale/shape)        │
 │                                                   │      │                                               │
 │ evaluate.py ─► metrics.json + figures data        │      │ loads ▼                                       │
 │ export.py ──► weights, config, golden files,      │─────►│ public/artifacts/v1/  (versioned, validated   │
 │               replays, lineage, checkpoints       │      │                        with zod at load)      │
 └───────────────────────────────────────────────────┘      └───────────────────────────────────────────────┘
```

### 3.2 Key decisions (write each into `docs/DECISIONS.md` with the rationale)

| Decision | Choice | Why | Rejected alternative |
|---|---|---|---|
| Sim + training language | Python + **JAX** (with Flax Linen, Optax, Distrax, Chex) | `vmap` + `jit` + `lax.scan` let the whole env and PPO run on GPU with tens of thousands of cars; PureJaxRL proves the pattern | PyTorch (fine, but vectorized envs + training loop fusion are clunkier); Brax/MJX (3D rigid bodies, overkill) |
| Vehicle model | Dynamic bicycle model with simplified Pacejka tires, friction ellipse, drag, rolling resistance, wind, steering lag, action delay | Real tire-slip behaviour (drifting, understeer in rain) at tiny compute cost; textbook model you can explain in an interview | Kinematic model (no slip, so adaptation is pointless); full 4-wheel model (slow, harder to port) |
| Policy size | MLP **3 × 128**, ELU | Enough capacity for this task; small enough for live browser inference for ~64 cars at 20 Hz in plain TypeScript | 256-wide (4× slower in browser); RNN policy (harder to port and train) |
| Adaptation module | MLP over flattened history (H = 20 steps) | Simplest to train and to port; RMA works with this | 1D conv / GRU (more to port, little gain at this scale) |
| Browser inference | Hand-written MLP in TypeScript over `Float32Array`s, in a Web Worker | Zero dependencies, tiny bundle, exact control for parity tests | onnxruntime-web / TF.js (heavy bundles; threading needs COOP/COEP headers) |
| Rendering | **Canvas 2D** with batched paths, `devicePixelRatio` aware | Hundreds of cars + rain particles at 60 fps with zero dependencies | PixiJS/WebGL (big dependency, unnecessary at this scale); SVG (too slow for many moving objects) |
| Charts | Hand-built SVG React components using `d3-scale`, `d3-shape`, `d3-array`, `d3-hierarchy` (math only, React renders) | Full control to meet the chart spec in Section 11.6 | Recharts/Chart.js (fighting defaults to hit the spec) |
| Web framework | **Vite + React + TypeScript (strict)** + React Router | Fast, boring, static output | Next.js (static-export edge cases, not needed) |
| Styling | Tailwind CSS with design tokens defined as CSS variables (Section 11) | Speed + consistency | Unstyled CSS modules (slower to build consistently) |
| State | Zustand store + URL query params for shareable state | Tiny, simple | Redux (overkill) |
| Worker RPC | Comlink | Clean typed worker calls | Hand-rolled postMessage protocol |
| Schema validation | zod (web), pydantic or dataclasses + manual checks (python) | Fail loudly on bad artifacts | None |
| Tests | pytest (sim), Vitest (web unit + parity), Playwright (e2e + screenshots) | Standard | — |
| Hosting | Vercel static (free), artifacts served from `/artifacts/v1/` with long cache headers | Free, fast, preview deploys | GitHub Pages (base-path quirks) |
| Checkpointing | `flax.serialization` msgpack (or Orbax) to Google Drive / Kaggle output every N updates | Colab disconnects | Keeping everything in RAM |

---

## 4. Repository layout

```
trustlap/
├── CLAUDE.md                     # Section 2 + project conventions
├── README.md                     # Section 16.1
├── LICENSE                       # MIT
├── docs/
│   ├── SPEC.md                   # this file
│   ├── PLAN.md                   # per-milestone plans (append-only)
│   ├── DECISIONS.md              # Section 3.2 table + any later decisions
│   ├── EXPLAINER.md              # plain-English explanations + interview Q&A
│   └── RUNBOOK.md                # how to train, evaluate, export, deploy
├── sim/
│   ├── pyproject.toml            # package "trustlap"; deps pinned after first working install
│   ├── trustlap/
│   │   ├── __init__.py
│   │   ├── config.py             # ALL constants, ranges, obs layout, hyperparams (dataclasses)
│   │   ├── rng.py                # portable PCG32 RNG (identical algorithm in TS) for track generation
│   │   ├── track.py              # procedural track generation + validation + track bank
│   │   ├── dynamics.py           # vehicle model: one control step = K physics substeps
│   │   ├── conditions.py         # sampling condition parameters; curriculum ranges
│   │   ├── env.py                # vectorized env: reset/step/obs/reward/termination
│   │   ├── scripted.py           # pure-pursuit + speed-profile baseline driver
│   │   ├── networks.py           # actor-critic, privileged encoder, adaptation module
│   │   ├── ppo.py                # PureJaxRL-style PPO (single agent, vmappable)
│   │   ├── curriculum.py         # automatic difficulty controller
│   │   ├── pbt.py                # population-based training wrapper + lineage log
│   │   ├── student.py            # DAgger-style adaptation training + grip head
│   │   ├── confidence.py         # student ensemble disagreement + calibration
│   │   ├── evaluate.py           # experiment suite -> metrics.json
│   │   ├── bench.py              # GPU vs CPU throughput benchmark
│   │   ├── export.py             # weights, config, golden files, replays, lineage, checkpoints
│   │   └── checkpoint.py         # save/resume to Drive/Kaggle output
│   ├── scripts/                  # thin CLIs: train_teacher.py, train_blind.py, train_student.py, ...
│   ├── notebooks/                # thin Colab/Kaggle wrappers that pip-install the repo and call scripts
│   └── tests/                    # pytest
├── web/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── public/
│   │   ├── artifacts/v1/         # exported by sim/trustlap/export.py (Section 9)
│   │   ├── fonts/                # self-hosted variable font files
│   │   └── og.png                # social preview image (Section 11.10)
│   ├── src/
│   │   ├── main.tsx, App.tsx, routes.tsx
│   │   ├── sim/                  # rng.ts, track.ts, dynamics.ts, conditions.ts, env.ts, obs.ts
│   │   ├── policy/               # mlp.ts, actorCritic.ts, adaptation.ts, ensemble.ts, loader.ts
│   │   ├── worker/               # sim.worker.ts, protocol.ts (Comlink)
│   │   ├── render/               # Renderer.ts, layers/{asphalt,track,puddles,rain,cars,trails,ghosts}.ts, camera.ts
│   │   ├── charts/               # LineChart, ReliabilityDiagram, BarWithCI, LineageTree, Sparkline, Tooltip, Legend, TableView
│   │   ├── components/           # Slider, Toggle, Segmented, Gauge, StatTile, Button, Sheet, Tooltip, Skeleton
│   │   ├── pages/                # Home, Playground, Lab, Results, Method, NotFound
│   │   ├── state/                # store.ts (zustand), urlState.ts
│   │   ├── data/                 # schemas.ts (zod), artifacts.ts (fetch + validate + cache)
│   │   ├── styles/               # tokens.css, base.css
│   │   └── copy/                 # all user-facing strings in one place (strings.ts)
│   └── tests/
│       ├── unit/                 # vitest
│       ├── parity/               # vitest against golden files
│       └── e2e/                  # playwright
└── .github/workflows/ci.yml
```

---

## 5. Milestones (build in this order)

Each milestone lists **tasks**, **acceptance checks**, and **what can go wrong** (more in Section 14). Do not start a milestone until the previous one's acceptance checks pass.

### M0: Scaffold (≈1–2 hours)

**Tasks**
1. Create the repo layout above. `git init`, MIT license, `.gitignore` (Python, Node, `*.msgpack` checkpoints, `runs/`, `.venv/`, `node_modules/`, `dist/`).
2. `CLAUDE.md` from Section 2 plus: how to run tests, code style (ruff + black for Python, eslint + prettier for TS), "never hand-type metrics."
3. `sim/`: `pyproject.toml` with `jax`, `flax`, `optax`, `distrax`, `chex`, `numpy`, `pytest`, `ruff`, `matplotlib`. Do not pin JAX's CUDA flavor in `pyproject.toml`. The notebooks install the right JAX build for the runtime.
4. `web/`: Vite React TS template, strict `tsconfig` (`"strict": true, "noUncheckedIndexedAccess": true`), Tailwind, React Router, Zustand, Comlink, zod, d3-scale/shape/array/hierarchy, Vitest, Playwright, eslint, prettier.
5. CI workflow (Section 12.3) with placeholder tests that pass.
6. Deploy the empty web app to Vercel to prove the pipeline works.

**Acceptance**
- `pytest` passes (one trivial test). `npm run typecheck && npm run lint && npm test && npm run build` pass.
- Vercel URL shows a placeholder page.

**What can go wrong**
- Vite's default base path breaks asset URLs on some hosts. Keep `base: '/'` for Vercel.
- Playwright needs browsers installed in CI (`npx playwright install --with-deps chromium`).

---

### M1: The world: tracks, physics, benchmark (≈ ½ day)

**Tasks**
1. `config.py`: all dataclasses from Sections 6–7 with defaults from Appendix A.
2. `rng.py`: PCG32 (64-bit state, 32-bit output, documented constants) with `next_u32`, `next_float01`, `next_range`. Implement **pure Python integers with explicit 64-bit masking** so the TS port can match bit-for-bit. Unit test with known first outputs for seed 42.
3. `track.py`: generator + validator + track bank (Section 6.1). Build a training bank (2,000 tracks, seeds 0–1999), a validation bank (100 tracks, seeds 100000–100099), and a held-out **test bank** (200 tracks, seeds 200000–200199). Save as `.npz` with padded arrays.
4. `dynamics.py`: Section 6.3, one control step = 4 substeps, pure JAX.
5. `env.py`: vectorized reset/step, observation, reward, termination (Sections 6.4–6.7), conditions sampling (Section 6.2).
6. `scripted.py`: pure pursuit steering + curvature-based target-speed profile + P-controller on speed, friction-aware (uses true μ, so it's a decent "competent human" baseline).
7. `bench.py`: steps/second for N ∈ {1, 64, 1k, 4k, 16k, 64k} cars on (a) GPU JAX, (b) CPU JAX, (c) a naive NumPy Python loop over cars (only up to 1k; extrapolate is NOT allowed; just report measured points). Output `bench.json`.
8. A debug plot script: track + puddles + one scripted lap trajectory colored by speed, saved as PNG.

**Acceptance**
- Tests: track validity (no self-intersections, min radius respected, closed loop, length in range) across 500 random seeds; dynamics sanity tests (Section 6.8) all pass; no NaNs in 10k random steps with random actions over full condition ranges.
- Scripted driver completes ≥ 95% of laps on dry training tracks and ≥ 70% on wet; plot looks right (brakes before corners, slides on puddles).
- `bench.json` exists with GPU numbers from Colab/Kaggle. Record the headline number.

**What can go wrong**
- Tire model blows up at low speed (division by `vx`). Use the low-speed blending in Section 6.3.4.
- Track generator produces figure-eights or pinches. The validator must reject them; regenerate with the next sub-seed.
- Nearest-point search for progress jumps across the track where two parts come close. Use the windowed search in Section 6.5.
- JAX recompiles every call because a Python float/array shape changes. Pass configs as static args or closures, and keep shapes fixed.

---

### M2: TS port + live web skeleton ⟶ **Ship line 1** (≈ ½ day)

**Tasks**
1. `export.py` v0: `config.json`, test-bank subset (24 showcase tracks), golden files for rng, track, dynamics, obs (Section 9.4).
2. TypeScript port: `rng.ts`, `track.ts`, `dynamics.ts`, `conditions.ts`, `obs.ts`, `env.ts`. Same algorithms, same order of operations.
3. Parity tests (Vitest) against golden files.
4. Worker: runs env + scripted driver at fixed 20 Hz sim rate with interpolation for 60 fps rendering (Section 10.6).
5. Renderer v1: asphalt, track, kerbs, puddles, cars, rain particles (Section 11.7).
6. Pages: Home hero with scripted cars driving, a rain slider, and a placeholder story. Playground skeleton with condition sliders.
7. Deploy.

**Acceptance**
- All parity tests pass with the tolerances in Section 9.4.
- 48 scripted cars render at 60 fps on a mid laptop (Chrome performance panel: no long tasks > 50 ms on the main thread).
- Moving the rain slider visibly changes car behaviour within one second.
- Lighthouse (desktop): Performance ≥ 90, Accessibility ≥ 95.

**Ship line 1:** "Built a GPU-parallel vehicle simulator (JAX) with an exact TypeScript port running live in the browser." Already a legit project.

---

### M3: First self-taught driver (blind PPO + curriculum) ⟶ **Ship line 2** (≈ ½–1 day)

**Tasks**
1. `networks.py`: actor-critic MLP (Section 7.1).
2. `ppo.py`: PureJaxRL-style PPO, entire rollout + update in one jitted `lax.scan` (Section 7.2).
3. `curriculum.py`: automatic difficulty (Section 7.4).
4. Train the **blind** driver (no privileged info, no history) with the curriculum. Log curves to JSON every update.
5. Export blind policy + normalization stats; TS `mlp.ts` inference + parity test on golden policy outputs.
6. Web: AI cars drive live on Home and Playground. Add the scripted driver as a ghost.

**Acceptance**
- Blind driver lap completion on validation bank at curriculum difficulty 0.5 ≥ scripted driver's on dry tracks. Training curve is monotone-ish (moving average rises, no collapse).
- Browser AI car laps the showcase tracks with the same completion behaviour as Python on a handful of fixed seeds (closed-loop parity check, Section 9.4).

**What can go wrong:** see Section 14.2 (reward hacking, spinning, NaNs, value-loss explosion, curriculum collapse).

**Ship line 2:** live, self-taught AI driver in the browser.

---

### M4: Teacher + population (PBT) (≈ ½–1 day)

**Tasks**
1. Privileged encoder + teacher policy (Section 7.5).
2. `pbt.py`: population of P members vmapped (Section 7.3), lineage logging.
3. Train the teacher population with the curriculum. Save 12 evenly spaced checkpoints of the best member's policy for the "watch it learn" timeline.
4. Export lineage (`lineage.json`), population curves (`population_curves.json`), checkpoints.

**Acceptance**
- Best teacher beats the blind driver on the validation bank at every difficulty level (paired comparison, same seeds).
- Lineage file contains every exploit/explore event with timestamps (update index), parent/child ids, and hyperparameters before/after.

---

### M5: The student (adaptation + grip gauge) ⟶ **Ship line 3** (≈ ½ day)

**Tasks**
1. `student.py`: adaptation module + grip head trained DAgger-style (Section 7.6).
2. Train an ensemble of 5 students with different seeds (Section 7.7).
3. Export student weights; TS adaptation inference + parity test.
4. Web: Playground telemetry panel with grip gauge (estimate vs truth), the "puddle test" button (drives the car into a puddle on a straight), and the confidence meter.

**Acceptance**
- Student lap completion within the "teacher gap" target (Section 8.3: report the number, don't fake it). Must beat blind on wet/puddle/combined conditions with non-overlapping 95% CIs, **or** the write-up honestly reports that it didn't.
- Grip estimate MAE reported; puddle step-response latency reported.

**Ship line 3:** the core story is live. **Apply to NVIDIA at this point at the latest.**

---

### M6: Evaluation + Results page (≈ ½ day)

**Tasks**
1. `evaluate.py`: the full protocol in Section 8, outputs `metrics.json` (schema in Section 9.3).
2. `confidence.py`: calibration (reliability diagram data, ECE, Brier) and the safety-governor tradeoff sweep.
3. Results page with all figures listed under Results in Section 10.2.

**Acceptance:** every number on Results is read from `metrics.json`; deleting a field shows a "not computed yet" state, not a crash.

---

### M7: Polish, Lab page, story, README, video (≈ ½–1 day)

**Tasks**
1. Training Lab page (Section 10.2, Training Lab): timeline scrubber running real checkpoints live, population curves, family tree, curriculum chart.
2. Swarm replay for the "thousands of cars" moment (Section 9.2 `replays/`).
3. Home scroll story with final copy.
4. Method page, model card, limitations.
5. OG image, favicon, meta tags.
6. README, GIF, 60–90 s demo video, `EXPLAINER.md` final pass.
7. Accessibility and performance pass (Section 11.9, 10.7), Playwright screenshot review at 390 px, 768 px, 1280 px, 1920 px.

---

## 6. Simulation specification

All numbers here are **defaults** that live in `config.py` (Appendix A). Units are SI throughout (m, s, kg, N, rad).

### 6.1 Track generation (`track.py`, ported to `track.ts`)

Track generation runs in **NumPy/float64** offline (Python) and in **float64 JS numbers** in the browser. It is not JAX: tracks are generated once into a padded bank, and the JAX env indexes into that bank.

**Algorithm (`generate_track(seed: u32) -> Track | None`)**
1. `rng = PCG32(seed)`.
2. Number of control points `n = rng.int_range(8, 14)` (inclusive).
3. Angles: `θ_i = 2π·i/n + rng.uniform(-0.35, 0.35)·(2π/n)`, then sort. Radii: `r_i = rng.uniform(110, 260)` m. Points `p_i = r_i·(cos θ_i, sin θ_i)`. Because angles are monotonic and radii positive, the control polygon is star-shaped and cannot self-intersect.
4. "Character" pass: for each point, with probability 0.3, pull it inward by `rng.uniform(0.15, 0.4)·r_i`. This creates hairpins and chicanes.
5. Fit a **closed centripetal Catmull–Rom spline** (α = 0.5) through the points (centripetal avoids cusps and self-loops). Sample densely (64 samples per segment).
6. Compute cumulative arclength and **resample at uniform spacing `ds = 2.0 m`**. Let `N` = number of points (must be ≤ `N_MAX = 2048`, so length ≤ 4,096 m).
7. Tangents by central differences, normals = left-rotated tangents, curvature `κ_i` by the turning angle between consecutive tangents divided by `ds`, smoothed with a 5-point moving average (circular).
8. Track width `W = rng.uniform(11, 14)` m, constant per track.
9. **Validate** (reject if any fails):
   - Length 900–2,600 m.
   - Minimum turning radius `1/max|κ| ≥ 18 m`.
   - Left and right boundary polylines (centerline ± W/2 · normal) have no self-intersections and don't intersect each other (check on a subsample every 4 points; O(N²) is fine offline; in TS it runs once per generated track, still under 30 ms).
   - Non-adjacent clearance: for any `|i − j| > 40` (circularly), `‖c_i − c_j‖ ≥ W + 8 m`. This prevents sections that nearly touch, which would let progress tracking jump across.
10. On rejection, retry with `seed' = splitmix32(seed ^ attempt)` for `attempt = 1..50`. Record `attempt` in the track object. If 50 attempts fail, raise (shouldn't happen. If it does, the parameters above are wrong).
11. Start at index 0, direction of increasing index.

**Track bank:** arrays `centerline[T, N_MAX, 2]`, `tangent[T, N_MAX, 2]`, `normal[T, N_MAX, 2]`, `curvature[T, N_MAX]`, `width[T]`, `n_points[T]`, float32 for JAX. Index modulo `n_points[track]` everywhere so padding is never read.

**Drawn tracks (browser only):** the user's raw pointer path → Ramer–Douglas–Peucker simplification (ε = 1.5 m in world units) → 8–30 control points → scale uniformly so the resulting spline length falls in 900–2,600 m → steps 5–9. On failure, show the specific reason (Section 10.2) and offer **Auto-fix**: Laplacian-smooth the control points (λ = 0.3) up to 10 iterations until valid.

### 6.2 Conditions (`conditions.py`)

Per episode, sample a condition vector. Some parts evolve during the episode (tire wear, wind gusts, sensor noise draws).

| Parameter | Symbol | Unit | Nominal (difficulty 0) | Full training range (difficulty 1) | Out-of-distribution test |
|---|---|---|---|---|---|
| Rain level | R | – | 0 | [0, 1] | 1.0 |
| Dry grip | μ_dry | – | 1.08 | [1.0, 1.15] | 0.25 ("ice") |
| Base grip | μ_base = μ_dry·(1 − 0.45·R) | – | derived | derived | derived |
| Puddle count | k | – | 0 | round(R · 8 · U(0.5, 1)) | 8 |
| Puddle grip multiplier (center) | m_j | – | – | [0.35, 0.6] | 0.25 |
| Puddle radius | ρ_j | m | – | [4, 10] | 10 |
| Tire wear rate | w | per km | 0 | [0, 0.25] | 0.5 |
| Mass multiplier | – | – | 1.0 | [0.85, 1.25] | 1.5 |
| Drag area | CdA | m² | 0.9 | [0.6, 1.4] | 1.8 |
| Rolling resistance | Crr | – | 0.015 | [0.010, 0.020] | 0.030 |
| Power multiplier | – | – | 1.0 | [0.75, 1.15] | 0.6 |
| Brake multiplier | – | – | 1.0 | [0.7, 1.1] | 0.5 |
| Steering time constant | τ | s | 0.05 | [0.02, 0.15] | 0.25 |
| Action delay | D | control steps (50 ms) | 0 | {0, 1, 2} | 4 |
| Sensor noise σ (normalized obs units) | σ_obs | – | 0 | [0, 0.05] | 0.15 |
| Mean wind speed | |w̄| | m/s | 0 | [0, 10], random direction | 15 |
| Gust strength (Ornstein–Uhlenbeck σ, θ = 0.5 s⁻¹) | σ_g | m/s | 0 | [0, 3] | 5 |

- **Puddle placement:** arclength `s_j ~ U(100, L − 50)`, lateral offset `~U(−0.35W, 0.35W)`. Grip multiplier field: `m(p) = Π_j [1 − (1 − m_j)·exp(−d_j²/(2(ρ_j/2)²))]`. Fixed `P_MAX = 8` puddle slots with an active mask for static shapes.
- **Tire grip factor:** `μ_tire = max(0.6, 1 − w · distance_travelled_km)`.
- **Local friction at an axle:** `μ_axle = μ_base · μ_tire · m(axle_position)`. Front and rear are computed separately, which is what makes puddles cause oversteer/understeer.
- **Curriculum interpolation:** for difficulty `d ∈ [0, 1]`, each continuous range is `nominal ± d · (half-width of full range)`, clipped to the full range. Discrete delay: `D_max = round(2d)`. Rain upper bound `= d`.

### 6.3 Vehicle dynamics (`dynamics.py`, ported to `dynamics.ts`)

#### 6.3.1 State per car

`x, y` (m, world), `ψ` yaw (rad), `vx, vy` body-frame velocity (m/s), `r` yaw rate (rad/s), `δ` actual steering angle (rad), action buffer `[D_MAX=4, 2]`, `wind_gust[2]` (m/s, world), plus bookkeeping in the env (Section 6.5).

#### 6.3.2 Vehicle constants (nominal)

`m = 800 kg`, `Iz = 1100 kg·m²` (scaled with mass multiplier), `lf = 1.2 m`, `lr = 1.4 m`, `L = 2.6 m`, `g = 9.81`, `δ_max = 0.42 rad`, `δ̇_max = 3.0 rad/s`, `P = 220 kW`, `F_drive_max = 7,000 N`, `F_brake_max = 1.3·m·g`, brake split 60% front / 40% rear, rear-wheel drive, `ρ_air = 1.225`, side area·coefficient `C_sA_s = 2.5 m²`, center of pressure 0.2 m behind the center of gravity. Tire (simplified Pacejka): `B = 9.0, C = 1.6, E = 0.95`.

#### 6.3.3 Equations (one substep, `dt_sub = 0.0125 s`, 4 substeps per 50 ms control step)

1. **Actions:** `a = (a_steer, a_long) ∈ [−1, 1]²` taken from the delay buffer (index `D`). `δ_cmd = a_steer·δ_max`. `a_long > 0` is throttle fraction, `< 0` is brake fraction.
2. **Steering actuator:** `δ̇ = clip((δ_cmd − δ)/τ, −δ̇_max, δ̇_max)`.
3. **Loads (static):** `Fz_f = m g lr/L`, `Fz_r = m g lf/L`.
4. **Longitudinal forces:** `F_drive = max(a_long, 0)·min(F_drive_max, P·p_mult/max(vx, 1))` at the rear. `F_brake = max(−a_long, 0)·F_brake_max·b_mult`, split front/rear, only while `vx > 0.05`. Clip each axle's longitudinal force to `|Fx_axle| ≤ μ_axle·Fz_axle`.
5. **Slip angles:** `vx_s = max(vx, 0.5)`; `α_f = atan2(vy + lf·r, vx_s) − δ`; `α_r = atan2(vy − lr·r, vx_s)`.
6. **Lateral forces:** `Fy = −μ·Fz·sin(C·atan(B·α − E·(B·α − atan(B·α))))` per axle (sign opposes slip). Friction ellipse: multiply by `sqrt(max(0, 1 − (Fx_axle/(μ·Fz))²))`.
7. **Aero and wind:** wind in body frame `w_b = Rot(−ψ)·(w̄ + gust)`. `u = vx − w_b.x`, `v = vy − w_b.y`. `F_drag = −½ρ·CdA·u|u|`, `F_side = −½ρ·C_sA_s·v|v|` acting at the center of pressure.
8. **Rolling resistance:** `F_rr = −Crr·m·g·tanh(vx/0.5)`.
9. **Body-frame dynamics:**
   - `m(v̇x − vy·r) = Fx_r + Fx_f·cos δ − Fy_f·sin δ + F_drag + F_rr`
   - `m(v̇y + vx·r) = Fy_r + Fy_f·cos δ + Fx_f·sin δ + F_side`
   - `Iz·ṙ = lf·(Fy_f·cos δ + Fx_f·sin δ) − lr·Fy_r − 0.2·F_side`
10. **Integrate (semi-implicit Euler):** update `vx, vy, r, δ` first, then `x += (vx cos ψ − vy sin ψ)·dt`, `y += (vx sin ψ + vy cos ψ)·dt`, `ψ += r·dt`. Wrap `ψ` to (−π, π]. Clamp `vx ≥ 0` (no reverse gear).
11. **Gust update** once per control step: OU process `g ← g − θ·g·dt + σ_g·√dt·N(0, I)`.

#### 6.3.4 Low-speed blending (prevents the classic slip-angle blow-up)

Let `speed = hypot(vx, vy)`. Compute kinematic-bicycle targets `r_kin = vx·tan δ / L`, `vy_kin = r_kin·lr`. Blend `λ = clip((speed − 1)/2, 0, 1)`: `vy ← λ·vy + (1 − λ)·vy_kin`, `r ← λ·r + (1 − λ)·r_kin`. Apply after each substep.

#### 6.3.5 Numerical guards

- In debug mode (`TRUSTLAP_DEBUG=1`), run with `jax_debug_nans` on a small batch. In training, check `jnp.isfinite` on state after each rollout chunk. If any car is non-finite: log the condition vector + state of the first bad car to a file, terminate that car's episode, increment `nan_resets` counter (logged). If `nan_resets` > 0.1% of episodes, abort the run. That means a physics bug, so fix it rather than mask it.
- Everything float32 on GPU. Set `jax.config.update("jax_default_matmul_precision", "highest")` when generating golden files (avoid TF32-style reduced precision on newer GPUs).

### 6.4 Observations (`obs.py` logic inside `env.py`, ported to `obs.ts`)

All in the car frame, normalized to roughly [−1, 1]. Exact order is defined once in `config.OBS_LAYOUT` and exported.

| Group | Size | Contents |
|---|---|---|
| Proprioception | 8 | `vx/30, vy/10, r/2, δ/δ_max, prev_a_steer, prev_a_long, e_y/(W/2), e_ψ/π` |
| Track lookahead | 24 | centerline points at arclengths `[5,10,15,20,30,40,55,70,90,110,135,160]` m ahead, in car frame, divided by 100 |
| Track width | 1 | `W/14` |
| **Total policy obs** | **33** | Gaussian noise `σ_obs` added to proprioception and lookahead |
| Privileged (teacher only) | 14 | `μ_front, μ_rear, μ_base, μ_tire, min puddle multiplier in the next 30 m, mass mult, CdA, Crr, power mult, brake mult, τ, D/4, wind (body frame, 2)`, scaled to ~[−1, 1] |
| History (student only) | 20 × 10 = 200 | last 20 control steps (1 s) of `[proprioception (8), action (2)]`, oldest first, zero-padded at episode start with a `valid` mask channel folded in by zeroing |

Honest note for the write-up: the teacher can "see" puddles ahead through the privileged vector, while the student can only feel them once it's on them. That's an intended, realistic gap.

### 6.5 Progress tracking

- Keep `idx` (nearest centerline index) per car. Each control step search the window `[idx − 5, idx + 25]` (circular) for the nearest point, then project onto the segment to get continuous arclength `s`.
- `Δs` = circular difference of `s`, clipped to `[−5, 5]` m. Cumulative backward progress > 15 m terminates the episode (wrong way).
- Lateral error `e_y` = signed distance from the centerline (positive = left). Heading error `e_ψ` = wrap(ψ − track heading).

### 6.6 Reward (per 50 ms step)

```
r_t = 0.1·Δs                          # progress in meters (≈ speed reward)
    − 0.02·(e_y / (W/2))²             # stay near the middle (weak)
    − 0.01·‖a_t − a_{t−1}‖²            # smooth driving
    + 10·[lap completed]
    − 10·[crash]
```

Rationale: progress dominates, so driving fast is rewarded. Small shaping keeps learning stable. Crashes cost about 100 m of progress.

### 6.7 Termination and reset

| Event | Rule | Type |
|---|---|---|
| Crash: off track | `|e_y| > W/2 + 1.0` | terminal (no bootstrap) |
| Crash: spun | `|e_ψ| > 100°` for > 1 s | terminal |
| Stall | speed < 1 m/s for > 3 s | terminal, penalty −2 |
| Wrong way | cumulative Δs < −15 m | terminal |
| Lap complete | progress ≥ track length | terminal with +10 bonus (training); evaluation records the lap time |
| Time limit | 2,400 steps (120 s) | **truncation: bootstrap with V(s')** |

**Training reset:** random track from the training bank, random start arclength, speed `U(5, 20)` m/s aligned with the track, conditions sampled at the current curriculum difficulty. **Evaluation reset:** start at `s = 0`, speed 5 m/s, one full lap.

### 6.8 Sanity tests (pytest, run on CPU)

1. Zero throttle on flat dry road: car decelerates (drag + rolling); never reverses.
2. Full throttle on a straight: speed approaches a terminal speed where power ≈ drag (check within 5% of the analytic value).
3. Constant small steer at low speed: turning radius ≈ `L / tan δ` (kinematic limit) within 10%.
4. Constant steer at high speed on low μ: understeer (radius larger than kinematic). On a puddle under the rear axle with throttle: oversteer (yaw rate overshoots).
5. Braking distance from 30 m/s: dry < wet < ice, and close to `v²/(2μg)` for dry within 15%.
6. Steering lag: step command → actual δ reaches 63% at ≈ τ (within one substep).
7. Action delay D: action takes effect exactly D control steps later.
8. Wind: a strong headwind lowers top speed, a crosswind induces lateral drift.
9. 10,000 random-action steps across random full-range conditions: all finite.
10. `vmap` over 4,096 cars gives the same result as looping 4 cars individually (first 4 rows).

---

## 7. Learning specification

### 7.1 Networks (`networks.py`, inference ported to `policy/*.ts`)

| Network | Input | Architecture | Output |
|---|---|---|---|
| Actor (blind) | obs (33) | MLP 3×128, ELU | action mean (2) + state-independent `log_std` param (init −0.5) |
| Actor (teacher/student) | obs (33) + z (8) = 41 | MLP 3×128, ELU | action mean (2) + `log_std` |
| Privileged encoder | privileged (14) | 14 → 64 → 8, ELU then tanh on output | z (8) |
| Critic (all agents) | obs (33) + privileged (14) | MLP 3×128, ELU | V (1) |
| Adaptation module (student) | history (200) | 200 → 256 → 128 (ELU) → 8 (tanh) | ẑ (8) |
| Grip head (student) | adaptation's 128-d hidden | 128 → 2, sigmoid × 1.3 | μ̂_front, μ̂_rear |

- **Asymmetric critic:** every critic gets the privileged vector (training-only), including the blind agent's. Better value estimates, same deployable policy. Say this explicitly in EXPLAINER.
- **Init:** orthogonal weights (gain √2 hidden layers, 0.01 actor output, 1.0 critic output), zero biases.
- **Action sampling:** Gaussian with the mean and `exp(log_std)`. The env clips actions to [−1, 1]; log-probabilities use the unclipped sample (PureJaxRL convention). No tanh squashing, which avoids Jacobian bugs.
- **Deployment:** the browser uses the **mean action** (deterministic). Evaluation in Python also uses the mean.

### 7.2 PPO (`ppo.py`)

Implement in the PureJaxRL style: env step, rollout (`lax.scan` over T steps), GAE, and update epochs all inside one jitted function, `vmap`-able over a population axis.

| Hyperparameter | Default |
|---|---|
| Envs per agent | 1,024 (raise to 2,048–4,096 if throughput allows) |
| Rollout length T | 64 steps (3.2 s of driving) |
| Minibatches | 8 |
| Update epochs | 4 |
| γ / GAE λ | 0.99 / 0.95 |
| Clip ε | 0.2 |
| Entropy coefficient | 0.003 (PBT-tuned for the teacher) |
| Value coefficient | 0.5 |
| Learning rate | 3e-4, linear decay to 0 over the run (PBT-tuned for the teacher) |
| Max grad norm | 0.5 |
| Advantage normalization | per minibatch |
| Observation normalization | running mean/var (Welford), updated from rollout batches, **frozen and exported** |
| Reward scaling | divide by running std of discounted return (PureJaxRL `NormalizeVecReward`), training only |

**Must-get-right details**
- **Terminated vs truncated:** GAE uses `done = terminated` for the bootstrap mask. On truncation (time limit), bootstrap from `V(s_final)`, which means storing the pre-reset observation. Write a unit test with a 3-step toy trajectory checking GAE by hand.
- **Auto-reset inside `env.step`** (as PureJaxRL does) with a fresh track, conditions and start state, using split keys.
- **PRNG discipline:** one root key per run → split into per-member → per-update → per-env keys. Never reuse a key. Test: two envs with different keys produce different condition samples.
- **Logging per update** (to `runs/<name>/log.jsonl`): env steps, SPS, mean episodic return, lap completion %, crash %, mean speed, episode length, approx KL, clip fraction, entropy, value loss, explained variance, curriculum difficulty, nan_resets, PBT events. Copy the log to persistent storage every N updates.
- **Health alarms** (print loudly and also log): approx KL > 0.05 for 5 consecutive updates; explained variance < 0 after 20 updates; entropy collapsing below −3 per action dim early; return NaN.

### 7.3 Population-based training (`pbt.py`)

- Population **P = 16** (32 if the benchmark shows headroom). Members share architecture; each has its own params, optimizer state, obs-norm stats, hyperparameters and PRNG stream. Stack them along a leading axis and `vmap` the PPO update over it.
- Tuned hyperparameters per member: learning rate (log-uniform [1e-4, 1e-3]) via `optax.inject_hyperparams`, entropy coefficient (log-uniform [1e-4, 1e-2]). Passed as arrays (not Python floats), so changing them never triggers recompilation.
- **Every 25 updates:** fitness = mean progress per second over the member's episodes in the last 25 updates (all members train at the same curriculum difficulty, so this is comparable). Rank the members. The bottom 25% each copy a random top-25% member (params, optimizer state, obs-norm stats), then perturb each hyperparameter by ×0.8 or ×1.25 (with probability 0.25, resample from the prior instead).
- **Lineage log:** every member gets a new unique id when it copies, so the history is a family tree. Log `{update, env_steps, child_id, parent_id, prev_id, hp_before, hp_after, fitness_parent, fitness_child_before}` and also periodic `{update, member_id, fitness, lr, ent}` snapshots.
- **Known caveat:** PBT tends to collapse the population onto copies of one lineage. That's fine for performance. It's also why the confidence ensemble (7.7) uses independently seeded students, not PBT members.

### 7.4 Automatic curriculum (`curriculum.py`)

- A single global difficulty `d ∈ [0, 1]` controls the condition ranges (Section 6.2). It starts at 0.
- Every 10 updates: success = fraction of finished episodes that ended in lap completion or time-limit truncation (not a crash), at the current `d`. If success > 0.8, then `d += 0.05`. If success < 0.5, then `d −= 0.05`. Clamp to [0, 1]. A step up requires two consecutive qualifying checks (hysteresis).
- 20% of envs always sample from `d = 0` ranges (prevents forgetting easy driving), and 10% sample one parameter at the current boundary (stretch exposure).
- Keep training after reaching `d = 1` until the update budget ends.
- Log `d` per update → `curriculum.json` (this becomes a chart).
- Name it honestly in the write-up: "an automatic curriculum inspired by Automatic Domain Randomization (OpenAI, 2019)." Full per-parameter ADR is a stretch goal.

### 7.5 Teacher (privileged) training

- Actor input = obs + `z = encoder(privileged)`. Encoder trained end to end with PPO.
- Train with PBT + curriculum. Select the **best member on the validation bank at d = 1** (never the test bank).
- Save 12 evenly spaced snapshots (by env steps) of the best final member's ancestor chain for the timeline (store policy + encoder + obs-norm at each snapshot; tiny).

### 7.6 Student: rapid motor adaptation (`student.py`)

Method from Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots" (2021): phase 1 trains a teacher with privileged environment info, phase 2 trains an adaptation module to estimate that info from recent state–action history.

- Freeze the teacher actor and encoder.
- Env state keeps a per-car rolling history buffer (shift register of the last 20 `[proprio, action]` rows) and its valid length.
- **DAgger-style on-policy training:** roll out with actions from `actor(obs, ẑ)` (the student's own estimate), so it learns in the states it actually visits. Labels `z = encoder(privileged)`. Loss = `MSE(ẑ, z) + 0.5·MSE(μ̂, μ_true_axles)`.
- Warm start: for the first 10% of student training, execute with the true `z` with probability β decaying linearly 1 → 0 (prevents early garbage data from constant crashes).
- Adam, lr 3e-4, 4 gradient steps per rollout of T = 32. Train until the validation ẑ-loss plateaus (expect ~20–60M env steps).
- **Stretch:** short PPO fine-tune of the actor with frozen φ.

### 7.7 Ensemble and confidence (`confidence.py`)

- Train **K = 5 students** with different seeds (vmapped, cheap).
- At each step: `ẑ_k` for k = 1..5 → `a_k = actor(obs, ẑ_k)`. **Uncertainty** `u = mean over action dims of std_k(a_k) + 0.5·mean over dims of std_k(ẑ_k)`. The executed action uses the mean of `ẑ_k`.
- **Calibration:** on evaluation rollouts (validation bank), record `(u_t, y_t)` where `y_t = 1` if the car crashes within the next 2 s (40 steps). Fit isotonic regression `u → p_crash`. Export it as a piecewise-linear table.
- **Report on test bank:** reliability diagram (10 equal-count bins), expected calibration error, Brier score, AUROC.
- **Safety governor:** if `p_crash > τ`, cap throttle at `max(0, 1 − (p − τ)/0.2)` and, above `τ + 0.2`, brake at `−0.4`. Sweep `τ ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, off}` → crash rate vs lap time curve with CIs.
- **Honest limitation:** all students share one teacher actor, so disagreement captures uncertainty about the road conditions, not about the driving policy itself. State this on the Method page.

### 7.8 Blind baseline

Same PPO, same curriculum, same network width, no `z`, no history. Train with a small PBT (P = 8) using roughly the same total env steps as the teacher's winning lineage, and state the compute comparison in the write-up.

### 7.9 Compute budget and scaling knobs

**Measure throughput in M1 before committing to a schedule.** Budget so that each training phase fits in ≤ 90 minutes on a free T4 (Colab or Kaggle).

| Phase | Default budget | If too slow, reduce in this order |
|---|---|---|
| Blind | ~150M env steps | envs/agent 1,024 → 512; P 8 → 4 |
| Teacher PBT | ~100M steps per member × 16 | P 16 → 8; steps/member → 60M |
| Students ×5 | ~40M steps each | K 5 → 3 |
| Evaluation | ~200 tracks × 12 suites × 5 drivers × 3 seeds | seeds 3 → 2; tracks 200 → 120 (report it) |

**Platform notes**
- Run `jax.devices()` first. If it shows CPU on a GPU runtime, install the CUDA build of JAX matching the runtime's CUDA (`pip install -U "jax[cuda12]"`), restart, re-check. Record working versions in `RUNBOOK.md` and pin them in the notebooks.
- **Colab (free):** sessions disconnect after idling or after several hours. Mount Drive and checkpoint every 50 updates.
- **Kaggle:** ~30 GPU hours/week. Prefer "Save Version → Save & Run All," which runs the notebook in the background without the tab open (good for long PBT runs). Write checkpoints and logs to `/kaggle/working/`.
- **Checkpoint contents:** params, optimizer states, obs-norm stats, PBT state + lineage so far, curriculum `d`, update index, PRNG key, config hash. The resume script refuses to resume if the config hash changed.
- **Memory sanity:** P × envs × T × (obs + privileged + history) × 4 bytes. With P = 16, 1,024 envs, T = 64, ~250 floats: ≈ 1 GB, fine on a 16 GB T4. Recompute if you change any of them.

---

## 8. Evaluation protocol (`evaluate.py` → `metrics.json`)

### 8.1 Drivers compared

`scripted` (knows true grip), `blind`, `student` (ensemble-mean ẑ), `student_governed` (with the safety governor at the chosen τ), `teacher` (oracle: knows everything; upper bound).

### 8.2 Condition suites

`dry`, `wet` (R ∈ [0.6, 1]), `puddles` (R = 0.8, 8 puddles), `wind` (10 m/s + gusts 3), `worn` (w = 0.25), `laggy` (τ = 0.15, D = 2), `heavy` (mass 1.25, power 0.75), `combined` (full d = 1 sampling), and out-of-distribution: `ice`, `extreme_lag` (τ = 0.25, D = 4), `noisy_sensors` (σ = 0.15), `very_heavy` (mass 1.5).

### 8.3 Protocol

- Test bank (200 unseen tracks) × 3 condition seeds per suite = 600 laps per driver per suite. **Paired:** every driver sees the identical (track, condition seed) pairs.
- **Metrics:** lap completion rate; crash rate; **paired lap time** (mean over laps completed by *all* compared drivers, which avoids survivor bias); mean speed; grip-estimate MAE (student); calibration metrics (Section 7.7); **teacher gap** = (teacher − student)/(teacher − blind) on completion rate.
- **Confidence intervals:** 95% paired cluster bootstrap (resample tracks, 2,000 resamples). Report differences between drivers as paired deltas with CIs.
- **Puddle step-response test:** a straight track, a single puddle (m = 0.4) at 200 m, car at 20 m/s. Latency = time from the rear axle entering the puddle until `|μ̂_rear − μ_rear| < 0.1`. Report the median and IQR over 200 trials, plus a representative time series for the chart.
- **Generalization:** the same metrics on 200 training-bank tracks vs the test bank.
- **Hygiene:** model selection only on the validation bank. Never tune on the test bank. Report every suite, including ones where the student loses.

---

## 9. Export and artifact contract (`export.py`)

### 9.1 Layout

```
web/public/artifacts/v1/
├── manifest.json            # schema_version, created_utc, git_sha, config_hash, files[{path, bytes, sha256}]
├── config.json              # physics constants, dt, substeps, obs layout, action scaling, condition ranges, network specs
├── policies/
│   ├── blind.json + blind.bin
│   ├── teacher_actor.json + .bin, teacher_encoder.json + .bin
│   └── student_{0..4}.json + .bin   # adaptation module + grip head
├── checkpoints/
│   ├── index.json           # [{i, env_steps, difficulty, mean_return, completion}]
│   └── teacher_{00..11}.json + .bin
├── calibration.json         # isotonic table {u: number[], p: number[]}, tau_default
├── tracks/showcase.json     # 24 seeds + names ("Hairpin Valley" etc. — generated, playful but short)
├── metrics.json             # Section 9.3
├── bench.json               # {device, rows: [{impl, n_cars, steps_per_s}]}
├── population_curves.json   # per member, ≤ 500 points each
├── lineage.json             # events + snapshots
├── curriculum.json          # [{update, env_steps, d, success}]
├── replays/swarm.json + swarm.bin   # the "thousands of cars" moment
└── golden/                  # parity fixtures (Section 9.4)
```

### 9.2 Binary formats

- **Weights `.bin`:** little-endian float32, layers concatenated in order. For each dense layer: kernel stored **[in, out] row-major** (Flax's native layout, so TS computes `y = x·W + b`), then bias [out]. The `.json` lists `{name, in, out, activation, byte_offset}` per layer, plus `obs_norm {mean[], std[]}`, `log_std[]`, and a sha256 of the `.bin`.
- **Swarm replay:** 2,048 cars × 300 frames (15 s at 20 Hz) × `(x, y, ψ)` as int16. Positions are quantized relative to the track bounding box (`x_q = round((x − x0)/(x1 − x0)·65535) − 32768`), and `ψ_q = round(ψ/π·32767)`. Plus `member[2048]` as uint8 for coloring and `crashed_at_frame[2048]` as int16 (−1 if never). About 3.7 MB raw, compressed in transit by the host. Record it from the best teacher population at mid-training (so some cars crash, which is more alive).

### 9.3 `metrics.json` schema (zod mirror in `web/src/data/schemas.ts`)

```json
{
  "schema_version": 1,
  "generated_utc": "...",
  "git_sha": "...",
  "headline": {
    "max_parallel_cars": 0, "gpu_speedup_vs_cpu": 0, "train_env_steps_total": 0,
    "train_wall_minutes": 0, "student_crash_reduction_vs_blind_combined": {"value": 0, "ci": [0, 0]},
    "teacher_gap_combined": {"value": 0, "ci": [0, 0]},
    "grip_mae": 0, "puddle_latency_ms": {"median": 0, "iqr": [0, 0]}
  },
  "suites": {
    "<suite>": {
      "<driver>": {"completion": {"value": 0, "ci": [0, 0]}, "crash": {...}, "paired_lap_time_s": {...}, "mean_speed_mps": {...}, "n_laps": 600}
    }
  },
  "paired_deltas": [{"suite": "...", "a": "student", "b": "blind", "metric": "completion", "value": 0, "ci": [0, 0]}],
  "calibration": {"bins": [{"u_mean": 0, "p_pred": 0, "p_obs": 0, "n": 0}], "ece": 0, "brier": 0, "auroc": 0},
  "governor_sweep": [{"tau": 0.1, "crash": {...}, "paired_lap_time_s": {...}}],
  "puddle_step": {"t_ms": [], "mu_true": [], "mu_hat": []},
  "generalization": {"train_bank": {...}, "test_bank": {...}}
}
```

Every headline field is optional in the zod schema. The UI renders "Not computed yet" if one is missing.

### 9.4 Parity tests (the thing that makes the browser trustworthy)

Golden files are produced by `export.py --golden` with `jax_default_matmul_precision = "highest"`. **To match float32 storage, the TS dynamics applies `Math.fround` to every state variable after each substep.**

| Fixture | Content | Tolerance (TS vs Python) |
|---|---|---|
| `rng.json` | first 1,000 outputs for 5 seeds | exact |
| `tracks.json` | 5 seeds: n_points, length, width, every 10th centerline point, attempt count | 1e-6 m (both float64) |
| `dynamics.json` | 2,000 random (state, conditions, action) → state after 1 control step | `|a − b| ≤ 1e-4 + 1e-4·|a|` per component |
| `obs.json` | 1,000 random (state, track, conditions) → obs (no noise) | same as dynamics |
| `policy_<name>.json` | 500 random inputs → outputs, for every exported network | 1e-4 abs |
| `closed_loop.json` | 20 (seed, track, condition) cases × 400 steps with the student: progress at step 400 and crash flag | progress within 2%; crash flag identical in ≥ 18/20 (chaotic dynamics amplify tiny float differences; this is expected and documented) |

### 9.5 Size budget

Total ≤ 15 MB; any single file ≤ 5 MB; `bench/metrics/curves/lineage` JSON each ≤ 1 MB (downsample curves to ≤ 500 points per member). CI fails if exceeded.

### 9.6 Versioning

Each export writes a new `vN/` folder. The web app reads `ARTIFACT_VERSION` from one constant. Vercel serves `/artifacts/*` with `Cache-Control: public, max-age=31536000, immutable`; never overwrite a published version.

---

## 10. Web app specification

### 10.1 Site map and navigation

| Route | Purpose |
|---|---|
| `/` | Home: live hero + the five-step story + results teaser |
| `/playground` | The app: drive conditions, tracks, drivers, telemetry |
| `/lab` | Training Lab: watch it learn, population, family tree, curriculum |
| `/results` | Every experiment, with charts, CIs and tables |
| `/how-it-works` | Method, model card, limitations, credits, reproduction |
| `*` | Not found |

**Top nav** (sticky, solid `--asphalt-800`, 56 px tall, 1 px `--line` bottom border): wordmark "TrustLap" left; links "Playground", "Training lab", "Results", "How it works"; GitHub icon link right. Active link: `--paint` text with a 2 px sodium underline. Mobile (< 768 px): wordmark + menu button opening a full-height sheet.

**Footer:** "Built by Hiten Jain, UC Davis." GitHub repo link, a one-line credit ("Training recipe adapted from PureJaxRL; adaptation method from RMA (Kumar et al., 2021)"), artifact version and commit hash in `--text-3`.

### 10.2 Pages

#### Home (`/`)

**Hero (full viewport height, min 640 px):**
- A full-bleed **live canvas**: one showcase track in overview camera, **48 student-driven cars** running the real exported networks in a Web Worker, rain falling at the current rain level.
- Bottom-left text block (max width 560 px):
  - Headline (display style): **"These cars taught themselves to drive in the rain."**
  - Body: "Every car here is a neural network running in your browser. Turn up the rain and watch them adapt."
  - Controls row: a large **Rain** slider (0–100%, labeled with the current value) and a **New track** button (dice icon).
  - Actions: primary button "Open the playground"; secondary text link "See how it was trained".
- Bottom-right, small live readout in `--text-2`: "48 cars · lap 3 · 2 crashes". This is the only place a middle dot is allowed, because it's a live status line.
- **One orchestrated motion moment:** on first load the track draws itself along its centerline (stroke-dash animation, 900 ms), then the cars launch from the grid. Under `prefers-reduced-motion`, the track is static and cars start immediately.
- While weights load (< 1 s typically): the track renders and cars sit on the grid with a thin progress line under the nav. Never show a spinner over the canvas.

**Story (five steps, a true sequence so numbering is appropriate):** desktop uses a sticky visual on the right (55% width) with text scrolling on the left (max 520 px). Mobile stacks the visual above each text block.

| Step | Title (final copy) | Body (≤ 60 words, written from metrics.json values) | Visual |
|---|---|---|---|
| 1 | "A physics engine built for thousands of cars" | Tire slip, drag, weight, wind and puddles, simulated for N cars at once on one GPU, X× faster than a CPU. | One car with tire-force arrows (live) + the GPU benchmark chart |
| 2 | "A population that teaches itself" | 16 drivers train side by side. Every few minutes the weakest copy the strongest and tweak how they learn. | Swarm replay (2,048 cars, recorded) + mini family tree |
| 3 | "A world that gets harder as they improve" | The simulator raises the difficulty on its own: more rain, worn tires, laggy steering, gusty wind. | Curriculum chart (difficulty vs training steps) with condition icons appearing as they unlock |
| 4 | "A driver that feels the road" | A second network reads the last second of motion and estimates the grip under each axle. No one tells it the road is wet. | Live puddle test: strip chart of true vs estimated grip |
| 5 | "A driver that knows when it's unsure" | Five copies of the sensing network vote. When they disagree, the car slows down before it crashes. | Ice patch demo with confidence halo + mini reliability diagram |

**Results teaser:** three stat tiles from `metrics.headline` (each with its CI and a one-line plain description), then "See all results".

#### Playground (`/playground`)

**Desktop layout (≥ 1024 px)**

```
┌──────────────────────────────────────────────────────────────┬───────────────────────┐
│ Track: [Hairpin Valley ▾] [Seed 1234] [New track] [Draw]      │ Presets               │
│ Driver: (Student | Blind | Teacher | Scripted)  Ghosts [☐☐☐]  │ [Dry] [Monsoon] [Ice] │
│ [▶ / ❚❚] [0.5× 1× 2×] [Restart lap] Camera (Overview|Chase)   │ [Broken car] [Chaos]  │
├──────────────────────────────────────────────────────────────┤                       │
│                                                              │ Weather               │
│                     LIVE CANVAS                              │  Rain ━━━━━○━━ 60%     │
│                                                              │  Puddles ━━○━━━━ 3     │
│                                                              │  Wind ━━━━○━ 8 m/s ⟳   │
│                                                              │  Gusts ━━○━━━━ 1.5     │
│                                                              │ Car                   │
├──────────────────────────────────────────────────────────────┤  Mass, Power, Brakes, │
│ Telemetry                                                    │  Drag, Tire wear      │
│ 143 km/h   Lap 1:12.40  Δ −0.84 vs Blind                     │ Electronics           │
│ Grip  front ▮▮▮▮▮▯▯ est 0.61 / true 0.58   rear …            │  Steering lag, Delay, │
│ Crash risk  ▮▮▯▯▯▯ 12%  (slows above 20%)                    │  Sensor noise         │
│ [speed sparkline] [grip true vs est sparkline] [risk]        │ Safety                │
│ Steering ◀━━○━━▶  Throttle/brake ▮▮▮▯                        │  Governor [on] τ 20%  │
│ [Run the puddle test] [Copy link] [Show tire forces]         │                       │
└──────────────────────────────────────────────────────────────┴───────────────────────┘
```

- The control rail is 340 px, scrolls independently, and groups controls into **Weather / Car / Electronics / Safety**. Groups are collapsible, with Weather open by default.
- **Every slider** shows the value + unit, and its track has a shaded band for the **training range**. When the value is outside that band, a small sodium-colored note appears under it: "Outside what it trained on." This makes the out-of-distribution idea tangible.
- **Presets** (chips) set all sliders: "Dry practice", "Monsoon", "Black ice", "Broken car" (lag + delay + noise), "Everything at once".
- **Drivers:**
  - The segmented control picks the focused driver.
  - Ghost toggles add other drivers on the same track and conditions, drawn translucent with dashed outlines.
  - Each driver has a fixed color + dash pattern everywhere on the site (Section 11.3).
- **Telemetry** (focused car):
  - Speed (km/h, with a toggle to mph), current lap time, best lap, and delta vs the first ghost.
  - Grip gauges: horizontal bars for the front and rear axle. The true μ is drawn as a blue tick, the estimate as a sodium bar.
  - Crash-risk meter with the governor threshold marked.
  - Three 10-second sparklines.
  - Input bars for steering and throttle/brake.
  - Tiny delight: the speed numeral's font width axis maps to speed (condensed at 0, expanded at top speed). Disabled under reduced motion.
- **Crash event:**
  - The car fades over 400 ms and leaves a skid mark.
  - A toast explains what happened in plain words, e.g. "Crashed at 182 km/h after hitting a puddle mid-corner. The safety governor was off." It has a "Restart lap" button.
  - Crashes are counted per driver.
- **Puddle test:** loads a straight with a single puddle, focuses the student, slows to 0.5×, and shows a larger grip chart, then a one-line result: "Noticed the slippery patch in 180 ms."
- **Draw a track:**
  - The canvas switches to a faint grid. The user drags to draw a loop, and releasing near the start closes it.
  - Live validation messages: "The track crosses itself." / "That corner is too tight for this car. Try a wider curve." / "Too short. Make the loop bigger."
  - Buttons: "Use this track", "Auto-fix", "Cancel".
  - Keyboard alternative: "Generate a random track" (accessibility).
- **Copy link:** writes all state into the URL (10.5) and copies it. Toast: "Link copied."
- **Record a clip (stretch):** 10 s `canvas.captureStream()` + `MediaRecorder` to a WebM download.

**Tablet (768–1023):** controls move under the canvas in a 2-column grid; telemetry becomes a row.
**Mobile (< 768):** canvas on top (4:3), a compact telemetry strip (speed, grip est/true, risk), and controls in a bottom sheet with tabs (Weather / Car / Electronics / Safety). Default to 16 cars max.

#### Training Lab (`/lab`)

1. **Watch it learn:**
   - The canvas runs 16 cars with the checkpoint at the scrubber position, under that checkpoint's curriculum difficulty.
   - The scrubber has 12 stops labeled by env steps ("40M steps"), plus a play button that advances every 6 s.
   - Captions are generated from `checkpoints/index.json` + a small hand-written caption per stop *based on what you actually observe* (e.g. "Hasn't learned to brake yet.").
2. **The population:** fitness over training for every member (thin lines in `--text-3`). The winning lineage is highlighted in sodium. Hovering a line highlights that member's lineage. Exploit events are small ticks on the x-axis.
3. **Family tree:** time on the x-axis, one lane per population slot. Copy events are curved connectors from the parent's lane to the child's. The winning lineage is sodium, everything else is muted. Hover shows hyperparameters before and after.
4. **The curriculum:** difficulty vs env steps (step line), with markers where each condition reached its full range.
5. **How the winners tuned themselves:** learning rate and entropy coefficient over time for the winning lineage, as **two small multiples** (never a dual axis).

#### Results (`/results`)

- Headline stat row (from `metrics.headline`) with CIs.
- Then a series of figure blocks. **Each chart's title states the finding** (written after seeing the data). The subtitle names the method and n. Each has a legend, hover tooltips, a "Show table" toggle, and a source line ("From metrics.json · generated <date> · commit <sha>").

| Figure | Form | Notes |
|---|---|---|
| Completion rate by condition | dot plot with 95% CI whiskers, one row per suite, drivers as colored dots | sort suites by student − blind delta |
| Student vs blind, paired difference | diverging dot plot around 0 with CIs | gray midpoint, the zero line emphasized |
| Paired lap time | dot plot with CIs | only laps all drivers completed |
| Out-of-distribution | same dot plot, separate panel, clear "never trained on these" subtitle | |
| Feeling a puddle | line chart, true μ (blue step) vs estimated μ (sodium) over time, puddle entry marked | report median latency + IQR |
| Does the risk estimate mean what it says? | reliability diagram with the diagonal, bin sizes as small bars beneath | ECE, Brier, AUROC printed |
| Safety vs speed | crash rate (y) vs paired lap time (x) for each governor threshold τ, connected, with CI crosses | label the chosen τ |
| GPU benchmark | steps/second vs number of cars, log–log, one line per implementation | label the headline speedup |
| Unseen vs seen tracks | dot plot train bank vs test bank | |

- End with **"What this doesn't show"**: limitations in plain words (simulated only, 2D, single car type, the teacher-shares-actor caveat, compute limits).

#### How it works (`/how-it-works`)

- A pipeline diagram (inline SVG, left-to-right): Simulator → Curriculum → Population (PPO) → Teacher → Student ensemble → Browser.
- One section per component with four short parts: **What it is**, **Why it's here**, **What I measured** (linked to Results), **Borrowed from** (citations).
- **Model card:** inputs, outputs, training data (simulated only), intended use (demo/education), limitations, compute used.
- **Reproduce it:** commands from `RUNBOOK.md`.

### 10.3 States every page must handle

| State | Behaviour |
|---|---|
| Loading artifacts | Skeleton blocks for charts. The canvas shows the track and parked cars. A thin progress line under the nav. |
| Artifact fetch/validation error | Inline message where the content would be: "Couldn't load the trained models. Check your connection and reload." + **Reload** button. Log the zod error to the console. |
| Metric missing in `metrics.json` | That figure shows "Not computed yet." Never a fake number. |
| Tab hidden | Pause the worker (`visibilitychange`) to save battery; resume on return. |
| Slow device | Measure frame time for the first 2 s. If the average is > 20 ms, halve the car count and rain particles, and note it in the readout ("Running 24 cars for smoother playback"). |
| No Web Worker / Canvas | Replace the canvas with a recorded GIF/video of the hero and a short note. |
| Reduced motion | No track-draw animation, no pulsing, static rain texture, swarm replay paused with a play button. |

### 10.4 Interaction details

- **Sliders:** arrow keys step by 1 unit, Shift+arrow by 10. Home/End jump to min/max. Value text is editable on double-click.
- **Hover tooltips** on charts: crosshair for lines, per-mark for dots. Tooltip shows driver name, value with unit, and CI.
- **Keyboard shortcuts** (Playground, shown in a "?" popover): Space play/pause, R restart lap, N new track, 1–4 select driver, C camera mode, T tire forces.
- **Focus management:** opening the draw mode moves focus to its toolbar; Esc cancels.

### 10.5 URL state

`/playground?track=s1234` or `track=hairpin-valley` (showcase slug) `&rain=60&puddles=3&wind=8&windDir=135&gust=1.5&mass=1.1&power=1&brake=1&drag=0.9&wear=0.1&lag=0.05&delay=0&noise=0&driver=student&ghosts=blind,teacher&gov=1&tau=20&speed=1&cam=overview`

Parse with zod, clamp to valid ranges, drop unknown keys. Update the URL with `replaceState`, debounced to 300 ms. Drawn tracks aren't URL-encodable; show "Drawn tracks can't be shared by link yet" when copying.

### 10.6 Worker and render loop

- **One sim worker per page canvas.** Comlink API: `init(config, networks, track, conditions, drivers, nCars, seed)`, `setConditions(partial)`, `setTrack(track)`, `setDrivers(...)`, `reset()`, `setSpeed(x)`, `pause()`, `resume()`, `runPuddleTest()`.
- **Fixed timestep:**
  - The worker advances the sim at exactly 20 Hz × speed multiplier, using an accumulator with `performance.now()`.
  - It catches up at most 4 steps per tick, so there's no spiral of death.
  - After each step it posts a **transferable `Float32Array` snapshot**: per car `[x, y, ψ, speed, slip, μf, μr, μ̂f, μ̂r, pCrash, flags]`, plus focused-car telemetry and an events array (crash, lap).
- **Main thread render** at the display rate with `requestAnimationFrame`. Keep the previous and current snapshots and interpolate (`α = (now − t_curr)/50 ms`, clamp 0–1; interpolate angles on the shortest arc).
- **Zero per-step allocation in the worker:** preallocate all obs/activation buffers. Batched MLP: pack all cars' inputs into one `[N × in]` buffer, then loop layer by layer (outer loop over output units, inner over inputs, sequential memory access).
- **Determinism:** the worker uses the TS PCG32 seeded from the URL seed, so the same link reproduces the same race (modulo frame timing).

### 10.7 Performance budgets (CI-enforced where possible)

| Metric | Budget |
|---|---|
| Initial JS (gzipped) | ≤ 200 KB. Code-split Lab, Results and the charts. |
| Fonts | ≤ 120 KB (one variable font, Latin subset, `font-display: swap`, preloaded) |
| Home artifacts before interactive | ≤ 1.5 MB (config + student ensemble + teacher actor + showcase seeds). Lazy-load the swarm replay when its section is within one viewport. |
| LCP (desktop) | ≤ 2.5 s |
| Frame rate | 60 fps with 48 cars on a mid laptop; ≥ 30 fps with 16 cars on a mid phone |
| Main-thread long tasks during play | none > 50 ms |
| Lighthouse | Performance ≥ 90, Accessibility ≥ 95, Best Practices ≥ 95, SEO ≥ 95 |

---

## 11. Design system

### 11.1 Concept

**"Wet asphalt at night under sodium lamps."** The site is a rainy night race seen from a broadcast camera overhead. Three things carry meaning, and everything else stays quiet:

- **Sodium amber** = the AI (the student car, primary actions, focus).
- **Rain blue** = water and grip (rain, puddles, true grip).
- **Asphalt** = everything else.

The one memorable thing is the **live canvas**. Typography and layout around it stay disciplined.

**Design review (already applied, keep these choices):**
- *Rejected:* near-black + one neon accent. That's a generic AI-site default. Instead, the base is a blue-gray asphalt with real texture, and both colors are environmental and carry meaning (light and water), not decoration.
- *Rejected:* monospace telemetry numbers. Archivo's condensed width with tabular figures gives the motorsport timing-screen feel without the "code font" cliché.
- *Rejected:* a grid of identical rounded cards. Content is laid out as a story with one sticky visual, and the Playground is a real tool layout.
- *Rejected:* ALL-CAPS tracked eyebrow labels above headings, gradient washes, soft drop shadows, and "→" on every button.

### 11.2 Color tokens (`web/src/styles/tokens.css`)

```css
:root {
  color-scheme: dark;
  /* surfaces */
  --asphalt:      #1C2127;  /* page background */
  --asphalt-800:  #252B33;  /* panels, nav, control rail */
  --asphalt-700:  #2F3640;  /* hover, raised, chart gridlines */
  --line:         #3A424D;  /* hairline borders, dividers */
  --runoff:       #181C21;  /* off-track area on canvas */
  --track:        #2A3038;  /* track surface on canvas */
  /* text */
  --paint:        #E9EDF0;  /* primary text, track edge paint (13.8:1 on asphalt) */
  --text-2:       #AEB7BF;  /* secondary (8.0:1) */
  --text-3:       #8A949D;  /* muted, ≥ 4.5:1 on asphalt and panels */
  /* accent */
  --sodium:       #F2B544;  /* UI accent: primary buttons, links, focus ring (8.9:1) */
  --sodium-ink:   #1C2127;  /* text on sodium */
  /* data: drivers (validated set, see 11.3) */
  --drv-student:  #C98500;
  --drv-teacher:  #9085E9;
  --drv-blind:    #D55181;
  --drv-scripted: #8A949D;  /* neutral, always dashed */
  /* sequential: grip / water (single hue, light → dark) */
  --grip-1: #DCEBF7; --grip-2: #A9CDEB; --grip-3: #6FA9DA; --grip-4: #3F7FBF; --grip-5: #285A8F;
  --rain:         #9FC5E8;  /* rain streaks (decorative, low opacity) */
  /* scenery only, never data */
  --kerb-red:     #C8463C;
  /* status (reserved; always with icon + text) */
  --status-critical: #E5534B;
  --status-good:     #3FAE6A;
}
```

- The site is **dark only** for v1 (it's a night race). If a light theme is added later, its data colors must be re-validated against the light surface. Don't just invert.
- Text contrast is checked (WCAG): `--paint` 13.8:1, `--text-2` 8.0:1, `--text-3` ≥ 4.5:1 on both `--asphalt` and `--asphalt-800`, `--sodium` 8.9:1.

### 11.3 Driver identity (used identically on canvas, charts, legends, telemetry)

| Driver | Color | Line/outline pattern | Canvas car |
|---|---|---|---|
| Student (adaptive) | `--drv-student` | solid | solid fill, warm headlight glow in rain |
| Teacher (oracle) | `--drv-teacher` | dotted (2 4) | solid fill |
| Blind | `--drv-blind` | dashed (6 4) | solid fill |
| Scripted | `--drv-scripted` | long dash (10 4) | outline only |

The three hued colors were run through the colorblind-safety validator against `--asphalt` (dark mode, all pairs): **all checks pass** (lightness band, chroma floor, CVD separation ΔE ≥ 13 worst pair, normal-vision ΔE ≥ 19, contrast ≥ 3:1). Scripted is a neutral gray with a distinct dash pattern, so identity is never color-alone. **Never** add a fifth driver color: fold extra series into "Other" or small multiples. Colors follow the entity, never its rank. Hiding a driver must not repaint the others.

### 11.4 Typography

- **One family: Archivo (variable, axes `wght` 100–900, `wdth` 62–125)**, self-hosted woff2, Latin subset, preloaded. Verify the width axis is present in the file you download. If not, use Archivo + Archivo Narrow / Archivo Expanded statics and adjust tokens.
- Width carries personality: **expanded for display** (the broadcast-graphics feel), **normal for reading**, **condensed for telemetry numerals**.

| Role | Size (px) | Weight | Width | Line height | Tracking | Notes |
|---|---|---|---|---|---|---|
| Display (hero) | clamp(44, 6vw, 76) | 760 | 125 | 1.02 | −0.02em | sentence case, max 14 words |
| H1 | 49 | 700 | 118 | 1.08 | −0.015em | |
| H2 | 31 | 650 | 112 | 1.15 | −0.01em | |
| H3 | 20 | 620 | 105 | 1.25 | 0 | |
| Body | 17 | 400 | 100 | 1.6 | 0 | max 68ch |
| Small / captions | 14 | 450 | 100 | 1.5 | 0 | `--text-2` |
| Labels (controls) | 13 | 520 | 100 | 1.3 | 0.005em | sentence case, `--text-2` |
| Telemetry numerals | 28–44 | 620 | 75 | 1.0 | 0 | `font-variant-numeric: tabular-nums` |
| Chart text | 12–13 | 450 | 100 | 1.3 | 0 | tabular-nums for ticks |

- Scale is a ≈1.25 ratio from 16. All headings sentence case. No all-caps anywhere. No italic-highlighted single words in headlines. Numbers get thin spaces/commas (`1,024`) and units with a normal space (`143 km/h`).

### 11.5 Layout, spacing, shape

- **Grid:** 12 columns, max content width 1,280 px, gutters 24 px (desktop) / 16 px (mobile). Text is left-aligned everywhere. The hero canvas is full-bleed.
- **Spacing scale:** 4, 8, 12, 16, 24, 32, 48, 64, 96, 128. Section padding 96 px desktop / 64 px mobile.
- **Radius:** controls 8 px, panels 12 px, chips/pills 999 px, canvas 0 (full-bleed) or 12 px when inset.
- **Depth:** no drop shadows. Separate layers with surface steps (`--asphalt` → `--asphalt-800` → `--asphalt-700`) and 1 px `--line` borders.
- **Breakpoints:** 390, 768, 1024, 1280, 1920. Test all five.

### 11.6 Charts (rules from the data-viz method)

1. **Pick the form first:** dot plots with CI whiskers for comparisons, lines for change over time, a reliability diagram for calibration, log–log for the benchmark. No pie charts, no 3D, **never a dual axis** (use small multiples).
2. **Marks:** 2 px lines, ≥ 8 px dots, 1.5 px CI whiskers with 6 px caps, 2 px surface-colored gap/ring where marks overlap.
3. **Color by job:** drivers use their fixed identity colors (categorical); grip uses the single-hue blue ramp (sequential); paired deltas use a diverging pair with a neutral gray zero line. Status colors only for crash/success states, always with icon + text.
4. **Text wears text tokens, never series colors.** Labels and values in `--paint`/`--text-2`; the colored mark beside them carries identity.
5. **Legend always present for ≥ 2 series,** plus direct labels at line ends when ≤ 4 series. Never a number on every point.
6. **Recessive structure:** gridlines `--asphalt-700` at 1 px, axis labels `--text-3`, no chart borders.
7. **Interaction by default:** crosshair + tooltip on lines, per-mark tooltips on dots, hit targets ≥ 24 px. Filters (driver toggles, suite filter) sit in one row above the chart.
8. **Accessibility:** every chart has a "Show table" toggle rendering the same data as an HTML table, an `aria-label` summarizing the finding, and dash patterns as secondary encoding.
9. **Titles state findings,** written after the data exists. If a finding is weak or negative, say so plainly.
10. **Render and look:** Playwright screenshots of every chart at 390 and 1280 px, reviewed for label collisions and overflow before merging.

### 11.7 Canvas visual spec (render layers, bottom to top)

1. **Runoff:** `--runoff` fill + a pre-rendered 256 px noise tile (random 1 px speckles, 4% opacity) for an aggregate texture.
2. **Track surface:** polygon between the boundaries, `--track`. With rain level R > 0, overlay `--grip-4` at opacity `0.08·R` plus sparse, slowly drifting specular streaks (wet sheen).
3. **Edges and kerbs:** edge lines in `--paint` at 0.25 m world width. On corners with radius < 60 m, alternating `--kerb-red`/`--paint` kerb blocks (2 m long) just outside the edge. Checkered start/finish line.
4. **Puddles:** radial-gradient ellipses in `--grip-4` (35% opacity at the center → 0). When R > 0, small expanding ripple rings (1.5 s loop, low opacity).
5. **Skid marks:** written to a persistent offscreen "skid layer" when slip angle > 6°: `#0F1215` at 35% opacity, 0.3 m wide per rear wheel. Cleared on lap restart.
6. **Trail (focused car):** the last 3 s of positions, colored by confidence on a single-hue sodium ramp (confident = full `--drv-student`, unsure = desaturated `#6B5A3A`), tapering width.
7. **Cars:** 4.4 × 1.9 m rounded rectangles (0.4 m radius) in driver color with a 1 px darker outline, a windshield highlight, and two small soft headlight cones (radial gradient, sodium, 20% opacity) when R > 0.3. Ghosts at 45% opacity with their dash-pattern outline. The focused car gets a 2 px `--paint` outline.
8. **Rain:** up to 1,200 screen-space streaks (count ∝ R), 10–18 px long, angled by the wind, `--rain` at 25% opacity. Reduced motion: a static streak texture instead.
9. **Overlays:**
   - Tire-force arrows (toggle): at each axle, length ∝ force, sodium for lateral, `--paint` for longitudinal.
   - Confidence halo when `pCrash > τ`: a sodium ring, slow 1.2 s pulse (static under reduced motion).
   - Car labels in Chase camera only.
10. **Camera:**
    - Overview fits the track's bounding box with 6% padding.
    - Chase follows the focused car at 3× zoom, **north-up (no rotation**, rotating cameras cause motion sickness), with critically damped smoothing.
    - Canvas resolution = CSS size × `min(devicePixelRatio, 2)`.

### 11.8 Motion

- Durations: 120 ms (hover/press), 200 ms (panels, sheets), 900 ms (the single hero track-draw). Easing `cubic-bezier(0.2, 0.8, 0.2, 1)`.
- No fade-and-slide-up on every section. Motion answers user actions (panel opens, slider changes, crash toast) or shows the simulation itself.
- `prefers-reduced-motion: reduce` disables the track draw, pulses, ripples and the speed-width effect, and swaps rain for a static texture.

### 11.9 Accessibility checklist

- All interactive elements reachable and operable by keyboard. Visible focus ring: 2 px `--sodium` with a 2 px offset.
- Native `<input type="range">` styled (not a div slider), with `aria-valuetext` including units ("60 percent rain").
- The canvas has `role="img"` with an `aria-label`, plus a visually hidden `aria-live="polite"` summary updated at most every 5 s ("Student car, lap 2, 143 kilometers per hour, estimated grip 0.61, no crashes").
- Color never the only signal: dash patterns, labels, icons.
- Touch targets ≥ 44 px on mobile.
- Respect `prefers-reduced-motion`. Test with keyboard only and with VoiceOver for one full flow per page.

### 11.10 Brand assets

- **Wordmark:** "TrustLap" in Archivo, width 125, weight 800, `--paint`.
- **Favicon:** a top-down car silhouette in `--sodium` on `--asphalt` (SVG + 32/180 px PNG).
- **OG image (1200×630):** generated by a Playwright script that loads `/` with a fixed seed and rain 70%, waits 3 s, screenshots the hero canvas, and composites the wordmark + headline. Same for a README hero image. (Optional: polish it in Canva, but the source should be the real sim.)

### 11.11 Copy rules

- Write for a curious visitor, not an ML researcher. Plain verbs, sentence case, units always.
- Name things by what users understand: "Grip estimate" (not "μ̂ latent"), "Crash risk" (not "p_crash"), "Steering lag" (not "τ"). Technical symbols appear only on How it works, defined on first use.
- Buttons say exactly what happens: "Restart lap", "Run the puddle test", "Copy link", "Use this track".
- Errors explain what happened and what to do, in the interface's voice, without apologizing.
- No hype words ("revolutionary", "cutting-edge"). Let numbers do the work.
- All strings live in `src/copy/strings.ts`.

---

## 12. Testing and CI

### 12.1 Python (`sim/tests/`, runs on CPU in CI with small batch sizes)

- `test_rng.py`: known outputs; same seed → same sequence; different seeds differ.
- `test_track.py`: 500 seeds valid; closed loop; uniform spacing; curvature sign sanity on a circle; rejection of hand-made bad tracks (figure-eight, pinch, too tight).
- `test_dynamics.py`: Section 6.8, all ten.
- `test_env.py`: obs shape/order matches `OBS_LAYOUT`; reward components on hand-built states; each termination rule fires correctly; auto-reset produces fresh tracks/conditions; history buffer shifts correctly; progress never jumps across the track on tight sections.
- `test_ppo.py`: GAE hand-check (terminated vs truncated); a tiny 2-D point-mass task learns in < 60 s on CPU (smoke test that the learning loop works at all).
- `test_pbt.py`: exploit copies params + optimizer + obs-norm stats; ids/lineage consistent; hyperparams stay in bounds.
- `test_curriculum.py`: difficulty rises with high success, falls with low, respects hysteresis and clamps.
- `test_export.py`: weight round trip (export → reload in NumPy → identical outputs); manifest hashes match; size budget.

### 12.2 Web (`web/tests/`)

- **Unit (Vitest):** URL state parse/clamp; zod schemas accept a sample `metrics.json` and reject broken ones; formatting helpers (units, CIs); interpolation of angles; MLP on a hand-computed 2-layer example.
- **Parity (Vitest):** Section 9.4, all fixtures. These are the most important tests in the repo.
- **E2E (Playwright, Chromium):**
  - Home loads, canvas non-blank (pixel variance check), rain slider changes the URL-less state and the rain particle count.
  - Playground: change preset → sliders update → URL updates; copy link → open link in new page → same state; draw mode with a scripted pointer path → valid track → cars drive; invalid path → error message shown.
  - Lab: scrub timeline → checkpoint label changes → canvas re-inits.
  - Results: every figure renders; "Show table" toggles; delete a metric field in a fixture → "Not computed yet" appears.
  - Keyboard-only run through Playground controls.
  - Visual snapshots at 390 / 768 / 1280 / 1920 px for each page (review diffs manually; don't auto-approve).
- **Accessibility:** `@axe-core/playwright` on each page: zero serious/critical violations.

### 12.3 CI (`.github/workflows/ci.yml`)

Jobs on every push/PR:
1. **sim:** set up Python, `pip install -e sim[dev]` with CPU JAX, `ruff check`, `pytest -q`.
2. **web:** Node LTS, `npm ci`, `npm run typecheck`, `npm run lint`, `npm test` (unit + parity using committed golden files), `npm run build`, bundle-size check, artifact-size check (Section 9.5).
3. **e2e:** build, `npx playwright install --with-deps chromium`, run Playwright against `vite preview`, upload the report as a CI artifact.
4. **lighthouse (optional):** Lighthouse CI against the preview build with the budgets in Section 10.7.

Vercel builds preview deployments for every PR automatically. Production deploys from `main` only after CI is green.

---

## 13. Deployment

- **Vercel project** rooted at `web/`, build `npm run build`, output `dist/`.
- `vercel.json`:
  - SPA rewrite: all non-file routes → `/index.html`.
  - Headers: `/artifacts/(.*)` and `/fonts/(.*)` → `Cache-Control: public, max-age=31536000, immutable`; HTML → `no-cache`.
  - Security headers: `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, a Content-Security-Policy allowing only self (fonts and artifacts are self-hosted, so no third-party origins are needed).
- Meta: `<title>TrustLap: AI cars that teach themselves to drive in the rain</title>`, description, OG/Twitter tags with `og.png`, canonical URL, `theme-color` `#1C2127`.
- Optional custom domain later. The `*.vercel.app` URL is fine for applications.

---

## 14. What can go wrong (risk register)

Format: **Symptom → likely cause → fix.** Claude Code: when something breaks, check this table first, then Section 15.

### 14.1 Environment and tooling

| Symptom | Likely cause | Fix |
|---|---|---|
| `jax.devices()` shows only CPU on a GPU runtime | wrong JAX build for the runtime's CUDA | `pip install -U "jax[cuda12]"`, restart runtime, re-check; record working versions |
| First training call takes minutes, then each call is slow too | recompilation every call (Python floats/shapes changing, non-static config) | make configs static/closed-over; hyperparams as arrays; check with `jax.log_compiles` |
| Out of GPU memory | P × envs × T too large, or history stored per step | reduce envs/agent, T, or P; store history in env state (rolling) rather than in the rollout buffer twice |
| Colab disconnects mid-run | idle timeout / session limit | checkpoint every 50 updates to Drive; resume script; use Kaggle background runs for long jobs |
| Results not reproducible | PRNG keys reused or not threaded through; nondeterministic GPU reductions | split keys properly; accept ~1e-6 GPU nondeterminism; seed everything; compare on CPU for debugging |
| Notebook works, script doesn't (or vice versa) | notebook state | notebooks only call package functions; restart-and-run-all before trusting results |

### 14.2 Physics and environment

| Symptom | Likely cause | Fix |
|---|---|---|
| NaNs at low speed | slip angle division by ~0 velocity | low-speed blending (6.3.4), `vx_s = max(vx, 0.5)` |
| Car vibrates/jitters when stopped | stiff tire forces with large dt | 4 substeps; low-speed blend; clamp tiny velocities to 0 |
| Car can't turn at all on wet roads | Pacejka sign/units wrong, or friction ellipse zeroing lateral force under braking | unit tests 3–5 in 6.8; plot Fy vs α curve and check its peak ≈ μFz |
| Unrealistic top speed | power/drag constants or force units off | analytic terminal-speed test |
| Progress jumps by hundreds of meters | nearest-point search finds a different part of the track | windowed search; clearance constraint in the track validator |
| Car "teleports" at the start line | arclength wrap not handled | circular differences everywhere; test a lap crossing |
| Puddles invisible to physics | friction sampled at car center only, or puddle mask not applied | sample per axle; test with a car parked on a puddle |
| Same conditions in every env | one key used for all envs | `jax.random.split(key, n_envs)`; test that samples differ |

### 14.3 Learning

| Symptom | Likely cause | Fix |
|---|---|---|
| Return flat at ~0 for a long time | crashes immediately; reward too sparse | start curriculum at d = 0; start speeds low; check reward scale (per-step ~0.1–0.5); confirm the scripted driver gets high return in the same env |
| Car learns to spin in circles or creep slowly | reward hacking: progress not measured correctly, or crash penalty too harsh vs progress | verify Δs is signed and clipped; wrong-way termination; reduce crash penalty or add the stall rule |
| Car hugs the inside and clips off the track | lookahead too short / normalization wrong | verify the lookahead points in the car frame visually; check obs normalization stats |
| Value loss explodes | unscaled rewards; bad truncation handling | reward normalization; bootstrap on truncation; value clipping as a last resort |
| Approx KL huge, policy collapses | learning rate too high, epochs too many | lower lr, epochs 4 → 2; PBT will also prune these members |
| Entropy collapses early, then plateau | ent coef too low | raise ent coef; PBT range |
| Curriculum oscillates up and down | thresholds too close; no hysteresis | 2 consecutive qualifying checks; widen thresholds; 20% easy envs |
| Curriculum never leaves d ≈ 0.2 | the task is too hard at those ranges (e.g. delay) | check per-parameter failure (log which condition was present at crash); widen the step-up window; train longer at that d |
| PBT population collapses to one lineage too early | exploit too aggressive | bottom 20% only; longer ready interval (40 updates) |
| Teacher no better than blind | privileged info not wired in, or the encoder output is saturated | test that changing μ in the privileged vector changes actions; check the z histogram |
| Student ẑ loss plateaus high | history too short or not informative | check history alignment (oldest→newest); H 20 → 30; include slip-related proprio (vy, r); ensure the β warm start |
| Student drives worse than blind | distribution shift (trained on teacher states only) | DAgger: roll out with the student's own ẑ (Section 7.6) |
| Grip estimate lags badly | history window dominated by old data | report it honestly; try H = 10 + H = 30 ablation; it's a result either way |
| Ensemble members identical | same seed or same data stream | different init keys and env keys per member |

### 14.4 Export and parity

| Symptom | Likely cause | Fix |
|---|---|---|
| Browser car drives badly, Python car fine | obs-norm stats not applied in TS, wrong obs order, or kernel transposed | parity tests (policy + obs); export `OBS_LAYOUT` and assert it in TS; kernels are [in, out] |
| Dynamics parity fails by ~1e-3 | float32 vs float64 | `Math.fround` state after each substep; same operation order; same atan2 argument order |
| Closed-loop parity fails badly | substep count, dt, action delay or steering lag implemented differently | compare first divergent step via golden trace dumps |
| Site breaks after a re-export | schema change without a version bump | new `vN/` folder; zod schema version check |

### 14.5 Web and UX

| Symptom | Likely cause | Fix |
|---|---|---|
| Choppy animation | simulation on the main thread or GC pauses | worker + preallocated buffers; interpolate renders |
| Blurry canvas on retina | canvas not scaled by DPR | width/height × min(DPR, 2), CSS size unchanged |
| Sim speeds up/slows with frame rate | stepping per frame instead of fixed timestep | accumulator at 20 Hz |
| Battery drain in background tabs | no pause | `visibilitychange` pause |
| Layout shift when fonts load | font swap metrics | preload font, `size-adjust`/fallback metrics, reserve heights |
| Text unreadable over the canvas | busy track behind text | position text over runoff area; a subtle `--asphalt` 70% backdrop panel behind hero text |
| Mobile overheating | too many cars + particles | device budget: 16 cars, 400 particles, auto-degrade |
| Drawn track invalid with no clue why | generic error | the specific messages in 10.2 + Auto-fix |
| Charts misleading | survivor bias in lap times; dual axes; unlabeled CIs | paired lap times; small multiples; CI explained in subtitles |

### 14.6 Project-level

| Risk | Mitigation |
|---|---|
| Not finishing before the application deadline | ship lines; apply at Ship line 3 at the latest, update later |
| Results are underwhelming (student ≈ blind) | that's a valid finding. Report it honestly, show *where* adaptation helps (puddles/worn tires) vs doesn't, and the teacher gap. Honest negative results beat inflated ones in interviews. |
| Can't explain the code in an interview | `EXPLAINER.md` kept current; build milestone by milestone, reading each diff |
| Overclaiming | Section 17 |

---

## 15. Debugging playbook (how to find a bug fast)

1. **Reproduce on CPU with 4 envs** and `TRUSTLAP_DEBUG=1` (`jax_debug_nans`, `jax_disable_jit` if needed).
2. **Visualize first:** `scripts/debug_rollout.py --policy X --track S --conditions C` renders a PNG strip + per-step CSV (state, obs, action, reward, μ per axle, events).
3. **Compare against the scripted driver** in the identical situation. If the scripted driver also fails, it's the environment, not the learner.
4. **Bisect parity issues** by dumping per-substep traces from both Python and TS for one case and diffing them.
5. **Training curves first, videos second:** check KL, entropy, explained variance and the curriculum before watching rollouts.
6. **Change one thing at a time,** and log it in `docs/PLAN.md`.

---

## 16. Presenting it

### 16.1 README (first screen matters most)

```
# TrustLap
AI race cars that teach themselves to drive in the rain, and know when they're unsure.

[live demo link]  [GIF of the hero: rain rising, cars adapting]

- <N> cars simulated in parallel on one GPU, <X>× faster than CPU (JAX)
- A population of <P> PPO agents with an automatic curriculum, <S>M env steps in <M> min on a free T4
- A learned adaptation module estimates road grip from 1 s of motion: crashes −<C>% vs a non-adaptive driver on puddle-heavy unseen tracks (95% CI <..>)
- Ensemble crash-risk estimate, calibrated (ECE <..>), drives a safety governor
```

Then: what it is (3 sentences), how it works (diagram), results table (from `metrics.json`), how to run (web + training), project structure, honest limitations, credits/citations, license.

### 16.2 Demo video (60–90 s, screen recording, no music needed)

0–10 s hero with rain rising. 10–25 s the swarm. 25–45 s puddle test with the grip gauge. 45–60 s ice: the risk halo appears and the car slows while the blind ghost crashes. 60–75 s Results: one chart. End card with the URL.

### 16.3 Resume bullets (fill with real numbers; never invent)

- *Built a GPU-parallel vehicle simulator in JAX (tire slip, drag, wind, puddles) running <N> cars at once, <X>× faster than CPU, with a float32-exact TypeScript port powering a live in-browser demo.*
- *Trained a population of <P> PPO driving agents with population-based training and an automatic difficulty curriculum (<S>M env steps on a free T4); distilled a privileged teacher into a student that infers road grip from 1 s of motion history (RMA), cutting crashes <C>% vs a non-adaptive baseline on unseen tracks.*
- *Added an ensemble-based crash-risk estimate (ECE <E>) driving a safety governor; evaluated 12 condition suites on 200 held-out tracks with paired bootstrap CIs.*

### 16.4 `docs/EXPLAINER.md` (for Hiten)

For every component: what it does in two sentences, why it's needed, the one equation or idea behind it, what would break without it. Then **20 likely interview questions** with short answers, e.g.:
- Why PPO and not DQN?
- What is GAE and why λ = 0.95?
- How does the student learn without being told the conditions?
- What's the difference between terminated and truncated, and why does it matter?
- How do you know the risk estimate is calibrated?
- Why JAX? What do `vmap`, `jit` and `scan` do here?
- How did you make the browser and Python physics match?
- What would you do with 10× more compute?
- What are the limitations and what would break in the real world?

---

## 17. Honesty and attribution (non-negotiable)

- Credit on the site and README: PureJaxRL (Lu et al.) for the training-loop pattern; RMA (Kumar, Fu, Pathak, Malik, 2021) for the teacher–student adaptation method; Automatic Domain Randomization (OpenAI, 2019) as inspiration for the curriculum; Population Based Training (Jaderberg et al., 2017); the Pacejka "Magic Formula" tire model.
- Describe the project as **combining and evaluating** published techniques in a new setting with a custom simulator, not inventing new algorithms.
- Every number comes from `metrics.json`/`bench.json`. If a result is weak, say so.
- Simulated only: no claims about real cars.
- The "thousands of cars" swarm is labeled "recorded from a GPU training run." The live canvases are labeled as running live in the browser.

---

## 18. Pre-launch checklist

- [ ] All tests green in CI; parity suite green.
- [ ] Every figure reads from `metrics.json`; deleting a field shows "Not computed yet."
- [ ] Headline numbers on Home, README and resume match `metrics.json` exactly.
- [ ] Lighthouse budgets met; 60 fps check on laptop; phone check on a real device.
- [ ] Keyboard-only and VoiceOver pass on Home and Playground; axe: zero serious issues.
- [ ] Screenshots reviewed at 390 / 768 / 1280 / 1920.
- [ ] OG image and favicon render (test the link in iMessage/Slack/LinkedIn preview).
- [ ] README GIF + live link; repo public; license; credits.
- [ ] `EXPLAINER.md` read end to end by Hiten.
- [ ] Resume bullets updated with real numbers.

---

## Appendix A: default constants and hyperparameters (put in `config.py`)

| Group | Name | Value |
|---|---|---|
| Time | control dt / substeps / physics dt | 0.05 s / 4 / 0.0125 s |
| Track | ds / N_MAX / width / length / min radius / clearance | 2.0 m / 2048 / 11–14 m / 900–2,600 m / 18 m / W + 8 m |
| Banks | train / validation / test seeds | 0–1999 / 100000–100099 / 200000–200199 |
| Car | m, Iz, lf, lr | 800 kg, 1,100 kg·m², 1.2 m, 1.4 m |
| Car | δ_max, δ̇_max, P, F_drive_max, F_brake_max | 0.42 rad, 3.0 rad/s, 220 kW, 7,000 N, 1.3·m·g |
| Tire | B, C, E | 9.0, 1.6, 0.95 |
| Aero | ρ, CdA nominal, C_sA_s, CoP offset | 1.225, 0.9 m², 2.5 m², 0.2 m |
| Obs | lookahead distances | 5, 10, 15, 20, 30, 40, 55, 70, 90, 110, 135, 160 m |
| Obs | sizes: policy / privileged / history | 33 / 14 / 20 × 10 |
| Reward | progress / center / smooth / lap / crash / stall | 0.1 / 0.02 / 0.01 / +10 / −10 / −2 |
| Episode | max steps | 2,400 |
| PPO | envs, T, minibatches, epochs, γ, λ, clip, ent, vf, lr, grad clip | 1024, 64, 8, 4, 0.99, 0.95, 0.2, 0.003, 0.5, 3e-4, 0.5 |
| PBT | P, ready interval, truncation fraction, perturb factors | 16, 25 updates, 25%, ×0.8 / ×1.25 (p = 0.25 resample) |
| Curriculum | check interval, up/down thresholds, step, easy fraction, boundary fraction | 10 updates, 0.8 / 0.5, 0.05, 20%, 10% |
| Student | H, lr, grad steps per rollout, T, grip loss weight, β warm start | 20, 3e-4, 4, 32, 0.5, first 10% |
| Ensemble | K, crash horizon, calibration | 5, 2 s, isotonic |
| Web | hero cars / playground cars / lab cars / mobile max | 48 / 1–5 / 16 / 16 |

## Appendix B: glossary (plain words, for EXPLAINER)

- **PPO:** a reinforcement learning method that improves the driver a little at a time, never changing it too much in one step.
- **Environment step:** one 50 ms tick of the simulated world for one car.
- **Curriculum:** making the task harder as the learner gets better.
- **PBT:** training many learners at once; the weak ones copy the strong ones and try slightly different learning settings.
- **Privileged information:** facts the simulator knows but a real car wouldn't (exact grip, mass, wind).
- **Teacher / student:** the teacher learns with privileged info; the student learns to act like the teacher using only what a real car could sense.
- **Calibration:** when the model says "20% chance of crashing," it should crash about 20% of the time.
- **Parity test:** checking that the Python and browser versions compute the same numbers.
