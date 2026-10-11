# Decisions

## Core stack (SPEC 3.2)

| Decision | Choice | Why | Rejected alternative |
|---|---|---|---|
| Sim + training language | Python + **JAX** (with Flax Linen, Optax, Distrax, Chex) | `vmap` + `jit` + `lax.scan` let the whole env and PPO run on GPU with tens of thousands of cars; PureJaxRL proves the pattern | PyTorch (fine, but vectorized envs + training loop fusion are clunkier); Brax/MJX (3D rigid bodies, overkill) |
| Vehicle model | Dynamic bicycle model with simplified Pacejka tires, friction ellipse, drag, rolling resistance, wind, steering lag, action delay | Real tire-slip behaviour (drifting, understeer in rain) at tiny compute cost; textbook model you can explain in an interview | Kinematic model (no slip, so adaptation is pointless); full 4-wheel model (slow, harder to port) |
| Policy size | MLP **3 × 128**, ELU | Enough capacity for this task; small enough for live browser inference for ~64 cars at 20 Hz in plain TypeScript | 256-wide (4× slower in browser); RNN policy (harder to port and train) |
| Adaptation module | MLP over flattened history (H = 20 steps) | Simplest to train and to port; RMA works with this | 1D conv / GRU (more to port, little gain at this scale) |
| Browser inference | Hand-written MLP in TypeScript over `Float32Array`s, in a Web Worker | Zero dependencies, tiny bundle, exact control for parity tests | onnxruntime-web / TF.js (heavy bundles; threading needs COOP/COEP headers) |
| Rendering | **Canvas 2D** with batched paths, `devicePixelRatio` aware | Hundreds of cars + rain particles at 60 fps with zero dependencies | PixiJS/WebGL (big dependency, unnecessary at this scale); SVG (too slow for many moving objects) |
| Charts | Hand-built SVG React components using `d3-scale`, `d3-shape`, `d3-array`, `d3-hierarchy` (math only, React renders) | Full control to meet the chart spec in SPEC 11.6 | Recharts/Chart.js (fighting defaults to hit the spec) |
| Web framework | **Vite + React + TypeScript (strict)** + React Router | Fast, boring, static output | Next.js (static-export edge cases, not needed) |
| Styling | Tailwind CSS with design tokens defined as CSS variables (SPEC 11) | Speed + consistency | Unstyled CSS modules (slower to build consistently) |
| State | Zustand store + URL query params for shareable state | Tiny, simple | Redux (overkill) |
| Worker RPC | Comlink | Clean typed worker calls | Hand-rolled postMessage protocol |
| Schema validation | zod (web), pydantic or dataclasses + manual checks (python) | Fail loudly on bad artifacts | None |
| Tests | pytest (sim), Vitest (web unit + parity), Playwright (e2e + screenshots) | Standard | — |
| Hosting | Vercel static (free), artifacts served from `/artifacts/v1/` with long cache headers | Free, fast, preview deploys | GitHub Pages (base-path quirks) |
| Checkpointing | `flax.serialization` msgpack (or Orbax) to Google Drive / Kaggle output every N updates | Colab disconnects | Keeping everything in RAM |

## Later decisions

### 2026-10-10 (M0)

- **Tailwind v4 instead of v3 (`tailwind.config.ts` → CSS `@theme`).** v4 is the current release; it configures through CSS (`@import "tailwindcss"` + `@theme`) and a first-party Vite plugin, so there is no `tailwind.config.ts`. The design tokens in SPEC 11.2 still live as CSS variables in `web/src/styles/tokens.css`, which is the spirit of the spec. Rejected: pinning v3 just to match the filename.
- **`ruff format` instead of black.** Ruff's formatter is black-compatible and we already depend on ruff for linting: one tool, no extra dependency.
- **Local JAX is 0.4.38 on Intel Mac.** Google stopped publishing macOS x86_64 jaxlib wheels after 0.4.38. Local dev (CPU tests, golden-file generation on CPU) runs on that version; CI (Linux) and Colab/Kaggle install the latest. Consequence: write code against APIs present in both (avoid features newer than jax 0.4.38 / flax 0.10). Pin exact versions after the first working Colab install (SPEC M0.3) and record them in `RUNBOOK.md`. If a needed API is missing locally, develop in a Linux container or Colab instead of bumping the floor.
- **Node 24 LTS**, installed user-locally at `~/.local/share/node` (no Homebrew on this machine). CI uses `node-version: lts/*`.
- **Dev dependencies beyond the spec list:** `@tailwindcss/vite` (Tailwind v4's build integration), `typescript-eslint` + `eslint-plugin-react-hooks` + `eslint-plugin-react-refresh` (the Vite template's lint set), `eslint-config-prettier` (stops eslint and prettier fighting), `jsdom` (DOM for Vitest component tests).

### 2026-10-11 (M1)

Spec changes (each one fixes something in the spec that was wrong or ambiguous; flagged to Hiten at the M1 checkpoint):

- **Curriculum interpolation (SPEC 6.2).** "nominal ± d · half-width, clipped to the full range" never reaches the full range when the nominal isn't the range's center: tire wear would top out at 0.125 (not 0.25), wind at 5 m/s (not 10), μ_dry would never go below 1.005. Now each bound moves linearly: `lo(d) = nominal + d·(lo − nominal)`, `hi(d) = nominal + d·(hi − nominal)`. Identical at d = 0 (the nominal point) and d = 1 (the full range). `config.Range.at_difficulty`.
- **Generated tracks get the drawn-track auto-fix (SPEC 6.1).** With the spec's parameters only 18% of first attempts pass the 18 m minimum-radius check (median tightest corner 11.6 m), so seeds needed many retries and some would eventually hit the 50-attempt limit. Generation now applies the spec's own "Auto-fix" (Laplacian-smooth the control points, λ = 0.3, up to 10 passes) before giving up on an attempt. Measured on 300 seeds: 93% valid on the first sub-seed; 90% of tracks still have a corner tighter than 30 m (median tightest 23 m); median length 1,074 m (was 1,235 m). The full banks needed at most 3 attempts. Rejected: loosening the radius limit (changes the car/track contract) or raising max_attempts (keeps the 82% waste and still risks a raise).
- **Closed-loop resampling (SPEC 6.1.6).** An exact 2.0 m spacing can't tile a loop of arbitrary length. `N = round(L / 2)`, spacing `ds = L / N` (within 0.1% of 2 m). The bank stores `length[T]` so `ds` is exact per track.
- **Curvature definition (6.1.7).** Raw κ_i is the signed angle from tangent i to tangent i+1 over ds, then a 5-point circular moving average (order of summation fixed for the TS port).

Pinned interpretations (the spec allowed more than one reading):

- **Action buffer `[4, 2]`** holds the 4 *previous* actions, newest first. D = 0 executes the current action; D = k executes `buffer[k − 1]`. Covers the OOD delay of 4.
- **Drive and brake hardware are fixed** at the nominal-mass values (`F_drive_max = 7,000 N`, `F_brake_max = 1.3 · 800 · g`), so a heavier car accelerates and brakes worse. Mass and yaw inertia scale with the mass multiplier.
- **Low-speed blending runs right after the velocity update and before the position update**, within each substep, so positions integrate the blended velocities.
- **Gust noise is an input** to `dynamics.control_step` (a standard-normal pair), keeping the step deterministic for golden files. The env draws it from `jax.random`.
- **Puddle arclengths are measured from the episode's start**, so a car never spawns inside one.
- **Spin and stall timers are integer step counters** (crash after the 21st consecutive step beyond 100°, stall after the 61st below 1 m/s) to avoid float drift in the "> 1 s" / "> 3 s" checks.
- **Wrong way** terminates with no penalty beyond the negative progress reward (the spec's termination table gives it none).
- **PCG32** uses the reference seeding with stream 54 (so seed 42 reproduces the reference demo's first outputs); `next_int` uses the modulo method; `splitmix32` is the 32-bit SplitMix variant (constants in `rng.py`).

Implementation choices:

- **One single-car implementation, two array backends.** `dynamics.py`, the core of `env.py` and `scripted.py` take an `xp` argument (`numpy` or `jax.numpy`). JAX runs them `vmap`ped on GPU/CPU; the benchmark's "naive NumPy loop" calls the same source per car. No second implementation to keep in sync.
- **Scripted driver (SPEC M1.6).** Pure pursuit alone (0.5 s lookahead) oscillated into spins; the car model is neutral-steer (identical normalized tire curves front and rear) and rear-wheel drive, so throttle in a corner causes power oversteer. The driver now uses: a 1 s lookahead (8–40 m) with yaw-rate damping; a corner-speed plan at 85% of true grip with braking planned at 45% of μ·g; a speed controller that requests an acceleration (3 m/s² per m/s of error) converted through the car's force limits; and throttle/brakes capped to 80% of the grip that cornering leaves free (friction circle). Tuned on the validation bank only. Result: 100% dry / 99.1% wet completion on all 2,000 training tracks.
- **Observation (not changed):** the specified car is close to neutral steer and quick to power-oversteer. That's realistic for a light rear-drive race car and makes throttle control part of the learning problem. If PPO struggles with it in M3, the conventional fix is a slightly stiffer rear tire (B_rear > B_front) for mild understeer; not done now because nothing requires it.
- **`rollout.py`** (new module): one-lap-per-car evaluation with no auto-reset, shared by the scripted check now and `evaluate.py` in M6.
