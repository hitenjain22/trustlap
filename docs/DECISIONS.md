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
