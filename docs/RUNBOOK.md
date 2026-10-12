# Runbook

How to set up, test, train, evaluate, export and deploy. Sections are filled in as milestones land.

## Local setup

### Python (`sim/`)

```bash
cd sim
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
```

Known-good local versions (Intel Mac, CPU): jax 0.4.38, jaxlib 0.4.38, flax 0.10.4, optax 0.2.5, distrax 0.1.5, chex 0.1.90. Intel Macs can't get newer JAX; see `DECISIONS.md`.

### Web (`web/`)

Node 24 LTS. On Hiten's Intel Mac it's installed at `~/.local/share/node` (no Homebrew), so first:

```bash
export PATH="$HOME/.local/share/node/bin:$PATH"   # add to ~/.zshrc to make it permanent
```

Then:

```bash
cd web
npm ci
npm run dev                                          # http://localhost:5173
npm run typecheck && npm run lint && npm test && npm run build
npx playwright install chromium                      # once
npm run test:e2e
```

## Deploy (Vercel)

One-time setup (needs Hiten's accounts):

1. Push the repo to GitHub (`gh repo create trustlap --public --source . --push`, or create it in the GitHub UI and `git remote add origin ... && git push -u origin main`).
2. In Vercel: **Add New → Project → Import** the GitHub repo.
3. Set **Root Directory** to `web`. Framework preset: Vite. Build command `npm run build`, output `dist` (the defaults).
4. Deploy. Every push to `main` deploys production; every PR gets a preview URL.

`web/vercel.json` holds the SPA rewrite, cache headers for `/artifacts/` and `/fonts/`, and the security headers (SPEC 13).

Without GitHub, from the terminal: `npx vercel` inside `web/` (log in when prompted), then `npx vercel --prod`.

## Simulator tools (M1)

All from `sim/` with the venv active. Outputs go to `sim/runs/` (gitignored).

```bash
python -m trustlap.track --out runs/banks              # train/validation/test banks (~80 s, 4 cores)
python scripts/eval_scripted.py --bank runs/banks/validation.npz --seeds 3   # scripted lap completion
python scripts/debug_plot.py --track-seed 200000 --rain 0.8 --out runs/debug_lap.png   # one lap: PNG + CSV
python -m trustlap.bench --out runs/bench.json         # throughput (GPU rows only on a GPU machine)
```

## GPU benchmark on Colab (M1)

1. Open https://colab.research.google.com → **File → Open notebook → GitHub** tab → paste `hitenjain22/trustlap` → pick `sim/notebooks/bench_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** If Colab warns the notebook isn't from Google, click **Run anyway**.
4. Wait ~10–15 min. The last cell downloads `bench.json` to your computer.
5. Put the file at `sim/runs/bench.json` (or hand it to Claude Code). It records the device, CPU and package versions itself. Keep GPU results by copying them to `sim/results/` (committed; `runs/` is not).

Known-good Colab setup: Tesla T4 (15 GB), Colab's preinstalled JAX **0.11.1** with CUDA, `pip install -e sim` on top (2026-10-11, commit `c0a2e73`). Full benchmark took ~25 min with the old CPU sizes; ~10 min now.

## Training

### Overnight CPU run (what's published now)

```bash
cd sim && source .venv/bin/activate
python -m trustlap.train --agent blind --env-steps 40e6 --out runs/blind --seed 1          # ~45 min
python -m trustlap.train --agent teacher --population 4 --env-steps 20e6 --out runs/teacher --seed 2   # ~70 min
python -m trustlap.student --teacher runs/teacher/final.msgpack --out runs/students --env-steps 8e6 --members 5 --envs 512 --seed 3   # ~25 min
python -m trustlap.evaluate --blind runs/blind/final.msgpack --teacher runs/teacher/final.msgpack \
    --students runs/students/students.msgpack --seeds 2 --out runs/eval                    # ~20 min
```

### Full-size GPU run on Kaggle (recommended next step)

1. Go to https://www.kaggle.com → sign in → **Create → New Notebook**.
2. **File → Import Notebook → GitHub** → paste `hitenjain22/trustlap` → pick `sim/notebooks/train_full.ipynb` → **Import**.
3. Right panel → **Session options**: **Accelerator → GPU T4 ×2** (or P100), **Internet → On** (needs a phone-verified Kaggle account).
4. Top right: **Save Version** → **Save & Run All (Commit)** → **Save**. It now runs in the background (you can close the tab). Kaggle allows ~12 h per run and ~30 GPU h per week.
5. When it finishes (the version shows "Successful"), open it → **Output** tab → download **`trustlap_full.zip`**.
6. Unzip into `sim/runs/` (giving `sim/runs/full/...`), then from `sim/`:

```bash
python -m trustlap.export --policy blind=runs/full/blind/final.msgpack --policy teacher=runs/full/teacher/final.msgpack \
    --students runs/full/students/students.msgpack --teacher runs/full/teacher/final.msgpack \
    --teacher-run runs/full/teacher --blind-run runs/full/blind --metrics runs/full/eval/metrics_partial.json \
    --train-hardware "NVIDIA T4 GPU (Kaggle)" --golden --closed-loop blind
cd ../web && npm test && npm run build      # parity + budgets must pass
```

7. Update the model-card compute line (`web/src/copy/strings.ts`, `how.card`) if it names hardware (`train_hardware` comes from `--train-hardware`), commit, push. If it disconnects, run all again: every stage resumes from its checkpoint.

## Publishing results

```bash
python -m trustlap.export --policy blind=runs/blind/final.msgpack --policy teacher=runs/teacher/final.msgpack \
    --students runs/students/students.msgpack --teacher runs/teacher/final.msgpack \
    --teacher-run runs/teacher --blind-run runs/blind --metrics runs/eval/metrics_partial.json \
    --train-hardware "laptop CPU (Intel, 4 threads) with JAX 0.4.38" --golden --closed-loop blind
```

This writes every artifact (weights, training records, metrics, calibration), regenerates the parity fixtures, refreshes the manifest and rewrites the README's numbers block from `metrics.json`.

## Export to the website (M2)

From `sim/` with the venv active:

```bash
python -m trustlap.export            # web/public/artifacts/v1/: config, showcase tracks, bench, manifest
python -m trustlap.export --golden   # also regenerate the parity fixtures in web/tests/parity/golden/
```

Run `--golden` after **any** change to physics, tracks, observations, reward, episode rules or the scripted driver; `sim/tests/test_export.py` fails until you do. Then run `npm test` in `web/` to confirm the TypeScript port still matches.

## Evaluation

Filled in at M6.

## README GIF

```bash
cd web && npm run build && npm run preview &        # http://localhost:4173
node scripts/readme-gif.mjs http://localhost:4173 ../docs/media/hero.gif   # needs uv (uvx) for Pillow
```

## Demo video

```bash
cd web && npm run build && npm run preview &        # http://localhost:4173
node scripts/demo-video.mjs http://localhost:4173 ~/Downloads/trustlap-demo.webm
```

About 85 s at 1280×720: hero with rain rising, Training lab timeline, puddle test, black ice with the governor and a blind ghost, Results. The captions describe what's on screen, not outcomes. The video is not committed (~14 MB); upload it and link it from the README.

## Replication run (3 training seeds, equal practice)

Kaggle: import `sim/notebooks/train_full.ipynb` (it runs `scripts/replicate_pipeline.sh`), GPU T4, Save & Run All; download `trustlap_rep.zip` (~4–5 h). Then, from `sim/`:

```bash
mkdir -p runs/rep_import && cd runs/rep_import && unzip -q ~/Downloads/trustlap_rep.zip && cd ../..
mv runs/rep_import/rep runs/rep && rmdir runs/rep_import
python -m trustlap.export --policy blind=runs/rep/seed1/blind/final.msgpack \
    --policy teacher=runs/rep/seed1/teacher/final.msgpack \
    --students runs/rep/seed1/students/students.msgpack --teacher runs/rep/seed1/teacher/final.msgpack \
    --teacher-run runs/rep/seed1/teacher --blind-run runs/rep/seed1/blind \
    --metrics runs/rep/metrics_partial.json --train-hardware "NVIDIA T4 GPU (Kaggle)" \
    --golden --closed-loop blind
cd ../web && npm test && npm run build && npm run test:e2e
```

Seed 1 is published (networks, charts); `metrics.json → replication` holds every seed's verdicts.
