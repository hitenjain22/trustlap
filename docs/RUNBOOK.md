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
5. Put the file at `sim/runs/bench.json` (or hand it to Claude Code), and paste the output of cell 3 (package versions) into the "Known-good Colab versions" list below.

Known-good Colab versions: *(fill in after the first run)*

## Training (Colab / Kaggle)

Filled in from M3 onward.

## Evaluation and export

Filled in from M2 (export v0) and M6 (evaluation).
