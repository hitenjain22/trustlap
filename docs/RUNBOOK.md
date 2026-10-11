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

## Training (Colab / Kaggle)

Filled in from M1 (benchmark) and M3 onward.

## Evaluation and export

Filled in from M2 (export v0) and M6 (evaluation).
