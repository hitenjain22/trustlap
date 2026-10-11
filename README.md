# TrustLap

AI race cars that teach themselves to drive in the rain, and know when they're unsure.

Work in progress. The build spec is in [`docs/SPEC.md`](docs/SPEC.md); progress is tracked milestone by milestone in [`docs/PLAN.md`](docs/PLAN.md).

## Layout

- `sim/`: Python + JAX simulator and training (runs offline on a GPU).
- `web/`: Vite + React + TypeScript site that runs the trained networks live in the browser.
- `docs/`: spec, plans, decisions, runbook, plain-English explainer.

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) to run it locally.

## License

MIT
