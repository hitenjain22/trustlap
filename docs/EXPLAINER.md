# TrustLap explained

Plain-English notes on every part of the project, written for Hiten to read after each milestone. Interviewers will ask about all of it, so the goal is that every section can be retold in your own words.

Each component gets: what it does (two sentences), why it's needed, the one equation or idea behind it, and what would break without it. The 20 interview questions come at the end once the components exist.

---

## The project layout (M0)

**What it is.** Two halves that never run at the same time. `sim/` is a Python package that simulates cars and trains the AI drivers offline on a GPU (Colab/Kaggle). `web/` is a static website that loads the trained networks as plain files and runs them in your browser.

**Why it's split like this.** Training needs a GPU and hours of compute; the website needs to be free to host and instant to load. So training produces files (weights, metrics, configs), and the website only reads files. There's no server doing work when someone visits.

**The one idea.** "Every number on the website comes from a file a script produced." The website never contains a hand-typed result. If the file is missing a number, the page says "Not computed yet."

**What would break without it.** If constants were copied by hand between Python and TypeScript, the browser car would drive slightly differently from the trained one, and the AI would crash in the browser for reasons no one could see. That's why `config.py` is the single source of truth, and why parity tests exist (coming in M2).

**Tooling, in one line each.**
- **JAX:** NumPy that runs on a GPU, compiles functions (`jit`), and turns a one-car function into a many-car function (`vmap`).
- **Vite + React + TypeScript:** the website's build tool, UI library and typed JavaScript.
- **Vitest / Playwright / pytest:** unit tests for the web, browser tests that click through the real site, and Python tests.
- **CI (GitHub Actions):** runs all the tests on every push, so a broken change is caught before it's deployed.
- **Vercel:** hosts the static site for free.
