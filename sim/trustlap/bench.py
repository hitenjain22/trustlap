"""Environment throughput benchmark: GPU JAX vs CPU JAX vs a plain NumPy loop (SPEC M1.7).

One "env step" = one 50 ms control step for one car: 4 physics substeps, progress tracking,
reward, termination and all observations (policy, privileged, history).

* `jax-<device>`: the batched training env (`Env.step`, includes auto-reset), random actions,
  100 steps fused in one `lax.scan`, timed after compilation.
* `numpy-loop`: the same single-car code (`core_step` + `observe` with xp=numpy) called in a
  Python loop over cars. Only measured up to 1,024 cars; nothing is extrapolated.

    python -m trustlap.bench --out bench.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import sample_conditions
from trustlap.config import CONFIG
from trustlap.env import Env, EnvState, core_step, observe, start_state
from trustlap.track import TrackBank, generate_track, make_bank

JAX_SIZES = (1, 64, 1024, 4096, 16384, 65536)
CPU_SIZES = (1, 64, 1024, 4096)  # CPU throughput plateaus by ~1k cars; bigger sizes only cost time
NUMPY_SIZES = (1, 64, 1024)
PACKAGES = ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "optax", "distrax", "chex", "numpy")
SCAN_STEPS = 100


def _median_time(fn: Any, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_jax(env: Env, n_cars: int, device: jax.Device, repeats: int = 5) -> float:
    """Env steps per second for `n_cars` cars on `device`."""
    with jax.default_device(device):
        bank = jax.device_put(env.bank, device)
        keys = jax.random.split(jax.random.key(0), n_cars)
        difficulty = jnp.ones(n_cars)
        state, _ = env._reset(keys, difficulty, bank)

        @jax.jit
        def run(state: EnvState, key: jax.Array) -> EnvState:
            def body(s: EnvState, k: jax.Array) -> tuple[EnvState, None]:
                ka, ks = jax.random.split(k)
                actions = jax.random.uniform(ka, (n_cars, 2), minval=-1.0, maxval=1.0)
                s, *_ = env._step(s, actions, jax.random.split(ks, n_cars), difficulty, bank)
                return s, None

            return jax.lax.scan(body, state, jax.random.split(key, SCAN_STEPS))[0]

        jax.block_until_ready(run(state, jax.random.key(1)))  # compile + warm up
        seconds = _median_time(
            lambda: jax.block_until_ready(run(state, jax.random.key(2))), repeats
        )
    return n_cars * SCAN_STEPS / seconds


def bench_numpy(bank: TrackBank, n_cars: int, steps: int) -> float:
    """Env steps per second for a Python loop over cars calling the NumPy implementation."""
    keys = jax.random.split(jax.random.key(0), n_cars)
    states = []
    for i in range(n_cars):
        tid = i % len(bank.width)
        cond = jax.tree.map(
            np.asarray, sample_conditions(keys[i], 1.0, bank.length[tid], bank.width[tid])
        )
        states.append(
            start_state(bank, np.int32(tid), np.float32(0.0), np.float32(10.0), cond, 1.0)
        )
    rng = np.random.default_rng(0)
    obs_noise = np.zeros(CONFIG.obs.policy_size - 1, np.float32)
    gust = np.zeros(2, np.float32)

    t0 = time.perf_counter()
    for _ in range(steps):
        actions = rng.uniform(-1, 1, (n_cars, 2)).astype(np.float32)
        for i in range(n_cars):
            state, _, _ = core_step(states[i], actions[i], gust, bank)
            _, sensed = observe(state, bank, obs_noise)
            states[i] = state._replace(last_proprio=sensed)
    return n_cars * steps / (time.perf_counter() - t0)


def _cpu_name() -> str:
    try:  # Linux (Colab, Kaggle, CI)
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """CLI: run every implementation and write bench.json."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("bench.json"))
    parser.add_argument("--jax-sizes", type=int, nargs="+", default=list(JAX_SIZES))
    parser.add_argument("--cpu-sizes", type=int, nargs="+", default=list(CPU_SIZES))
    parser.add_argument("--numpy-sizes", type=int, nargs="+", default=list(NUMPY_SIZES))
    parser.add_argument("--numpy-steps", type=int, default=5)
    parser.add_argument("--tracks", type=int, default=32, help="tracks in the benchmark bank")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bank_np = make_bank([generate_track(args.seed * 1000 + i) for i in range(args.tracks)])
    env = Env(bank_np)
    default = jax.devices()[0]
    cpu = jax.devices("cpu")[0]
    rows: list[dict[str, Any]] = []
    out = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "device": getattr(default, "device_kind", default.platform),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "jax_version": jax.__version__,
        "packages": _package_versions(),
        "step_definition": (
            "one 50 ms control step for one car: 4 physics substeps, progress, reward, observations"
        ),
        "rows": rows,
    }

    def record(impl: str, n: int, sps: float) -> None:
        rows.append({"impl": impl, "n_cars": n, "steps_per_s": round(sps, 1)})
        print(f"{impl:14s} {n:7d} cars  {sps:14,.0f} steps/s", flush=True)
        args.out.write_text(json.dumps(out, indent=2))  # save after every row: long runs can die

    if default.platform != "cpu":
        for n in args.jax_sizes:
            record(f"jax-{default.platform}", n, bench_jax(env, n, default))
    for n in args.cpu_sizes:
        record("jax-cpu", n, bench_jax(env, n, cpu, repeats=3))
    for n in args.numpy_sizes:
        record("numpy-loop", n, bench_numpy(bank_np, n, args.numpy_steps))

    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
