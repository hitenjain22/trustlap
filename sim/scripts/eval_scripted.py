"""Lap completion of the scripted driver on a track bank (SPEC M1 acceptance).

python scripts/eval_scripted.py --bank runs/banks/validation.npz --seeds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import suite_conditions
from trustlap.env import Env
from trustlap.rollout import run_laps
from trustlap.scripted import scripted_action
from trustlap.track import load_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("runs/banks/validation.npz"))
    parser.add_argument("--suites", nargs="+", default=["dry", "wet", "puddles"])
    parser.add_argument("--seeds", type=int, default=3, help="condition seeds per track")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON summary path")
    args = parser.parse_args()

    bank = load_bank(args.bank)
    env = Env(bank)
    n_tracks = len(bank.seeds)
    track_ids = np.repeat(np.arange(n_tracks), args.seeds)
    policy = lambda state, obs, bank: scripted_action(state, bank, xp=jnp)  # noqa: E731
    summary = {}
    for i, suite in enumerate(args.suites):
        keys = jax.random.split(jax.random.key(args.seed * 1000 + i), len(track_ids))
        lengths = jnp.asarray(bank.length)[track_ids]
        widths = jnp.asarray(bank.width)[track_ids]
        conds = jax.vmap(
            lambda k, length, width, suite=suite: suite_conditions(k, suite, length, width)
        )(keys, lengths, widths)
        res, _ = run_laps(env, track_ids, conds, policy, seed=args.seed)
        row = {
            "laps": int(len(track_ids)),
            "completion": float(res.completed.mean()),
            "crash": float(res.crashed.mean()),
            "stall": float(res.stalled.mean()),
            "wrong_way": float(res.wrong_way.mean()),
            "timeout": float(res.timed_out.mean()),
            "lap_time_s_median": float(np.nanmedian(res.lap_time_s))
            if res.completed.any()
            else None,
            "lap_time_s_max": float(np.nanmax(res.lap_time_s)) if res.completed.any() else None,
            "mean_speed_mps": float(res.mean_speed_mps.mean()),
        }
        summary[suite] = row
        print(
            f"{suite:8s} "
            + "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
        )
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
