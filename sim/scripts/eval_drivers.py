"""Paired lap evaluation of trained policies vs the scripted driver (M3/M4 acceptance checks).

Every driver sees identical (track, condition seed) pairs. Prints completion and crash rates
with 95% bootstrap CIs (resampling tracks) and writes a JSON summary.

    python scripts/eval_drivers.py --policy blind=runs/blind/final.msgpack --difficulty 0 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import sample_conditions
from trustlap.env import Env
from trustlap.networks import ActorCritic
from trustlap.ppo import RunningStats, policy_action
from trustlap.rollout import run_laps
from trustlap.scripted import scripted_action
from trustlap.track import load_bank
from trustlap.train import load_policy


def policy_fn(path: Path):  # noqa: ANN201
    pol = load_policy(path)
    net = ActorCritic(privileged_actor="encoder" in pol["params"]["params"])
    obs_stats = RunningStats(**{k: jnp.asarray(v) for k, v in pol["obs_stats"].items()})
    priv_stats = RunningStats(**{k: jnp.asarray(v) for k, v in pol["priv_stats"].items()})
    return lambda state, obs, bank: policy_action(net, pol["params"], obs_stats, priv_stats, obs)


def bootstrap_ci(per_track: np.ndarray, rng: np.random.Generator, n: int = 2000) -> list[float]:
    """95% CI of the mean by resampling tracks (rows) with replacement."""
    idx = rng.integers(0, len(per_track), (n, len(per_track)))
    means = per_track[idx].mean(axis=(1, 2)) if per_track.ndim == 2 else per_track[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bank", type=Path, default=Path("runs/banks/validation.npz"))
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--difficulty", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--seeds", type=int, default=3, help="condition seeds per track")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    env = Env(bank)
    n_tracks = len(bank.seeds)
    track_ids = np.repeat(np.arange(n_tracks), args.seeds)
    drivers = {"scripted": lambda state, obs, bank: scripted_action(state, bank, xp=jnp)}
    for item in args.policy:
        name, path = item.split("=", 1)
        drivers[name] = policy_fn(Path(path))
    rng = np.random.default_rng(args.seed)
    summary: dict = {"bank": str(args.bank), "laps_per_driver": int(len(track_ids)), "results": {}}
    for d in args.difficulty:
        keys = jax.random.split(jax.random.key(args.seed + int(d * 1000)), len(track_ids))
        lengths = jnp.asarray(bank.length)[track_ids]
        widths = jnp.asarray(bank.width)[track_ids]
        conds = jax.vmap(lambda k, length, width, d=d: sample_conditions(k, d, length, width))(
            keys, lengths, widths
        )
        row = {}
        for name, fn in drivers.items():
            res, _ = run_laps(env, track_ids, conds, fn, seed=args.seed)
            comp = res.completed.reshape(n_tracks, args.seeds).astype(float)
            crash = res.crashed.reshape(n_tracks, args.seeds).astype(float)
            row[name] = {
                "completion": float(comp.mean()),
                "completion_ci": bootstrap_ci(comp, rng),
                "crash": float(crash.mean()),
                "crash_ci": bootstrap_ci(crash, rng),
                "lap_time_s_median": float(np.nanmedian(res.lap_time_s))
                if res.completed.any()
                else None,
                "mean_speed_mps": float(res.mean_speed_mps.mean()),
            }
            r = row[name]
            ci = r["completion_ci"]
            print(
                f"d={d:.2f} {name:9s} completion {r['completion']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
                f"  crash {r['crash']:.3f}  lap {r['lap_time_s_median']}"
                f"  speed {r['mean_speed_mps']:.1f}",
                flush=True,
            )
        summary["results"][f"{d:.2f}"] = row
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
