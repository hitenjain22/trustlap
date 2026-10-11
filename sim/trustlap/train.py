"""Training loop for a population of PPO agents with the automatic curriculum (SPEC 7.2–7.5).

The host loop calls one jitted, population-`vmap`ped PPO update per iteration, steps the
curriculum, logs to `runs/<name>/log.jsonl`, raises health alarms, checkpoints, and (for
P > 1) runs population-based training through `pbt.py`.

    python -m trustlap.train --agent blind --env-steps 40e6 --out runs/blind
    python -m trustlap.train --agent teacher --population 4 --env-steps 50e6 --out runs/teacher
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from trustlap.config import CONFIG, Config, config_hash
from trustlap.curriculum import Curriculum
from trustlap.env import Env
from trustlap.ppo import PPO, TrainState
from trustlap.track import load_bank


class TrainingAbortedError(RuntimeError):
    """Raised on NaNs or a physics bug (SPEC 2 rule 10: fail loudly)."""


@dataclasses.dataclass
class Alarms:
    """Health alarms from SPEC 7.2, printed loudly and logged."""

    high_kl_streak: int = 0

    def check(self, update: int, m: dict[str, float]) -> list[str]:
        out = []
        self.high_kl_streak = self.high_kl_streak + 1 if m["approx_kl"] > 0.05 else 0
        if self.high_kl_streak >= 5:
            out.append(f"approx KL {m['approx_kl']:.3f} > 0.05 for {self.high_kl_streak} updates")
        if update >= 20 and m["explained_variance"] < 0:
            out.append(f"explained variance {m['explained_variance']:.2f} < 0")
        if update < 200 and m["log_std_mean"] < -3:
            out.append(f"log std {m['log_std_mean']:.2f} collapsing early")
        if not math.isfinite(m["value_loss"]) or not m["params_finite"]:
            out.append("non-finite loss or parameters")
        return out


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_checkpoint(path: Path, ts: TrainState, extra: dict[str, Any]) -> None:
    """Everything needed to resume: train state (incl. env state and keys), curriculum, PBT."""
    payload = {"train_state": serialization.to_state_dict(ts), "extra": json.dumps(extra)}
    _atomic_write(path, serialization.msgpack_serialize(payload))


def load_checkpoint(path: Path, template: TrainState) -> tuple[TrainState, dict[str, Any]]:
    payload = serialization.msgpack_restore(path.read_bytes())
    ts = serialization.from_state_dict(template, payload["train_state"])
    return ts, json.loads(payload["extra"])


def save_policy(path: Path, ts: TrainState, member: int, info: dict[str, Any]) -> None:
    """One member's deployable pieces: params + observation statistics (msgpack)."""
    take = lambda x: np.asarray(x[member])  # noqa: E731
    payload = {
        "params": jax.tree.map(take, serialization.to_state_dict(ts.params)),
        "obs_stats": jax.tree.map(take, ts.obs_stats._asdict()),
        "priv_stats": jax.tree.map(take, ts.priv_stats._asdict()),
        "info": json.dumps(info),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, serialization.msgpack_serialize(payload))


def load_policy(path: Path) -> dict[str, Any]:
    payload = serialization.msgpack_restore(path.read_bytes())
    payload["info"] = json.loads(payload["info"])
    return payload


def train(
    agent: str,
    out: Path,
    env_steps: float,
    population: int = 1,
    n_envs: int | None = None,
    seed: int = 0,
    bank_path: Path = Path("runs/banks/train.npz"),
    checkpoint_every: int = 25,
    snapshots: int = 12,
    pbt_enabled: bool | None = None,
    cfg: Config = CONFIG,
) -> Path:
    """Train `population` agents of kind `agent` ("blind" or "teacher"). Returns the run dir."""
    out.mkdir(parents=True, exist_ok=True)
    privileged_actor = agent == "teacher"
    env = Env(load_bank(bank_path), cfg)
    ppo = PPO(env, privileged_actor=privileged_actor, n_envs=n_envs, cfg=cfg)
    n = ppo.n_envs
    steps_per_update = n * ppo.rollout_len
    total_updates = max(1, int(env_steps // steps_per_update))
    pbt_enabled = population > 1 if pbt_enabled is None else pbt_enabled

    from trustlap import pbt  # local: PBT is only needed for populations

    cur = Curriculum(cfg.curriculum)
    rng = np.random.default_rng(seed)
    hp = (
        pbt.initial_hyperparams(population, rng, cfg)
        if population > 1
        else pbt.default_hyperparams(cfg)
    )
    lineage = pbt.Lineage(population)

    d_np, s_np = cur.env_arrays(n)
    keys = jax.random.split(jax.random.PRNGKey(seed), population)
    init = jax.jit(jax.vmap(ppo.init, in_axes=(0, None, None)))
    ts = init(keys, jnp.asarray(d_np), jnp.asarray(s_np))
    update_fn = jax.jit(jax.vmap(ppo.update, in_axes=(0, None, None, 0, 0, None)))

    ckpt = out / "checkpoint.msgpack"
    start = 0
    meta = {"agent": agent, "population": population, "n_envs": n, "config_hash": config_hash(cfg)}
    if ckpt.exists():
        ts, extra = load_checkpoint(ckpt, ts)
        if extra["config_hash"] != meta["config_hash"]:
            raise TrainingAbortedError(
                f"{ckpt} was made with a different config; refusing to resume"
            )
        start = extra["update"]
        cur.load_state_dict(extra["curriculum"])
        hp = {k: np.asarray(v, np.float32) for k, v in extra["hp"].items()}
        lineage = pbt.Lineage.from_dict(extra["lineage"])
        print(f"resumed from update {start}")
    (out / "meta.json").write_text(json.dumps({**meta, "total_updates": total_updates}, indent=1))

    alarms = Alarms()
    log = (out / "log.jsonl").open("a")
    fitness_window: list[np.ndarray] = []
    snapshot_at = {round((i + 1) * total_updates / snapshots) for i in range(snapshots)}
    t_start = time.time()
    for u in range(start, total_updates):
        t0 = time.perf_counter()
        d_np, s_np = cur.env_arrays(n)
        lr = hp["lr"] * (1.0 - u / total_updates)  # linear decay to 0 (SPEC 7.2)
        ts, m = update_fn(
            ts,
            jnp.asarray(d_np),
            jnp.asarray(s_np),
            jnp.asarray(lr),
            jnp.asarray(hp["ent"]),
            env.bank,
        )
        m = {k: np.asarray(v) for k, v in m.items()}  # [P] each
        dt = time.perf_counter() - t0

        if (m["nan_resets"].sum() > 0.001 * max(1.0, m["episodes"].sum())) and u > 5:
            raise TrainingAbortedError(
                f"update {u}: {m['nan_resets'].sum()} NaN resets; physics bug, not masking it"
            )
        if not m["params_finite"].all():
            raise TrainingAbortedError(f"update {u}: non-finite parameters")

        cur.record(float(m["cur_episodes"].sum()), float(m["cur_success"].sum()))
        cur.step()

        eps = np.maximum(m["episodes"], 1)
        # Fitness: mean progress per second over finished episodes (SPEC 7.3).
        fitness = m["progress_sum"] / np.maximum(m["length_sum"] * cfg.time.control_dt_s, 1e-6)
        fitness_window.append(fitness)
        pbt_events: list[dict] = []
        if pbt_enabled and (u + 1) % cfg.pbt.ready_interval_updates == 0:
            mean_fit = np.mean(fitness_window, axis=0)
            ts, hp, pbt_events = pbt.exploit_explore(
                ts, hp, mean_fit, lineage, u + 1, (u + 1) * steps_per_update, rng, cfg
            )
            fitness_window.clear()

        best = int(np.argmax(fitness))
        row = {
            "update": u + 1,
            "env_steps": (u + 1) * steps_per_update * population,
            "env_steps_per_member": (u + 1) * steps_per_update,
            "wall_s": round(time.time() - t_start, 1),
            "sps": round(steps_per_update * population / dt),
            "difficulty": cur.difficulty,
            "curriculum_success": None
            if math.isnan(cur.last_success)
            else round(cur.last_success, 4),
            "return": (m["return_sum"] / eps).round(3).tolist(),
            "completion": (m["laps"] / eps).round(4).tolist(),
            "crash": (m["crashes"] / eps).round(4).tolist(),
            "stall": (m["stalls"] / eps).round(4).tolist(),
            "episode_len": (m["length_sum"] / eps).round(1).tolist(),
            "mean_speed": (m["speed_sum"] / (n * ppo.rollout_len)).round(2).tolist(),
            "fitness": fitness.round(3).tolist(),
            "approx_kl": m["approx_kl"].round(5).tolist(),
            "clip_frac": m["clip_frac"].round(4).tolist(),
            "entropy": m["entropy"].round(3).tolist(),
            "value_loss": m["value_loss"].round(4).tolist(),
            "explained_variance": m["explained_variance"].round(3).tolist(),
            "lr": np.asarray(lr).round(7).tolist(),
            "ent_coef": np.asarray(hp["ent"]).round(6).tolist(),
            "nan_resets": int(m["nan_resets"].sum()),
            "member_ids": lineage.ids.tolist(),
            "pbt_events": pbt_events,
        }
        warnings = alarms.check(u, {k: float(np.mean(v)) for k, v in m.items()})
        if warnings:
            row["alarms"] = warnings
            print(f"!!! ALARM update {u + 1}: " + "; ".join(warnings), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (u + 1) % 5 == 0 or u == start:
            print(
                f"[{agent}] update {u + 1}/{total_updates} steps {row['env_steps'] / 1e6:.1f}M "
                f"sps {row['sps']} d={cur.difficulty:.2f} ret {np.mean(row['return']):.2f} "
                f"laps {np.mean(row['completion']):.2f} crash {np.mean(row['crash']):.2f} "
                f"speed {np.mean(row['mean_speed']):.1f} kl {np.mean(row['approx_kl']):.4f} "
                f"ev {np.mean(row['explained_variance']):.2f}",
                flush=True,
            )
        if (u + 1) in snapshot_at:
            info = {
                "update": u + 1,
                "env_steps": row["env_steps_per_member"],
                "difficulty": cur.difficulty,
                "return": row["return"][best],
                "completion": row["completion"][best],
                "member_id": int(lineage.ids[best]),
                "member_index": best,
            }
            save_policy(out / "snapshots" / f"update_{u + 1:05d}.msgpack", ts, best, info)
        if (u + 1) % checkpoint_every == 0 or u + 1 == total_updates:
            extra = {
                **meta,
                "update": u + 1,
                "curriculum": cur.state_dict(),
                "hp": {k: np.asarray(v).tolist() for k, v in hp.items()},
                "lineage": lineage.to_dict(),
            }
            save_checkpoint(ckpt, ts, extra)
            (out / "lineage.json").write_text(json.dumps(lineage.to_dict()))
    log.close()
    final_fit = np.mean(fitness_window, axis=0) if fitness_window else fitness
    best = int(np.argmax(final_fit))
    save_policy(
        out / "final.msgpack",
        ts,
        best,
        {
            "member_index": best,
            "member_id": int(lineage.ids[best]),
            "difficulty": cur.difficulty,
            "updates": total_updates,
        },
    )
    for i in range(population):
        save_policy(
            out / "members" / f"member_{i:02d}.msgpack",
            ts,
            i,
            {"member_index": i, "member_id": int(lineage.ids[i])},
        )
    print(
        f"done: {total_updates} updates, "
        f"{total_updates * steps_per_update * population / 1e6:.1f}M env steps, "
        f"{(time.time() - t_start) / 60:.1f} min; final difficulty {cur.difficulty:.2f}",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", choices=["blind", "teacher"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-steps", type=float, required=True, help="per population member")
    parser.add_argument("--population", type=int, default=1)
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank", type=Path, default=Path("runs/banks/train.npz"))
    parser.add_argument("--no-pbt", action="store_true")
    args = parser.parse_args()
    train(
        args.agent,
        args.out,
        args.env_steps,
        args.population,
        args.envs,
        args.seed,
        args.bank,
        pbt_enabled=False if args.no_pbt else None,
    )


if __name__ == "__main__":
    main()
