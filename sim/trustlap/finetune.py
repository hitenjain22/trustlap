"""Fine-tune the driving network on the students' own estimate (after RMA phase 2).

The Adaptive AI drives with the teacher's driving network fed the students' estimated latent ẑ.
That network was only ever trained on the true latent z, so estimation errors (largest on dry
roads and in wind) turn straight into driving errors. Here PPO continues training the driving
network while it acts on ẑ = mean of the frozen students' estimates; the critic still sees the
privileged observation (asymmetric critic). Same curriculum mix as the students
(teacher's final difficulty, 20% easy cars, 10% stretch), lower learning rate.

Output: a policy file in the teacher's format whose actor is used with the students.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from trustlap.config import CONFIG, Config, config_hash
from trustlap.curriculum import Curriculum
from trustlap.env import Env, Observation
from trustlap.ppo import PPO
from trustlap.student import Adaptation, load_students, load_teacher
from trustlap.track import load_bank
from trustlap.train import TrainingAbortedError, _atomic_write


def student_latent_fn(students_path: Path, cfg: Config = CONFIG) -> Any:
    """obs → mean ẑ over the frozen student ensemble, [N, latent]."""
    params = jax.tree.map(jnp.asarray, load_students(students_path)["params"])
    adapt = Adaptation(cfg.network)

    def latent(obs: Observation) -> jax.Array:
        z_k = jax.vmap(lambda p: adapt.apply(p, obs.history)[0])(params)  # [K, N, latent]
        return jnp.mean(z_k, axis=0)

    return latent


def finetune(
    teacher_path: Path,
    students_path: Path,
    out: Path,
    env_steps: float,
    n_envs: int = 1024,
    seed: int = 0,
    bank_path: Path = Path("runs/banks/train.npz"),
    cfg: Config = CONFIG,
) -> Path:
    """PPO on the teacher's actor with the students' latent. Writes out/actor.msgpack."""
    out.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(teacher_path)
    env = Env(load_bank(bank_path), cfg)
    ppo = PPO(
        env,
        privileged_actor=True,
        n_envs=n_envs,
        latent_fn=student_latent_fn(students_path, cfg),
        cfg=cfg,
    )
    cur = Curriculum(cfg.curriculum, difficulty=teacher.difficulty)
    d_np, s_np = cur.env_arrays(n_envs)
    d, st = jnp.asarray(d_np), jnp.asarray(s_np)
    ts = ppo.init(jax.random.PRNGKey(seed), d, st)
    ts = ts._replace(
        params=teacher.params,
        opt_state=ppo.optimizer.init(teacher.params),
        obs_stats=teacher.obs_stats,
        priv_stats=teacher.priv_stats,
    )
    update = jax.jit(ppo.update)
    lr = jnp.float32(cfg.finetune.lr)
    ent = jnp.float32(cfg.finetune.ent_coef)
    steps_per_update = n_envs * ppo.rollout_len
    total = max(1, int(env_steps // steps_per_update))
    log = (out / "log.jsonl").open("w")
    t0 = time.time()
    for u in range(total):
        ts, m = update(ts, d, st, lr, ent, env.bank)
        m = {k: np.asarray(v) for k, v in m.items()}
        if not bool(m["params_finite"]):
            raise TrainingAbortedError(f"fine-tune update {u}: non-finite parameters")
        eps = max(float(m["episodes"]), 1.0)
        row = {
            "update": u + 1,
            "env_steps": (u + 1) * steps_per_update,
            "wall_s": round(time.time() - t0, 1),
            "completion": round(float(m["laps"]) / eps, 4),
            "crash": round(float(m["crashes"]) / eps, 4),
            "approx_kl": round(float(m["approx_kl"]), 6),
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (u + 1) % 25 == 0 or u == 0:
            print(
                f"[finetune] update {u + 1}/{total} laps {row['completion']:.2f} "
                f"crash {row['crash']:.2f} kl {row['approx_kl']:.4f}",
                flush=True,
            )
    log.close()
    info = {
        "updates": total,
        "env_steps": total * steps_per_update,
        "teacher": str(teacher_path),
        "students": str(students_path),
        "difficulty": teacher.difficulty,
        "config_hash": config_hash(cfg),
        "kind": "student_actor",
    }
    payload = {
        "params": jax.tree.map(np.asarray, serialization.to_state_dict(ts.params)),
        "obs_stats": jax.tree.map(np.asarray, ts.obs_stats._asdict()),
        "priv_stats": jax.tree.map(np.asarray, ts.priv_stats._asdict()),
        "info": json.dumps(info),
    }
    _atomic_write(out / "actor.msgpack", serialization.msgpack_serialize(payload))
    print(f"done: {total} updates, {(time.time() - t0) / 60:.1f} min → {out / 'actor.msgpack'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-steps", type=float, required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank", type=Path, default=Path("runs/banks/train.npz"))
    args = parser.parse_args()
    finetune(args.teacher, args.students, args.out, args.env_steps, args.envs, args.seed, args.bank)


if __name__ == "__main__":
    main()
