"""The student: rapid motor adaptation (SPEC 7.6; Kumar, Fu, Pathak, Malik, "RMA", 2021).

Phase 1 trained a teacher whose actor sees z = encoder(privileged conditions). Phase 2 (here)
trains an adaptation module that estimates z from the last second of motion: 20 steps of
[proprioception (8), action (2)] = 200 numbers. A grip head on the same hidden layer estimates
the friction under each axle (μ̂_front, μ̂_rear) for the website's grip gauge.

Training is DAgger-style: the car is driven by the frozen teacher actor fed the *student's*
estimate ẑ, so the student learns in the states it actually visits. Labels are the teacher's
z (and the true μ). During the first 10% of training, z is the true one with probability β
decaying 1 → 0, which avoids training on early garbage. K students with different seeds are
trained side by side (vmapped) for the confidence ensemble (SPEC 7.7).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from trustlap.config import CONFIG, Config, config_hash
from trustlap.curriculum import Curriculum
from trustlap.env import Env, EnvState, Observation
from trustlap.networks import ROOT2, ActorCritic, dense
from trustlap.ppo import RunningStats, normalize
from trustlap.track import TrackBank, load_bank
from trustlap.train import TrainingAbortedError, load_policy


class Adaptation(nn.Module):
    """history (200) → 256 → 128 (ELU) → ẑ (8, tanh); grip head 128 → 2, sigmoid × 1.3."""

    cfg: Any = CONFIG.network

    @nn.compact
    def __call__(self, history: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = history
        for i, width in enumerate(self.cfg.adaptation_hidden):
            h = nn.elu(dense(width, ROOT2, f"adapt_{i}")(h))
        z_hat = jnp.tanh(dense(self.cfg.latent_size, 1.0, "adapt_out")(h))
        grip = nn.sigmoid(dense(2, 1.0, "grip_out")(h)) * self.cfg.grip_scale
        return z_hat, grip


class Teacher(NamedTuple):
    params: Any
    obs_stats: RunningStats
    priv_stats: RunningStats
    difficulty: float


def load_teacher(path: Path) -> Teacher:
    pol = load_policy(path)
    if "encoder" not in pol["params"]["params"]:
        raise ValueError(f"{path} is not a teacher (no privileged encoder)")
    return Teacher(
        params=jax.tree.map(jnp.asarray, pol["params"]),
        obs_stats=RunningStats(**{k: jnp.asarray(v) for k, v in pol["obs_stats"].items()}),
        priv_stats=RunningStats(**{k: jnp.asarray(v) for k, v in pol["priv_stats"].items()}),
        difficulty=float(pol["info"].get("difficulty", 1.0)),
    )


def true_axle_grip(privileged: jax.Array, cfg: Config = CONFIG) -> jax.Array:
    """Recover raw μ_front, μ_rear from the normalized privileged vector (first two entries)."""
    from trustlap.config import privileged_scaling

    center, half = privileged_scaling(cfg)
    return privileged[..., :2] * jnp.asarray(half[:2]) + jnp.asarray(center[:2])


class StudentState(NamedTuple):
    params: Any
    opt_state: Any
    env_state: EnvState
    obs: Observation
    key: jax.Array


class StudentTrainer:
    """Train one student (vmap over seeds for the ensemble)."""

    def __init__(
        self, env: Env, teacher: Teacher, n_envs: int = 1024, cfg: Config = CONFIG
    ) -> None:
        self.env = env
        self.teacher = teacher
        self.cfg = cfg
        self.n_envs = n_envs
        self.net = Adaptation(cfg.network)
        self.teacher_net = ActorCritic(privileged_actor=True, cfg=cfg.network)
        self.optimizer = optax.adam(cfg.student.lr)

    def init(self, key: jax.Array, difficulty: jax.Array, stretch: jax.Array) -> StudentState:
        k_net, k_env, k_run = jax.random.split(key, 3)
        params = self.net.init(k_net, jnp.zeros(self.cfg.obs.history_size))
        env_state, obs = self.env._reset(
            jax.random.split(k_env, self.n_envs), difficulty, stretch, self.env.bank
        )
        return StudentState(params, self.optimizer.init(params), env_state, obs, k_run)

    def _teacher_z(self, privileged: jax.Array) -> jax.Array:
        t = self.teacher
        return self.teacher_net.apply(
            t.params, normalize(t.priv_stats, privileged), method=ActorCritic.encode
        )

    def _act(self, obs: jax.Array, z: jax.Array) -> jax.Array:
        t = self.teacher
        mean = self.teacher_net.apply(
            t.params, normalize(t.obs_stats, obs), z, method=ActorCritic.act
        )
        return jnp.clip(mean, -1.0, 1.0)

    def update(
        self,
        s: StudentState,
        difficulty: jax.Array,
        stretch: jax.Array,
        beta: jax.Array,
        bank: TrackBank,
    ) -> tuple[StudentState, dict[str, jax.Array]]:
        """Roll out T steps (teacher actor fed ẑ, or z with probability β), then 4 Adam steps."""
        sc = self.cfg.student

        def step(carry: StudentState, _: Any) -> tuple[StudentState, tuple]:
            st = carry
            key, k_mix, k_env = jax.random.split(st.key, 3)
            history = st.obs.history
            z_hat, grip_hat = self.net.apply(st.params, history)
            z = self._teacher_z(st.obs.privileged)
            use_true = jax.random.uniform(k_mix, (self.n_envs, 1)) < beta
            action = self._act(st.obs.policy, jnp.where(use_true, z, z_hat))
            env_state, obs, _, info, _ = self.env._step(
                st.env_state,
                action,
                jax.random.split(k_env, self.n_envs),
                difficulty,
                stretch,
                bank,
            )
            mu = true_axle_grip(st.obs.privileged, self.cfg)
            done = info.terminated | info.truncated
            stats = {
                "episodes": jnp.sum(done),
                "laps": jnp.sum(info.lap_completed),
                "crashes": jnp.sum(info.crashed),
                "nan_resets": jnp.sum(info.nan_reset),
                "grip_abs_err": jnp.mean(jnp.abs(grip_hat - mu)),
            }
            return st._replace(env_state=env_state, obs=obs, key=key), ((history, z, mu), stats)

        s, ((hist, z, mu), stats) = jax.lax.scan(step, s, None, length=sc.rollout_len)
        hist = hist.reshape(-1, hist.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        mu = mu.reshape(-1, 2)

        def loss_fn(params: Any) -> tuple[jax.Array, dict]:
            z_hat, grip_hat = self.net.apply(params, hist)
            z_loss = jnp.mean((z_hat - z) ** 2)
            grip_loss = jnp.mean((grip_hat - mu) ** 2)
            return z_loss + sc.grip_loss_weight * grip_loss, {
                "z_loss": z_loss,
                "grip_loss": grip_loss,
            }

        def grad_step(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], dict]:
            params, opt_state = carry
            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), aux

        (params, opt_state), aux = jax.lax.scan(
            grad_step, (s.params, s.opt_state), None, length=sc.grad_steps_per_rollout
        )
        metrics = {
            **{k: jnp.sum(v) for k, v in stats.items() if k != "grip_abs_err"},
            "grip_mae": jnp.mean(stats["grip_abs_err"]),
            "z_loss": aux["z_loss"][0],  # before this update's steps: an honest on-policy number
            "grip_loss": aux["grip_loss"][0],
            "params_finite": jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)])
            ),
        }
        return s._replace(params=params, opt_state=opt_state), metrics


def save_students(path: Path, params: Any, info: dict[str, Any]) -> None:
    """All K students' adaptation params (leading axis K) + training info."""
    payload = {
        "params": jax.tree.map(np.asarray, serialization.to_state_dict(params)),
        "info": json.dumps(info),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.msgpack_serialize(payload))


def load_students(path: Path) -> dict[str, Any]:
    payload = serialization.msgpack_restore(path.read_bytes())
    payload["info"] = json.loads(payload["info"])
    return payload


def train_students(
    teacher_path: Path,
    out: Path,
    env_steps: float,
    members: int = CONFIG.ensemble.members,
    n_envs: int = 512,
    seed: int = 0,
    bank_path: Path = Path("runs/banks/train.npz"),
    cfg: Config = CONFIG,
) -> Path:
    """Train `members` students (different seeds) against one frozen teacher."""
    out.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(teacher_path)
    env = Env(load_bank(bank_path), cfg)
    trainer = StudentTrainer(env, teacher, n_envs, cfg)
    sc = cfg.student
    steps_per_update = n_envs * sc.rollout_len
    total = max(1, int(env_steps // steps_per_update))
    warm = int(sc.warm_start_frac * total)
    cur = Curriculum(cfg.curriculum, difficulty=teacher.difficulty)
    d_np, s_np = cur.env_arrays(n_envs)
    d, st = jnp.asarray(d_np), jnp.asarray(s_np)
    init = jax.jit(jax.vmap(trainer.init, in_axes=(0, None, None)))
    state = init(jax.random.split(jax.random.PRNGKey(seed), members), d, st)
    update = jax.jit(jax.vmap(trainer.update, in_axes=(0, None, None, None, None)))
    log = (out / "log.jsonl").open("w")
    t0 = time.time()
    for u in range(total):
        beta = max(0.0, 1.0 - u / max(1, warm)) if warm > 0 else 0.0
        state, m = update(state, d, st, jnp.float32(beta), env.bank)
        m = {k: np.asarray(v) for k, v in m.items()}
        if not m["params_finite"].all():
            raise TrainingAbortedError(f"student update {u}: non-finite parameters")
        eps = np.maximum(m["episodes"], 1)
        row = {
            "update": u + 1,
            "env_steps_per_member": (u + 1) * steps_per_update,
            "wall_s": round(time.time() - t0, 1),
            "beta": round(beta, 4),
            "z_loss": m["z_loss"].round(5).tolist(),
            "grip_loss": m["grip_loss"].round(5).tolist(),
            "grip_mae": m["grip_mae"].round(4).tolist(),
            "completion": (m["laps"] / eps).round(4).tolist(),
            "crash": (m["crashes"] / eps).round(4).tolist(),
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (u + 1) % 25 == 0 or u == 0:
            print(
                f"[student×{members}] update {u + 1}/{total} β={beta:.2f} "
                f"z_loss {np.mean(row['z_loss']):.4f} "
                f"grip MAE {np.mean(row['grip_mae']):.3f} laps {np.mean(row['completion']):.2f} "
                f"crash {np.mean(row['crash']):.2f}",
                flush=True,
            )
        if (u + 1) % 100 == 0 or u + 1 == total:
            info = {
                "updates": u + 1,
                "env_steps_per_member": (u + 1) * steps_per_update,
                "members": members,
                "teacher": str(teacher_path),
                "difficulty": teacher.difficulty,
                "config_hash": config_hash(cfg),
            }
            save_students(out / "students.msgpack", state.params, info)
    log.close()
    print(
        f"done: {total} updates × {members} students, {(time.time() - t0) / 60:.1f} min", flush=True
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-steps", type=float, required=True, help="per student")
    parser.add_argument("--members", type=int, default=CONFIG.ensemble.members)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_students(args.teacher, args.out, args.env_steps, args.members, args.envs, args.seed)


if __name__ == "__main__":
    main()
