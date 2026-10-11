"""Population-based training (SPEC 7.3; Jaderberg et al., 2017).

Members share an architecture; each has its own parameters, optimizer state, observation
statistics, hyperparameters and PRNG stream, stacked on a leading axis. Every 25 updates the
bottom 25% (by fitness = mean progress per second) copy a random top-25% member and perturb
its learning rate and entropy coefficient (×0.8 or ×1.25; with probability 0.25, resample
from the prior). Every copy gets a fresh id, so the history is a family tree (`Lineage`).
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from trustlap.config import CONFIG, Config
from trustlap.ppo import TrainState

HP_NAMES = ("lr", "ent")


def _ranges(cfg: Config) -> dict[str, tuple[float, float]]:
    return {"lr": cfg.pbt.lr_range, "ent": cfg.pbt.ent_range}


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, size: int | None = None) -> Any:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def default_hyperparams(cfg: Config = CONFIG) -> dict[str, np.ndarray]:
    """A single agent with the spec's default learning rate and entropy coefficient."""
    return {
        "lr": np.array([cfg.ppo.lr], np.float32),
        "ent": np.array([cfg.ppo.ent_coef], np.float32),
    }


def initial_hyperparams(
    population: int, rng: np.random.Generator, cfg: Config = CONFIG
) -> dict[str, np.ndarray]:
    """Log-uniform draws from the PBT priors."""
    return {
        k: _log_uniform(rng, *r, population).astype(np.float32) for k, r in _ranges(cfg).items()
    }


class Lineage:
    """Member ids per slot, exploit/explore events and periodic fitness snapshots."""

    def __init__(self, population: int) -> None:
        self.ids = np.arange(population)
        self.next_id = population
        self.events: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": self.ids.tolist(),
            "next_id": self.next_id,
            "events": self.events,
            "snapshots": self.snapshots,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lineage:
        lin = cls(len(d["ids"]))
        lin.ids = np.asarray(d["ids"])
        lin.next_id = d["next_id"]
        lin.events = d["events"]
        lin.snapshots = d["snapshots"]
        return lin


def copy_member(ts: TrainState, src: int, dst: int) -> TrainState:
    """Copy the learnable state of member `src` into slot `dst` (not its envs or PRNG stream)."""

    def cp(tree: Any) -> Any:
        return jax.tree.map(lambda x: x.at[dst].set(x[src]), tree)

    return ts._replace(
        params=cp(ts.params),
        opt_state=cp(ts.opt_state),
        obs_stats=cp(ts.obs_stats),
        priv_stats=cp(ts.priv_stats),
        ret_stats=cp(ts.ret_stats),
    )


def exploit_explore(
    ts: TrainState,
    hp: dict[str, np.ndarray],
    fitness: np.ndarray,
    lineage: Lineage,
    update: int,
    env_steps: int,
    rng: np.random.Generator,
    cfg: Config = CONFIG,
) -> tuple[TrainState, dict[str, np.ndarray], list[dict[str, Any]]]:
    """One PBT round. Returns the new state, hyperparameters and the logged events."""
    p = len(fitness)
    hp = {k: v.copy() for k, v in hp.items()}
    lineage.snapshots.append(
        {
            "update": update,
            "env_steps": env_steps,
            "members": [
                {
                    "id": int(lineage.ids[i]),
                    "fitness": float(fitness[i]),
                    "lr": float(hp["lr"][i]),
                    "ent": float(hp["ent"][i]),
                }
                for i in range(p)
            ],
        }
    )
    n_trunc = max(1, int(round(cfg.pbt.truncation_frac * p)))
    order = np.argsort(fitness)
    bottom, top = order[:n_trunc], order[-n_trunc:]
    events = []
    for dst in bottom:
        src = int(rng.choice(top))
        if src == dst:
            continue
        before = {k: float(hp[k][dst]) for k in HP_NAMES}
        ts = copy_member(ts, src, int(dst))
        for k, (lo, hi) in _ranges(cfg).items():
            if rng.random() < cfg.pbt.resample_prob:
                hp[k][dst] = _log_uniform(rng, lo, hi)
            else:
                hp[k][dst] = np.clip(hp[k][src] * rng.choice(cfg.pbt.perturb_factors), lo, hi)
        prev_id = int(lineage.ids[dst])
        lineage.ids[dst] = lineage.next_id
        lineage.next_id += 1
        event = {
            "update": update,
            "env_steps": env_steps,
            "slot": int(dst),
            "child_id": int(lineage.ids[dst]),
            "parent_id": int(lineage.ids[src]),
            "prev_id": prev_id,
            "hp_before": before,
            "hp_after": {k: float(hp[k][dst]) for k in HP_NAMES},
            "fitness_parent": float(fitness[src]),
            "fitness_child_before": float(fitness[dst]),
        }
        lineage.events.append(event)
        events.append(event)
    return ts, hp, events
