"""Evaluation rollouts: one lap per car from the start line, all cars in parallel (SPEC 6.7, 8.3).

No auto-reset: each car runs until its episode ends (lap, crash, stall, wrong way) or the
2,400-step time limit, then freezes. Used by the scripted-driver check now and `evaluate.py` later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import Conditions
from trustlap.config import CONFIG, Config
from trustlap.env import Env, EnvState, Observation, core_step, observe
from trustlap.track import TrackBank

# policy(state [one car], obs [one car], bank) -> action [2]
PolicyFn = Callable[[EnvState, Observation, TrackBank], jax.Array]


class LapResult(NamedTuple):
    """Per-car outcome of one evaluation lap (numpy arrays, [N])."""

    completed: np.ndarray  # bool
    crashed: np.ndarray  # bool (off track or spun)
    stalled: np.ndarray  # bool
    wrong_way: np.ndarray  # bool
    timed_out: np.ndarray  # bool
    lap_time_s: np.ndarray  # float, NaN unless completed
    progress_m: np.ndarray
    mean_speed_mps: np.ndarray


def run_laps(
    env: Env,
    track_ids: np.ndarray,
    conds: Conditions,
    policy: PolicyFn,
    seed: int = 0,
    cfg: Config = CONFIG,
) -> tuple[LapResult, dict[str, np.ndarray]]:
    """Drive one lap per car. Returns results and a trace {x, y, speed, done} of shape [T, N]."""
    n = len(track_ids)
    key = jax.random.key(seed)
    state, obs = env.eval_reset(jax.random.split(key, n), jnp.asarray(track_ids, jnp.int32), conds)
    bank = env.bank
    obs_size = cfg.obs.proprio_size + 2 * len(cfg.obs.lookahead_m)

    def one_car(state: EnvState, obs: Observation, done: Any, k: jax.Array) -> tuple[Any, ...]:
        k_gust, k_obs = jax.random.split(k)
        action = policy(state, obs, bank)
        new_state, _, info = core_step(
            state, action, jax.random.normal(k_gust, (2,)), bank, cfg, jnp
        )
        new_obs, sensed = observe(new_state, bank, jax.random.normal(k_obs, (obs_size,)), cfg, jnp)
        new_state = new_state._replace(last_proprio=sensed)
        keep = lambda new, old: jnp.where(done, old, new)  # noqa: E731
        return jax.tree.map(keep, new_state, state), jax.tree.map(keep, new_obs, obs), info

    @jax.jit
    def run(state: EnvState, obs: Observation, key: jax.Array) -> tuple[Any, ...]:
        def body(carry: tuple[Any, ...], k: jax.Array) -> tuple[tuple[Any, ...], dict[str, Any]]:
            state, obs, done, outcome, end_step, speed_sum = carry
            new_state, new_obs, info = jax.vmap(one_car)(state, obs, done, jax.random.split(k, n))
            ending = ~done & (info.terminated | info.truncated)
            code = jnp.select(
                [info.lap_completed, info.crashed, info.stalled, info.wrong_way, info.truncated],
                [1, 2, 3, 4, 5],
                0,
            )
            outcome = jnp.where(ending, code, outcome)
            end_step = jnp.where(ending, info.steps, end_step)
            speed_sum = speed_sum + jnp.where(done, 0.0, info.speed_mps)
            done = done | ending
            trace = {
                "x": new_state.car.x_m,
                "y": new_state.car.y_m,
                "speed": info.speed_mps,
                "done": done,
            }
            return (new_state, new_obs, done, outcome, end_step, speed_sum), trace

        zeros_i = jnp.zeros(n, jnp.int32)
        init = (state, obs, jnp.zeros(n, bool), zeros_i, zeros_i, jnp.zeros(n))
        (state, _, _, outcome, end_step, speed_sum), trace = jax.lax.scan(
            body, init, jax.random.split(key, cfg.episode.max_steps)
        )
        return state, outcome, end_step, speed_sum, trace

    final, outcome, end_step, speed_sum, trace = run(state, obs, jax.random.fold_in(key, 1))
    outcome, end_step = np.asarray(outcome), np.asarray(end_step)
    completed = outcome == 1
    dt = cfg.time.control_dt_s
    result = LapResult(
        completed=completed,
        crashed=outcome == 2,
        stalled=outcome == 3,
        wrong_way=outcome == 4,
        timed_out=outcome == 5,
        lap_time_s=np.where(completed, end_step * dt, np.nan),
        progress_m=np.asarray(final.progress_m),
        mean_speed_mps=np.asarray(speed_sum) / np.maximum(end_step, 1),
    )
    return result, {k: np.asarray(v) for k, v in trace.items()}
