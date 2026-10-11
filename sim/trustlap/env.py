"""The driving environment (SPEC 6.4–6.7): progress tracking, observations, reward, termination.

Two layers:

* `core_step` / `observe`: one car, deterministic given its inputs, written against an array
  module `xp` (NumPy or jax.numpy) like `dynamics.py`. The TypeScript port mirrors these, the
  golden files test them, and the NumPy benchmark loops over them.
* `Env`: the batched JAX environment used for training. It adds randomness (resets, observation
  noise, gusts) and PureJaxRL-style auto-reset, and `vmap`s + `jit`s everything.

Coordinates: world frame for positions; observations are in the car frame (x forward, y left).
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import Conditions, puddle_grip, sample_conditions
from trustlap.config import CONFIG, Config, privileged_scaling
from trustlap.dynamics import (
    CarState,
    axle_grip,
    control_step,
    initial_car_state,
    tire_grip_factor,
    wrap_angle,
)
from trustlap.track import TrackBank


class EnvState(NamedTuple):
    """Everything the environment tracks for one car."""

    car: CarState
    cond: Conditions
    track_id: Any  # [] int32 index into the bank
    idx: Any  # [] int32 nearest centerline point
    s_m: Any  # [] arclength position along the centerline, in [0, L)
    progress_m: Any  # [] signed distance driven along the track since the episode started
    steps: Any  # [] int32 control steps this episode
    spin_steps: Any  # [] int32 consecutive steps with |heading error| > 100°
    stall_steps: Any  # [] int32 consecutive steps below 1 m/s
    prev_action: Any  # [2] the policy's previous action (clipped)
    last_proprio: Any  # [8] proprioception the policy last observed (with noise)
    history: Any  # [H, 10] last H rows of [proprio, action], oldest first, zero-padded
    history_len: Any  # [] int32 valid rows in `history`
    difficulty: Any  # [] curriculum difficulty this episode was sampled at


class StepInfo(NamedTuple):
    """Per-step diagnostics (all [] unless noted)."""

    delta_s_m: Any
    lap_completed: Any  # bool
    crashed: Any  # bool: off track or spun
    stalled: Any  # bool
    wrong_way: Any  # bool
    terminated: Any  # bool: any of the above (no bootstrap)
    truncated: Any  # bool: time limit (bootstrap from V(s'))
    speed_mps: Any
    mu_front: Any
    mu_rear: Any
    steps: Any  # int32, steps this episode including this one
    progress_m: Any


class Observation(NamedTuple):
    """What each kind of network gets to see."""

    policy: Any  # [33]
    privileged: Any  # [14] teacher/critic only
    history: Any  # [200] student only


# ---------------------------------------------------------------------------------------------
# Track geometry helpers (one car)
# ---------------------------------------------------------------------------------------------


def _track_scalars(bank: TrackBank, tid: Any, xp: ModuleType) -> tuple[Any, Any, Any]:
    n = bank.n_points[tid]
    length = bank.length[tid]
    return n, length, length / n.astype(length.dtype)


def wrap_s(s: Any, length: Any, xp: ModuleType = np) -> Any:
    """Wrap an arclength into [0, L)."""
    return s - length * xp.floor(s / length)


def point_at(bank: TrackBank, tid: Any, s: Any, xp: ModuleType = np) -> tuple[Any, Any]:
    """Centerline point [..., 2] and unit tangent [..., 2] at arclength s (linear interpolation)."""
    n, length, ds = _track_scalars(bank, tid, xp)
    u = wrap_s(s, length, xp) / ds
    i0f = xp.floor(u)
    f = u - i0f
    i0 = i0f.astype(xp.int32) % n
    i1 = (i0 + 1) % n
    c0, c1 = bank.centerline[tid, i0], bank.centerline[tid, i1]
    t0, t1 = bank.tangent[tid, i0], bank.tangent[tid, i1]
    f2 = f[..., None]
    point = c0 + f2 * (c1 - c0)
    tangent = t0 + f2 * (t1 - t0)
    tangent = tangent / xp.sqrt(xp.sum(tangent * tangent, axis=-1, keepdims=True))
    return point, tangent


def curvature_at(bank: TrackBank, tid: Any, s: Any, xp: ModuleType = np) -> Any:
    """Centerline curvature (1/m) at arclength s (linear interpolation)."""
    n, length, ds = _track_scalars(bank, tid, xp)
    u = wrap_s(s, length, xp) / ds
    i0f = xp.floor(u)
    f = u - i0f
    i0 = i0f.astype(xp.int32) % n
    k0, k1 = bank.curvature[tid, i0], bank.curvature[tid, (i0 + 1) % n]
    return k0 + f * (k1 - k0)


def locate(
    bank: TrackBank, tid: Any, x: Any, y: Any, idx: Any, cfg: Config = CONFIG, xp: ModuleType = np
) -> tuple[Any, Any]:
    """Nearest centerline index within [idx − 5, idx + 25] and the projected arclength.

    The window keeps the tracker from jumping to a different part of the track that happens
    to be close by (SPEC 6.5). Returns (new idx, s in [0, L)).
    """
    ep = cfg.episode
    n, length, ds = _track_scalars(bank, tid, xp)
    offsets = xp.arange(-ep.search_back, ep.search_ahead + 1)
    cand = (idx + offsets) % n
    pts = bank.centerline[tid, cand]
    dx = pts[:, 0] - x
    dy = pts[:, 1] - y
    i = cand[xp.argmin(dx * dx + dy * dy)]

    # Project onto the segment after i; if the car is behind point i, onto the one before.
    def project(a_idx: Any) -> tuple[Any, Any]:
        a = bank.centerline[tid, a_idx]
        b = bank.centerline[tid, (a_idx + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        t = ((x - a[0]) * ex + (y - a[1]) * ey) / (ex * ex + ey * ey)
        return t, a_idx

    t_next, _ = project(i)
    i_prev = (i - 1 + n) % n
    t_prev, _ = project(i_prev)
    use_prev = t_next < 0.0
    seg = xp.where(use_prev, i_prev, i)
    t = xp.clip(xp.where(use_prev, t_prev, t_next), 0.0, 1.0)
    s = wrap_s((seg.astype(t.dtype) + t) * ds, length, xp)
    return i, s


def lateral_and_heading_error(
    bank: TrackBank, tid: Any, s: Any, x: Any, y: Any, yaw: Any, xp: ModuleType = np
) -> tuple[Any, Any]:
    """Signed lateral offset e_y (m, + = left of centerline) and heading error e_ψ (rad)."""
    point, tangent = point_at(bank, tid, s, xp)
    e_y = -(x - point[0]) * tangent[1] + (y - point[1]) * tangent[0]
    e_psi = wrap_angle(yaw - xp.arctan2(tangent[1], tangent[0]), xp)
    return e_y, e_psi


# ---------------------------------------------------------------------------------------------
# Observations (one car)
# ---------------------------------------------------------------------------------------------


def proprioception(
    state: EnvState, bank: TrackBank, cfg: Config = CONFIG, xp: ModuleType = np
) -> Any:
    """[8] vx/30, vy/10, r/2, δ/δmax, prev a_steer, prev a_long, e_y/(W/2), e_ψ/π (no noise)."""
    o = cfg.obs
    car = state.car
    width = bank.width[state.track_id]
    e_y, e_psi = lateral_and_heading_error(
        bank, state.track_id, state.s_m, car.x_m, car.y_m, car.yaw_rad, xp
    )
    return xp.stack(
        [
            car.vx_mps / o.vx_scale_mps,
            car.vy_mps / o.vy_scale_mps,
            car.yaw_rate_radps / o.yaw_rate_scale_radps,
            car.steer_rad / cfg.car.steer_max_rad,
            state.prev_action[0],
            state.prev_action[1],
            e_y / (width / 2.0),
            e_psi / math.pi,
        ]
    )


def lookahead(state: EnvState, bank: TrackBank, cfg: Config = CONFIG, xp: ModuleType = np) -> Any:
    """[24] centerline points 5 … 160 m ahead in the car frame, divided by 100, as (x, y) pairs."""
    car = state.car
    dists = xp.asarray(cfg.obs.lookahead_m, dtype=car.x_m.dtype)
    pts, _ = point_at(bank, state.track_id, state.s_m + dists, xp)
    dx = pts[:, 0] - car.x_m
    dy = pts[:, 1] - car.y_m
    c, s = xp.cos(car.yaw_rad), xp.sin(car.yaw_rad)
    local = xp.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)
    return xp.reshape(local / cfg.obs.lookahead_scale_m, (-1,))


def puddle_min_ahead(
    state: EnvState, bank: TrackBank, cfg: Config = CONFIG, xp: ModuleType = np
) -> Any:
    """Lowest puddle grip multiplier at 0/10/20/30 m ahead on the car's current lateral line."""
    car = state.car
    e_y, _ = lateral_and_heading_error(
        bank, state.track_id, state.s_m, car.x_m, car.y_m, car.yaw_rad, xp
    )
    dists = xp.asarray(cfg.obs.puddle_ahead_m, dtype=car.x_m.dtype)
    pts, tangents = point_at(bank, state.track_id, state.s_m + dists, xp)
    probe_x = pts[:, 0] - e_y * tangents[:, 1]
    probe_y = pts[:, 1] + e_y * tangents[:, 0]
    grips = xp.stack(
        [
            puddle_grip(xp.stack([probe_x[k], probe_y[k]]), state.cond, xp)
            for k in range(len(cfg.obs.puddle_ahead_m))
        ]
    )
    return xp.min(grips)


def privileged(state: EnvState, bank: TrackBank, cfg: Config = CONFIG, xp: ModuleType = np) -> Any:
    """[14] true conditions, normalized to ~[−1, 1] over the training ranges (teacher, critic)."""
    car, cond = state.car, state.cond
    mu_f, mu_r = axle_grip(car, cond, cfg, xp)
    wind_x = cond.wind_mps[0] + car.gust_mps[0]
    wind_y = cond.wind_mps[1] + car.gust_mps[1]
    c, s = xp.cos(car.yaw_rad), xp.sin(car.yaw_rad)
    raw = xp.stack(
        [
            mu_f,
            mu_r,
            cond.mu_base,
            tire_grip_factor(car.odometer_m, cond, cfg, xp),
            puddle_min_ahead(state, bank, cfg, xp),
            cond.mass_mult,
            cond.drag_area_m2,
            cond.rolling_coef,
            cond.power_mult,
            cond.brake_mult,
            cond.steer_tau_s,
            cond.delay_steps.astype(car.x_m.dtype),
            c * wind_x + s * wind_y,
            -s * wind_x + c * wind_y,
        ]
    )
    center, half = privileged_scaling(cfg)
    return (raw - xp.asarray(center, dtype=raw.dtype)) / xp.asarray(half, dtype=raw.dtype)


def observe(
    state: EnvState, bank: TrackBank, noise: Any, cfg: Config = CONFIG, xp: ModuleType = np
) -> tuple[Observation, Any]:
    """All observations for one car. `noise` [32] is standard normal (zeros for none), scaled by
    the episode's sensor-noise σ and added to proprioception + lookahead.

    Returns (observation, noisy proprioception) — the latter goes into the history buffer.
    """
    width = bank.width[state.track_id]
    sensed = xp.concatenate([proprioception(state, bank, cfg, xp), lookahead(state, bank, cfg, xp)])
    sensed = sensed + state.cond.obs_noise * noise
    policy = xp.concatenate([sensed, xp.reshape(width / cfg.obs.width_scale_m, (1,))])
    obs = Observation(
        policy=policy,
        privileged=privileged(state, bank, cfg, xp),
        history=xp.reshape(state.history, (-1,)),
    )
    return obs, sensed[: cfg.obs.proprio_size]


# ---------------------------------------------------------------------------------------------
# Reset and step (one car)
# ---------------------------------------------------------------------------------------------


def place_puddles(
    cond: Conditions, bank: TrackBank, tid: Any, start_s: Any, xp: ModuleType = np
) -> Conditions:
    """Convert puddle track coordinates (arclength from the start, lateral offset) to world xy."""
    pts, tangents = point_at(bank, tid, start_s + cond.puddle_s_m, xp)
    lat = cond.puddle_lateral_m[:, None]
    normals = xp.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
    return cond._replace(puddle_xy_m=pts + lat * normals)


def start_state(
    bank: TrackBank,
    tid: Any,
    start_s: Any,
    speed_mps: Any,
    cond: Conditions,
    difficulty: Any,
    cfg: Config = CONFIG,
    xp: ModuleType = np,
) -> EnvState:
    """A car on the centerline at arclength `start_s`, aligned with the track."""
    f = xp.float32
    _, length, ds = _track_scalars(bank, tid, xp)
    start_s = wrap_s(xp.asarray(start_s, f), length, xp)
    point, tangent = point_at(bank, tid, start_s, xp)
    cond = place_puddles(cond, bank, tid, start_s, xp)
    car = initial_car_state(
        point[0], point[1], xp.arctan2(tangent[1], tangent[0]), speed_mps, cfg, xp
    )
    zero_i = xp.zeros((), xp.int32)
    o = cfg.obs
    return EnvState(
        car=car,
        cond=cond,
        track_id=xp.asarray(tid, xp.int32),
        idx=xp.floor(start_s / ds).astype(xp.int32),
        s_m=start_s,
        progress_m=xp.zeros((), f),
        steps=zero_i,
        spin_steps=zero_i,
        stall_steps=zero_i,
        prev_action=xp.zeros(2, f),
        last_proprio=xp.zeros(o.proprio_size, f),
        history=xp.zeros((o.history_len, o.history_row_size), f),
        history_len=zero_i,
        difficulty=xp.asarray(difficulty, f),
    )


def core_step(
    state: EnvState,
    action: Any,
    gust_noise: Any,
    bank: TrackBank,
    cfg: Config = CONFIG,
    xp: ModuleType = np,
) -> tuple[EnvState, Any, StepInfo]:
    """Advance one car one control step: physics, progress, reward, termination. No reset.

    `action` [2] in [−1, 1] (clipped). `gust_noise` [2] standard normal.
    Returns (new state, reward, info). Observations are computed separately by `observe`.
    """
    ep, rw = cfg.episode, cfg.reward
    action = xp.clip(action, -1.0, 1.0)
    tid = state.track_id
    _, length, _ = _track_scalars(bank, tid, xp)
    width = bank.width[tid]

    car = control_step(state.car, state.cond, action, gust_noise, cfg, xp)
    idx, s = locate(bank, tid, car.x_m, car.y_m, state.idx, cfg, xp)
    raw_ds = s - state.s_m
    delta_s = xp.clip(
        raw_ds - length * xp.round(raw_ds / length), -ep.delta_s_clip_m, ep.delta_s_clip_m
    )
    progress = state.progress_m + delta_s
    e_y, e_psi = lateral_and_heading_error(bank, tid, s, car.x_m, car.y_m, car.yaw_rad, xp)
    speed = xp.sqrt(car.vx_mps * car.vx_mps + car.vy_mps * car.vy_mps)

    dt = cfg.time.control_dt_s
    spin_steps = xp.where(xp.abs(e_psi) > ep.spin_angle_rad, state.spin_steps + 1, 0)
    stall_steps = xp.where(speed < ep.stall_speed_mps, state.stall_steps + 1, 0)
    steps = state.steps + 1

    off_track = xp.abs(e_y) > width / 2.0 + ep.offtrack_margin_m
    spun = spin_steps > int(round(ep.spin_time_s / dt))
    crashed = off_track | spun
    stalled = stall_steps > int(round(ep.stall_time_s / dt))
    wrong_way = progress < -ep.wrong_way_m
    lap = progress >= length
    terminated = crashed | stalled | wrong_way | lap
    truncated = (steps >= ep.max_steps) & ~terminated

    da = action - state.prev_action
    reward = (
        rw.progress_per_m * delta_s
        - rw.center * (e_y / (width / 2.0)) ** 2
        - rw.smooth * xp.sum(da * da)
        + rw.lap_bonus * lap
        - rw.crash_penalty * crashed
        - rw.stall_penalty * stalled
    )

    row = xp.concatenate([state.last_proprio, action])
    history = xp.concatenate([state.history[1:], row[None, :]], axis=0)
    new_state = state._replace(
        car=car,
        idx=idx,
        s_m=s,
        progress_m=progress,
        steps=steps,
        spin_steps=spin_steps,
        stall_steps=stall_steps,
        prev_action=action,
        history=history,
        history_len=xp.minimum(state.history_len + 1, cfg.obs.history_len),
    )
    mu_f, mu_r = axle_grip(car, state.cond, cfg, xp)
    info = StepInfo(
        delta_s_m=delta_s,
        lap_completed=lap,
        crashed=crashed,
        stalled=stalled,
        wrong_way=wrong_way,
        terminated=terminated,
        truncated=truncated,
        speed_mps=speed,
        mu_front=mu_f,
        mu_rear=mu_r,
        steps=steps,
        progress_m=progress,
    )
    return new_state, reward, info


# ---------------------------------------------------------------------------------------------
# Batched JAX environment
# ---------------------------------------------------------------------------------------------


class Env:
    """Batched training environment on a track bank (JAX). All methods are jitted and vmapped.

    Usage:
        env = Env(bank)
        state, obs = env.reset(keys, difficulty)                        # keys [N], difficulty [N]
        state, obs, reward, info, final_obs = env.step(state, actions, keys, difficulty)
    """

    def __init__(self, bank: TrackBank, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self.bank = TrackBank(*(jnp.asarray(a) for a in bank))
        self.n_tracks = int(bank.width.shape[0])
        self._reset = jax.jit(jax.vmap(self._reset_one, in_axes=(0, 0, None)))
        self._step = jax.jit(jax.vmap(self._step_one, in_axes=(0, 0, 0, 0, None)))
        self._eval_reset = jax.jit(jax.vmap(self._eval_reset_one, in_axes=(0, 0, 0, None)))

    # One car ---------------------------------------------------------------------------------
    def _observe(
        self, state: EnvState, key: jax.Array, bank: TrackBank
    ) -> tuple[EnvState, Observation]:
        size = self.cfg.obs.proprio_size + 2 * len(self.cfg.obs.lookahead_m)
        obs, sensed = observe(state, bank, jax.random.normal(key, (size,)), self.cfg, jnp)
        return state._replace(last_proprio=sensed), obs

    def _reset_one(
        self, key: jax.Array, difficulty: jax.Array, bank: TrackBank
    ) -> tuple[EnvState, Observation]:
        """Training reset: random track, random start, speed U(5, 20) m/s, conditions at d."""
        ep = self.cfg.episode
        k_track, k_start, k_speed, k_cond, k_obs = jax.random.split(key, 5)
        tid = jax.random.randint(k_track, (), 0, self.n_tracks)
        length = bank.length[tid]
        start_s = jax.random.uniform(k_start, (), maxval=length)
        speed = jax.random.uniform(
            k_speed, (), minval=ep.train_speed_min_mps, maxval=ep.train_speed_max_mps
        )
        cond = sample_conditions(k_cond, difficulty, length, bank.width[tid], self.cfg)
        state = start_state(bank, tid, start_s, speed, cond, difficulty, self.cfg, jnp)
        return self._observe(state, k_obs, bank)

    def _eval_reset_one(
        self, key: jax.Array, tid: jax.Array, cond: Conditions, bank: TrackBank
    ) -> tuple[EnvState, Observation]:
        """Evaluation reset: start at s = 0 at 5 m/s on a given track with given conditions."""
        state = start_state(
            bank, tid, 0.0, self.cfg.episode.eval_speed_mps, cond, 1.0, self.cfg, jnp
        )
        return self._observe(state, key, bank)

    def _step_one(
        self,
        state: EnvState,
        action: jax.Array,
        key: jax.Array,
        difficulty: jax.Array,
        bank: TrackBank,
    ) -> tuple[EnvState, Observation, jax.Array, StepInfo, Observation]:
        k_gust, k_obs, k_reset = jax.random.split(key, 3)
        state, reward, info = core_step(
            state, action, jax.random.normal(k_gust, (2,)), bank, self.cfg, jnp
        )
        state, obs = self._observe(state, k_obs, bank)
        done = info.terminated | info.truncated
        reset_state, reset_obs = self._reset_one(k_reset, difficulty, bank)
        pick = lambda fresh, old: jnp.where(done, fresh, old)  # noqa: E731
        return (
            jax.tree.map(pick, reset_state, state),
            jax.tree.map(pick, reset_obs, obs),
            reward,
            info,
            obs,  # pre-reset observation, for bootstrapping on truncation
        )

    # Batched ---------------------------------------------------------------------------------
    def reset(self, keys: jax.Array, difficulty: jax.Array) -> tuple[EnvState, Observation]:
        """Reset N cars. `keys` [N] (split them yourself), `difficulty` [N]."""
        return self._reset(keys, difficulty, self.bank)

    def eval_reset(
        self, keys: jax.Array, track_ids: jax.Array, conds: Conditions
    ) -> tuple[EnvState, Observation]:
        """Evaluation start for N cars on given tracks and conditions (batched `Conditions`)."""
        return self._eval_reset(keys, track_ids, conds, self.bank)

    def step(
        self, state: EnvState, actions: jax.Array, keys: jax.Array, difficulty: jax.Array
    ) -> tuple[EnvState, Observation, jax.Array, StepInfo, Observation]:
        """Step N cars with auto-reset. Returns (state, obs, reward, info, pre-reset obs).

        A car whose episode ended this step comes back already reset (new track, start and
        conditions drawn at `difficulty` [N]); `info` describes the step that ended it.
        """
        return self._step(state, actions, keys, difficulty, self.bank)
