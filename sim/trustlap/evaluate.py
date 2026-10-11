"""Evaluation protocol (SPEC 8) → `metrics.json` (schema SPEC 9.3).

* Drivers: scripted (knows true grip), blind, student (ensemble-mean ẑ), student_governed
  (student + safety governor at the τ chosen on the validation bank), teacher (oracle).
* Suites: 8 in-distribution + 4 out-of-distribution (`conditions.SUITES`).
* Paired: every driver sees the identical (track, condition seed) pairs of the test bank.
* 95% CIs: cluster bootstrap over tracks (2,000 resamples); paired deltas the same way.
* Paired lap time: mean over laps completed by *all* drivers (no survivor bias).
* Model selection (governor τ, calibration fit) uses the validation bank only.

    python -m trustlap.evaluate --blind runs/blind/final.msgpack \\
        --teacher runs/teacher/final.msgpack --students runs/students/students.msgpack
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import OOD_SUITES, SUITES, Conditions, nominal_conditions, suite_conditions
from trustlap.confidence import ensemble_act, governed_action, isotonic_fit, reliability
from trustlap.config import CONFIG
from trustlap.env import Env, EnvState, Observation, core_step, observe
from trustlap.networks import ActorCritic
from trustlap.ppo import RunningStats, policy_action
from trustlap.scripted import scripted_action
from trustlap.student import load_students, load_teacher
from trustlap.track import TrackBank, build_track, frames, load_bank, make_bank
from trustlap.train import load_policy

# act(state, obs, bank, calib_u, calib_p, tau) -> (action [2], uncertainty [], grip_hat [2]).
# tau = +inf disables the safety governor (only the student uses it).
ActFn = Callable[..., tuple[jax.Array, jax.Array, jax.Array]]

DRIVERS = ("scripted", "blind", "student", "student_governed", "teacher")
CRASH_HORIZON_STEPS = int(round(CONFIG.ensemble.crash_horizon_s / CONFIG.time.control_dt_s))
NAN2 = jnp.full(2, jnp.nan)


# ---------------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------------


def _policy(path: Path) -> ActFn:
    pol = load_policy(path)
    net = ActorCritic(privileged_actor="encoder" in pol["params"]["params"])
    params = jax.tree.map(jnp.asarray, pol["params"])
    o = RunningStats(**{k: jnp.asarray(v) for k, v in pol["obs_stats"].items()})
    p = RunningStats(**{k: jnp.asarray(v) for k, v in pol["priv_stats"].items()})

    def act(
        state: EnvState,
        obs: Observation,
        bank: TrackBank,
        cu: jax.Array,
        cp: jax.Array,
        tau: jax.Array,
    ) -> Any:
        return policy_action(net, params, o, p, obs), jnp.nan, NAN2

    return act


def _scripted() -> ActFn:
    def act(
        state: EnvState,
        obs: Observation,
        bank: TrackBank,
        cu: jax.Array,
        cp: jax.Array,
        tau: jax.Array,
    ) -> Any:
        return scripted_action(state, bank, CONFIG, jnp), jnp.nan, NAN2

    return act


def _student(teacher_path: Path, students_path: Path) -> ActFn:
    teacher = load_teacher(teacher_path)
    params = jax.tree.map(jnp.asarray, load_students(students_path)["params"])

    def act(
        state: EnvState,
        obs: Observation,
        bank: TrackBank,
        cu: jax.Array,
        cp: jax.Array,
        tau: jax.Array,
    ) -> Any:
        out = ensemble_act(teacher, params, obs)
        action = governed_action(out.action, jnp.interp(out.uncertainty, cu, cp), tau)
        return action, out.uncertainty, out.grip_mean

    return act


# ---------------------------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Laps:
    """Per-lap results, arrays [N] (N = tracks × seeds, track-major)."""

    completed: np.ndarray
    crashed: np.ndarray
    lap_time_s: np.ndarray  # NaN unless completed
    mean_speed: np.ndarray
    grip_mae: np.ndarray  # NaN for drivers without a grip estimate
    u: np.ndarray | None = None  # [T, N] per-step uncertainty (students, if recorded)
    end_step: np.ndarray | None = None


_CHUNK = 200
_compiled: dict[tuple[int, int], Callable] = {}


def run(
    env: Env,
    track_ids: np.ndarray,
    conds: Conditions,
    act: ActFn,
    calib: tuple[np.ndarray, np.ndarray] | None = None,
    record_u: bool = False,
    seed: int = 0,
    tau: float = float("inf"),
) -> Laps:
    """One lap per car from the start line; stops early once every car has finished."""
    n = len(track_ids)
    bank = env.bank
    cu, cp = (
        (jnp.asarray(calib[0]), jnp.asarray(calib[1])) if calib else (jnp.zeros(2), jnp.zeros(2))
    )
    tau_j = jnp.float32(tau)
    key = jax.random.key(seed)
    state, obs = env.eval_reset(jax.random.split(key, n), jnp.asarray(track_ids, jnp.int32), conds)
    obs_size = CONFIG.obs.policy_size - 1

    key_id = (id(act), n)
    if key_id not in _compiled:

        @jax.jit
        def chunk(
            carry: tuple, keys: jax.Array, cu: jax.Array, cp: jax.Array, tau_j: jax.Array
        ) -> tuple:
            def one(st: EnvState, ob: Observation, done: jax.Array, k: jax.Array) -> Any:
                k_gust, k_obs = jax.random.split(k)
                action, u, grip = act(st, ob, bank, cu, cp, tau_j)
                new, _, info = core_step(
                    st, action, jax.random.normal(k_gust, (2,)), bank, CONFIG, jnp
                )
                new_obs, sensed = observe(
                    new, bank, jax.random.normal(k_obs, (obs_size,)), CONFIG, jnp
                )
                new = new._replace(last_proprio=sensed)
                keep = lambda a, b: jnp.where(done, b, a)  # noqa: E731
                grip_err = jnp.mean(jnp.abs(grip - jnp.stack([info.mu_front, info.mu_rear])))
                return (
                    jax.tree.map(keep, new, st),
                    jax.tree.map(keep, new_obs, ob),
                    info,
                    u,
                    grip_err,
                )

            def body(c: tuple, k: jax.Array) -> tuple:
                st, ob, done, outcome, end, speed_sum, gerr_sum = c
                st2, ob2, info, u, gerr = jax.vmap(one)(st, ob, done, jax.random.split(k, n))
                ending = ~done & (info.terminated | info.truncated)
                code = jnp.select(
                    [
                        info.lap_completed,
                        info.crashed | info.stalled | info.wrong_way,
                        info.truncated,
                    ],
                    [1, 2, 3],
                    0,
                )
                outcome = jnp.where(ending, code, outcome)
                end = jnp.where(ending, info.steps, end)
                live = ~done
                speed_sum = speed_sum + jnp.where(live, info.speed_mps, 0.0)
                gerr_sum = gerr_sum + jnp.where(live, gerr, 0.0)
                done = done | ending
                return (st2, ob2, done, outcome, end, speed_sum, gerr_sum), jnp.where(
                    live, u, jnp.nan
                )

            return jax.lax.scan(body, carry, keys)

        _compiled[key_id] = chunk
    chunk = _compiled[key_id]
    zeros_i = jnp.zeros(n, jnp.int32)
    carry = (state, obs, jnp.zeros(n, bool), zeros_i, zeros_i, jnp.zeros(n), jnp.zeros(n))
    us = []
    steps = 0
    key = jax.random.fold_in(key, 1)
    while steps < CONFIG.episode.max_steps:
        key, k = jax.random.split(key)
        carry, u = chunk(carry, jax.random.split(k, _CHUNK), cu, cp, tau_j)
        if record_u:
            us.append(np.asarray(u))
        steps += _CHUNK
        if bool(jnp.all(carry[2])):
            break
    _, _, _, outcome, end, speed_sum, gerr_sum = (
        np.asarray(x) if i > 1 else x for i, x in enumerate(carry)
    )
    completed = outcome == 1
    denom = np.maximum(end, 1)
    return Laps(
        completed=completed,
        crashed=outcome == 2,
        lap_time_s=np.where(completed, end * CONFIG.time.control_dt_s, np.nan),
        mean_speed=speed_sum / denom,
        grip_mae=gerr_sum / denom,
        u=np.concatenate(us, axis=0) if record_u else None,
        end_step=end,
    )


def suite_batch(
    bank: TrackBank, suite: str, seeds: int, n_tracks: int, seed: int
) -> tuple[np.ndarray, Conditions]:
    track_ids = np.repeat(np.arange(n_tracks), seeds)
    # Stable across processes (Python's hash() of a str is randomized per run).
    keys = jax.random.split(jax.random.key(zlib.crc32(suite.encode()) + seed), len(track_ids))
    lengths = jnp.asarray(bank.length)[track_ids]
    widths = jnp.asarray(bank.width)[track_ids]
    conds = jax.vmap(lambda k, length, width: suite_conditions(k, suite, length, width))(
        keys, lengths, widths
    )
    return track_ids, conds


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------


def cluster_bootstrap(
    per_track: np.ndarray, rng: np.random.Generator, n: int = 2000
) -> list[float]:
    """95% CI of a mean over a [tracks, seeds] array (NaNs ignored), resampling tracks."""
    idx = rng.integers(0, per_track.shape[0], (n, per_track.shape[0]))
    with np.errstate(invalid="ignore"):
        means = np.nanmean(per_track[idx].reshape(n, -1), axis=1)
    means = means[np.isfinite(means)]
    if len(means) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def stat(values: np.ndarray, tracks: int, rng: np.random.Generator) -> dict[str, Any]:
    grid = values.reshape(tracks, -1).astype(float)
    value = float(np.nanmean(grid)) if np.isfinite(grid).any() else float("nan")
    return {"value": value, "ci": cluster_bootstrap(grid, rng)}


def ratio_bootstrap(
    num: Callable[[np.ndarray], float], tracks: int, rng: np.random.Generator, n: int = 2000
) -> tuple[float, list[float]]:
    """Point value and 95% CI of a statistic computed from a resampled track index set."""
    base = num(np.arange(tracks))
    vals = np.array([num(rng.integers(0, tracks, tracks)) for _ in range(n)])
    vals = vals[np.isfinite(vals)]
    ci = (
        [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        if len(vals)
        else [float("nan")] * 2
    )
    return float(base), ci


def crash_labels(laps: Laps, horizon: int = CRASH_HORIZON_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """(u_t, y_t) pairs over all live steps: y = 1 if the car crashes within the next 2 s."""
    assert laps.u is not None and laps.end_step is not None
    t = np.arange(laps.u.shape[0])[:, None]
    crash_at = np.where(laps.crashed, laps.end_step, np.iinfo(np.int32).max)[None, :]
    y = ((crash_at - (t + 1)) < horizon) & ((t + 1) <= crash_at)
    live = np.isfinite(laps.u)
    return laps.u[live], y[live].astype(float)


# ---------------------------------------------------------------------------------------------
# Puddle step response (SPEC 8.3)
# ---------------------------------------------------------------------------------------------


def stadium_bank(
    straight_m: float = 700.0, radius_m: float = 100.0, width_m: float = 12.0
) -> TrackBank:
    """A long straight-sided loop for the step-response test (closed, so the env works)."""
    ds = CONFIG.track.ds_target_m
    pts = []
    for x in np.arange(0, straight_m, ds):
        pts.append((x, -radius_m))
    for a in np.arange(-np.pi / 2, np.pi / 2, ds / radius_m):
        pts.append((straight_m + radius_m * np.cos(a), radius_m * np.sin(a)))
    for x in np.arange(straight_m, 0, -ds):
        pts.append((x, radius_m))
    for a in np.arange(np.pi / 2, 3 * np.pi / 2, ds / radius_m):
        pts.append((radius_m * np.cos(a), radius_m * np.sin(a)))
    centerline = np.array(pts, float)
    seg = np.linalg.norm(np.roll(centerline, -1, 0) - centerline, axis=1)
    length = float(seg.sum())
    tangent, normal, curvature = frames(
        centerline, length / len(centerline), CONFIG.track.curvature_smooth_window
    )
    track = dataclasses.replace(
        build_track(np.zeros((4, 2)) + [[0, 0], [1, 0], [1, 1], [0, 1]], width_m, 0, 0),
        centerline=centerline,
        tangent=tangent,
        normal=normal,
        curvature=curvature,
        length_m=length,
        control_points=np.zeros((0, 2)),
    )
    return make_bank([track])


def puddle_step(act: ActFn, trials: int = 200, seed: int = 0) -> dict[str, Any]:
    """Single puddle (m = 0.4) 200 m down a straight; car at 20 ± 2 m/s. Latency = time from the
    rear grip first dropping by more than 0.1 (the tolerance) until |μ̂_rear − μ_rear| < 0.1."""
    bank = stadium_bank()
    env = Env(bank)
    rng = np.random.default_rng(seed)
    c = nominal_conditions(CONFIG, jnp)
    p = CONFIG.conditions.puddle_slots
    lateral = rng.uniform(-1.5, 1.5, trials).astype(np.float32)
    conds = jax.vmap(
        lambda lat: c._replace(
            puddle_s_m=jnp.zeros(p).at[0].set(200.0),
            puddle_lateral_m=jnp.zeros(p).at[0].set(lat),
            puddle_mult=jnp.ones(p).at[0].set(0.4),
            puddle_radius_m=jnp.full(p, 8.0),
            puddle_active=jnp.zeros(p).at[0].set(1.0),
        )
    )(jnp.asarray(lateral))
    state, obs = env.eval_reset(
        jax.random.split(jax.random.key(seed), trials), jnp.zeros(trials, jnp.int32), conds
    )
    # 20 m/s ± 2 so the 200 trials differ (entry speed changes how fast the rear crosses the edge).
    speeds = jnp.asarray(rng.uniform(18.0, 22.0, trials).astype(np.float32))
    state = state._replace(car=state.car._replace(vx_mps=speeds))
    zero2 = jnp.zeros(2)

    @jax.jit
    def roll(state: EnvState, obs: Observation) -> tuple[jax.Array, jax.Array]:
        def body(c: tuple, _: Any) -> tuple:
            st, ob = c

            def one(s: EnvState, o: Observation) -> Any:
                a, _, grip = act(s, o, bank_j, zero2, zero2, jnp.float32(jnp.inf))
                new, _, info = core_step(s, a, jnp.zeros(2), bank_j, CONFIG, jnp)
                no, sensed = observe(
                    new, bank_j, jnp.zeros(CONFIG.obs.policy_size - 1), CONFIG, jnp
                )
                return new._replace(last_proprio=sensed), no, info.mu_rear, grip[1]

            st, ob, mu, mu_hat = jax.vmap(one)(st, ob)
            return (st, ob), (mu, mu_hat)

        _, (mu, mu_hat) = jax.lax.scan(body, (state, obs), None, length=260)
        return mu, mu_hat

    bank_j = env.bank
    mu, mu_hat = (np.asarray(x) for x in roll(state, obs))  # [T, trials]
    dt_ms = CONFIG.time.control_dt_s * 1000
    base = mu[0]
    latencies = []
    for i in range(trials):
        drop = np.nonzero(mu[:, i] < base[i] - 0.1)[0]
        if len(drop) == 0:
            continue
        t0 = drop[0]
        ok = np.nonzero(np.abs(mu_hat[t0:, i] - mu[t0:, i]) < 0.1)[0]
        latencies.append(ok[0] * dt_ms if len(ok) else np.nan)
    lat = np.array(latencies)
    finite = lat[np.isfinite(lat)]
    # Representative series: the median-latency trial, from 1 s before the drop to 3 s after.
    rep = int(np.nanargmin(np.abs(lat - np.nanmedian(lat)))) if len(finite) else 0
    drop0 = int(np.nonzero(mu[:, rep] < base[rep] - 0.1)[0][0]) if len(finite) else 0
    lo, hi = max(0, drop0 - 20), min(mu.shape[0], drop0 + 60)
    return {
        "latency_ms": {
            "median": float(np.median(finite)) if len(finite) else None,
            "iqr": [float(np.percentile(finite, 25)), float(np.percentile(finite, 75))]
            if len(finite)
            else None,
            "trials": int(len(lat)),
            "never_settled": int(np.sum(~np.isfinite(lat))),
        },
        "series": {
            "t_ms": [float((t - drop0) * dt_ms) for t in range(lo, hi)],
            "mu_true": mu[lo:hi, rep].round(4).tolist(),
            "mu_hat": mu_hat[lo:hi, rep].round(4).tolist(),
        },
    }


# ---------------------------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    test = load_bank(args.test_bank)
    val = load_bank(args.val_bank)
    test_tracks = min(args.tracks, len(test.seeds))
    val_tracks = len(val.seeds)
    env_test, env_val = Env(test), Env(val)
    drivers: dict[str, ActFn] = {
        "scripted": _scripted(),
        "blind": _policy(args.blind),
        "teacher": _policy(args.teacher),
    }
    student = _student(args.teacher, args.students)
    drivers["student"] = student
    log = lambda *a: print(f"[eval {time.time() - t_start:6.0f}s]", *a, flush=True)  # noqa: E731

    # 1. Calibration fit + governor τ choice on the VALIDATION bank (model selection hygiene).
    calib_suites = ("combined", "puddles", "wet", "ice", "extreme_lag")
    u_all, y_all = [], []
    for suite in calib_suites:
        tids, conds = suite_batch(val, suite, 1, val_tracks, args.seed + 1)
        laps = run(env_val, tids, conds, student, record_u=True, seed=args.seed + 1)
        u, y = crash_labels(laps)
        u_all.append(u)
        y_all.append(y)
    u_val, y_val = np.concatenate(u_all), np.concatenate(y_all)
    table_u, table_p = isotonic_fit(u_val, y_val)
    calib = (table_u, table_p)
    log(f"calibration fit on {len(u_val):,} validation steps ({y_val.mean():.3%} positive)")

    taus = list(CONFIG.ensemble.governor_taus)
    tids, conds = suite_batch(val, "combined", 2, val_tracks, args.seed + 2)
    base = run(env_val, tids, conds, student, calib, seed=args.seed + 2)
    best_tau, best_crash = None, base.crashed.mean()
    for tau in taus:
        gov = run(env_val, tids, conds, student, calib, seed=args.seed + 2, tau=tau)
        both = base.completed & gov.completed
        slower = (
            np.nanmean(gov.lap_time_s[both]) / np.nanmean(base.lap_time_s[both]) - 1
            if both.any()
            else np.inf
        )
        log(
            f"validation τ={tau}: crash {gov.crashed.mean():.3f} "
            f"(ungoverned {base.crashed.mean():.3f}), lap time {slower:+.1%}"
        )
        if slower <= args.max_slowdown and gov.crashed.mean() < best_crash:
            best_tau, best_crash = tau, gov.crashed.mean()
    tau_default = best_tau if best_tau is not None else taus[len(taus) // 2]
    log(
        f"chosen τ = {tau_default} (lowest validation crash rate within "
        f"{args.max_slowdown:.0%} lap-time cost)"
    )
    drivers["student_governed"] = student

    # 2. Every suite × driver on the TEST bank, paired.
    suites_out: dict[str, Any] = {}
    paired: list[dict[str, Any]] = []
    raw: dict[str, dict[str, Laps]] = {}
    for suite in SUITES:
        tids, conds = suite_batch(test, suite, args.seeds, test_tracks, args.seed + 10)
        raw[suite] = {}
        for name in DRIVERS:
            raw[suite][name] = run(
                env_test,
                tids,
                conds,
                drivers[name],
                calib,
                record_u=name == "student",
                seed=args.seed + 10,
                tau=tau_default if name == "student_governed" else float("inf"),
            )
        all_done = np.all([raw[suite][d].completed for d in DRIVERS], axis=0)
        suites_out[suite] = {}
        for name in DRIVERS:
            laps = raw[suite][name]
            paired_time = np.where(all_done, laps.lap_time_s, np.nan)
            entry = {
                "completion": stat(laps.completed, test_tracks, rng),
                "crash": stat(laps.crashed, test_tracks, rng),
                "paired_lap_time_s": stat(paired_time, test_tracks, rng)
                if all_done.any()
                else None,
                "mean_speed_mps": stat(laps.mean_speed, test_tracks, rng),
                "n_laps": int(len(laps.completed)),
                "n_paired_laps": int(all_done.sum()),
            }
            if name.startswith("student"):
                entry["grip_mae"] = stat(laps.grip_mae, test_tracks, rng)
            suites_out[suite][name] = entry
        for a, b in (
            ("student", "blind"),
            ("student", "scripted"),
            ("teacher", "student"),
            ("student_governed", "student"),
        ):
            for metric, attr in (("completion", "completed"), ("crash", "crashed")):
                diff = getattr(raw[suite][a], attr).astype(float) - getattr(
                    raw[suite][b], attr
                ).astype(float)
                paired.append(
                    {
                        "suite": suite,
                        "a": a,
                        "b": b,
                        "metric": metric,
                        **stat(diff, test_tracks, rng),
                    }
                )
        s = suites_out[suite]
        log(
            f"{suite:14s} completion: "
            + "  ".join(f"{d} {s[d]['completion']['value']:.3f}" for d in DRIVERS)
        )

    # 3. Headline numbers.
    comb = raw["combined"]
    seeds = args.seeds

    def per_track(laps: Laps, attr: str) -> np.ndarray:
        return getattr(laps, attr).reshape(test_tracks, seeds).astype(float)

    def crash_reduction(idx: np.ndarray) -> float:
        b = per_track(comb["blind"], "crashed")[idx].mean()
        s_ = per_track(comb["student"], "crashed")[idx].mean()
        return float((b - s_) / b) if b > 0 else float("nan")

    def teacher_gap(idx: np.ndarray) -> float:
        t = per_track(comb["teacher"], "completed")[idx].mean()
        s_ = per_track(comb["student"], "completed")[idx].mean()
        b = per_track(comb["blind"], "completed")[idx].mean()
        return float((t - s_) / (t - b)) if abs(t - b) > 1e-9 else float("nan")

    red, red_ci = ratio_bootstrap(crash_reduction, test_tracks, rng)
    gap, gap_ci = ratio_bootstrap(teacher_gap, test_tracks, rng)
    grip_all = np.concatenate([raw[s]["student"].grip_mae for s in SUITES if s not in OOD_SUITES])

    # 4. Calibration quality on the TEST bank (student rollouts of the calibration suites).
    u_t, y_t = [], []
    for suite in calib_suites:
        u, y = crash_labels(raw[suite]["student"])
        u_t.append(u)
        y_t.append(y)
    u_test, y_test = np.concatenate(u_t), np.concatenate(y_t)
    p_test = np.interp(u_test, table_u, table_p)
    rel = reliability(p_test, y_test)
    order = np.argsort(p_test, kind="stable")
    for b_, chunk in zip(rel["bins"], np.array_split(order, len(rel["bins"])), strict=True):
        b_["u_mean"] = float(np.mean(u_test[chunk]))

    # 5. Governor sweep on the test bank (combined suite).
    tids, conds = suite_batch(test, "combined", args.seeds, test_tracks, args.seed + 10)
    sweep_raw = {"off": comb["student"]}
    for tau in taus:
        sweep_raw[str(tau)] = run(
            env_test, tids, conds, student, calib, seed=args.seed + 10, tau=tau
        )
    sweep_done = np.all([v.completed for v in sweep_raw.values()], axis=0)
    sweep = []
    for key_, laps in sweep_raw.items():
        sweep.append(
            {
                "tau": None if key_ == "off" else float(key_),
                "crash": stat(laps.crashed, test_tracks, rng),
                "paired_lap_time_s": stat(
                    np.where(sweep_done, laps.lap_time_s, np.nan), test_tracks, rng
                ),
            }
        )
    log("governor sweep done")

    # 6. Puddle step response and generalization (train bank vs test bank, combined suite).
    step = puddle_step(student, seed=args.seed)
    log(f"puddle latency median {step['latency_ms']['median']} ms")
    train = load_bank(args.train_bank)
    env_train = Env(train)
    tids, conds = suite_batch(train, "combined", args.seeds, test_tracks, args.seed + 20)
    gen = {"train_bank": {}, "test_bank": {}}
    for name in ("blind", "student", "teacher"):
        laps = run(env_train, tids, conds, drivers[name], calib, seed=args.seed + 20)
        gen["train_bank"][name] = {"completion": stat(laps.completed, test_tracks, rng)}
        gen["test_bank"][name] = {"completion": suites_out["combined"][name]["completion"]}

    return {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "test_tracks": test_tracks,
            "condition_seeds": seeds,
            "laps_per_driver_per_suite": test_tracks * seeds,
            "bootstrap_resamples": 2000,
            "governor_tau": tau_default,
            "calibration_fit": (
                f"validation bank, suites {list(calib_suites)}, {len(u_val):,} steps"
            ),
            "eval_wall_minutes": round((time.time() - t_start) / 60, 1),
        },
        "headline": {
            "student_crash_reduction_vs_blind_combined": {"value": red, "ci": red_ci},
            "teacher_gap_combined": {"value": gap, "ci": gap_ci},
            "grip_mae": float(np.nanmean(grip_all)),
            "puddle_latency_ms": step["latency_ms"],
        },
        "suites": suites_out,
        "paired_deltas": paired,
        "calibration": {
            **rel,
            "table": {"u": table_u.tolist(), "p": table_p.tolist()},
            "tau_default": tau_default,
        },
        "governor_sweep": sweep,
        "puddle_step": step["series"],
        "generalization": gen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--test-bank", type=Path, default=Path("runs/banks/test.npz"))
    parser.add_argument("--val-bank", type=Path, default=Path("runs/banks/validation.npz"))
    parser.add_argument("--train-bank", type=Path, default=Path("runs/banks/train.npz"))
    parser.add_argument("--tracks", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--max-slowdown", type=float, default=0.05, help="governor τ choice: max lap-time cost"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(args)
    (args.out / "metrics_partial.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(f"wrote {args.out / 'metrics_partial.json'}")


if __name__ == "__main__":
    main()
