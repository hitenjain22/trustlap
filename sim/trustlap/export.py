"""Export artifacts for the website and parity fixtures for its tests (SPEC 9).

    python -m trustlap.export                 # web/public/artifacts/v1/
    python -m trustlap.export --golden        # also web/tests/parity/golden/

Artifacts are what the site loads (config, showcase tracks, benchmark, later weights and
metrics). Golden files are reference outputs computed here, in JAX on the CPU, that the
TypeScript port must reproduce (SPEC 9.4).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import Conditions, sample_conditions, suite_conditions
from trustlap.config import CONFIG, config_hash, physics_hash, to_dict
from trustlap.dynamics import CarState, control_step
from trustlap.env import EnvState, core_step, observe, place_puddles, point_at, start_state
from trustlap.rng import PCG32
from trustlap.scripted import scripted_action
from trustlap.track import TrackBank, generate_track, make_bank

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_VERSION = "v1"
ARTIFACTS = REPO / "web" / "public" / "artifacts" / ARTIFACT_VERSION
GOLDEN = REPO / "web" / "tests" / "parity" / "golden"
RESULTS = REPO / "sim" / "results"

GOLDEN_TRACK_SEEDS = (0, 1, 42, 200017, 4_000_000_000)
HISTORY_CASES = 32  # env fixtures carry a random history only for the first 32 cases (size)

# Showcase names, assigned from the twistiest track to the most flowing one.
SHOWCASE_NAMES = (
    "Hairpin Valley",
    "Corkscrew Gap",
    "Switchback Ridge",
    "Knot Garden",
    "Elbow Creek",
    "Viper Bend",
    "Paperclip",
    "Fishhook Point",
    "Serpent Hill",
    "Zigzag Basin",
    "Comma Lake",
    "Kidney Bean",
    "Teardrop",
    "Boomerang",
    "Lantern Loop",
    "Harbor Run",
    "Long Meadow",
    "Sodium Mile",
    "Wet Asphalt",
    "Night Shift",
    "Rain Gauge",
    "Drift Plains",
    "Open Water",
    "Sweeper Bay",
)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _f32(x: Any) -> Any:
    """Floats as float32 values printed with 9 significant digits (exact float32 round trip)."""
    arr = np.asarray(x)
    if arr.dtype.kind == "b":
        return arr.tolist()
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64).tolist()
    arr = arr.astype(np.float32)
    return np.vectorize(lambda v: float(f"{v:.9g}"), otypes=[float])(arr).tolist()


def _soa(tree: Any) -> dict[str, Any]:
    """NamedTuple of batched arrays → {field: nested list} (struct of arrays)."""
    out = {}
    for name, value in tree._asdict().items():
        out[name] = _soa(value) if hasattr(value, "_asdict") else _f32(value)
    return out


def _compact_env(state: dict[str, Any], drop_cond: bool = False) -> dict[str, Any]:
    """Keep only the first HISTORY_CASES histories (the rest are all zeros by construction)."""
    out = dict(state)
    out["history"] = out["history"][:HISTORY_CASES]
    if drop_cond:
        out.pop("cond")  # core_step never changes the conditions
    return out


def _write_json(path: Path, data: Any, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, separators=None if indent else (",", ":")))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------------------------


def showcase_tracks(count: int = CONFIG.banks.showcase_count) -> list[dict[str, Any]]:
    """Pick `count` test-bank tracks spread from twistiest to most flowing, and name them."""
    start, stop = CONFIG.banks.test
    rows = []
    for seed in range(start, stop):
        track = generate_track(seed)
        tight = float(
            np.mean(np.abs(track.curvature) > 1.0 / 30.0)
        )  # share of the lap in tight corners
        rows.append((tight, seed, track))
    rows.sort(key=lambda r: (-r[0], r[1]))
    picks = [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]
    out = []
    for name, (_, seed, track) in zip(SHOWCASE_NAMES, picks, strict=True):
        out.append(
            {
                "seed": seed,
                "name": name,
                "slug": name.lower().replace(" ", "-"),
                "length_m": round(track.length_m, 1),
                "width_m": round(track.width_m, 2),
                "min_radius_m": round(1.0 / float(np.max(np.abs(track.curvature))), 1),
                "aspect": round(
                    float(np.ptp(track.centerline[:, 0]) / np.ptp(track.centerline[:, 1])), 3
                ),
            }
        )
    return out


def export_artifacts(out_dir: Path = ARTIFACTS) -> None:
    """Write config, showcase tracks, benchmark and the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", to_dict(), indent=1)
    _write_json(out_dir / "tracks" / "showcase.json", {"tracks": showcase_tracks()}, indent=1)
    bench = RESULTS / "bench_t4.json"
    if bench.exists():
        shutil.copyfile(bench, out_dir / "bench.json")
    files = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json"):
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(out_dir).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "config_hash": config_hash(),
        "files": files,
    }
    _write_json(out_dir / "manifest.json", manifest, indent=1)
    print(f"artifacts → {out_dir} ({len(files)} files)")


# ---------------------------------------------------------------------------------------------
# Golden parity fixtures
# ---------------------------------------------------------------------------------------------


def golden_rng() -> dict[str, Any]:
    seeds = [0, 1, 42, 123456789, 4294967295]
    cases = []
    for seed in seeds:
        rng = PCG32(seed)
        cases.append({"seed": seed, "u32": [rng.next_u32() for _ in range(1000)]})
    return {"stream": 54, "cases": cases}


def golden_tracks() -> dict[str, Any]:
    cases = []
    for seed in GOLDEN_TRACK_SEEDS:
        t = generate_track(seed)
        cases.append(
            {
                "seed": seed,
                "attempt": t.attempt,
                "smooth_iters": t.smooth_iters,
                "n_points": t.n_points,
                "length_m": t.length_m,
                "width_m": t.width_m,
                "every10_centerline": t.centerline[::10].tolist(),
                "every10_tangent": t.tangent[::10].tolist(),
                "every10_curvature": t.curvature[::10].tolist(),
            }
        )
    return {"cases": cases}


def _random_conditions(key: jax.Array, n: int, center_xy: jax.Array) -> Conditions:
    """Full-range conditions, 10% with OOD extremes, puddles scattered around each car."""
    k_c, k_ood, k_p = jax.random.split(key, 3)
    conds = jax.vmap(lambda k: sample_conditions(k, 1.0, 1200.0, 12.0))(jax.random.split(k_c, n))
    ood = jax.random.uniform(k_ood, (n,)) < 0.1
    c = CONFIG.conditions
    conds = conds._replace(
        mu_dry=jnp.where(ood, c.mu_dry.ood, conds.mu_dry),
        delay_steps=jnp.where(ood, c.delay_ood_steps, conds.delay_steps),
        steer_tau_s=jnp.where(ood, c.steer_tau_s.ood, conds.steer_tau_s),
        mass_mult=jnp.where(ood, c.mass_mult.ood, conds.mass_mult),
    )
    conds = conds._replace(mu_base=conds.mu_dry * (1.0 - c.rain_grip_drop * conds.rain))
    offsets = jax.random.uniform(k_p, (n, c.puddle_slots, 2), minval=-20.0, maxval=20.0)
    return conds._replace(puddle_xy_m=center_xy[:, None, :] + offsets)


def _random_car(key: jax.Array, n: int) -> CarState:
    ks = jax.random.split(key, 11)
    slow = jax.random.uniform(ks[9], (n,)) < 0.2  # exercise the low-speed blend
    vx = jnp.where(
        slow,
        jax.random.uniform(ks[3], (n,), maxval=3.0),
        jax.random.uniform(ks[3], (n,), maxval=60.0),
    )
    return CarState(
        x_m=jax.random.uniform(ks[0], (n,), minval=-50, maxval=50),
        y_m=jax.random.uniform(ks[1], (n,), minval=-50, maxval=50),
        yaw_rad=jax.random.uniform(ks[2], (n,), minval=-math.pi, maxval=math.pi),
        vx_mps=vx,
        vy_mps=jax.random.uniform(ks[4], (n,), minval=-5, maxval=5),
        yaw_rate_radps=jax.random.uniform(ks[5], (n,), minval=-2, maxval=2),
        steer_rad=jax.random.uniform(ks[6], (n,), minval=-0.42, maxval=0.42),
        action_buffer=jax.random.uniform(ks[7], (n, 4, 2), minval=-1, maxval=1),
        gust_mps=2.0 * jax.random.normal(ks[8], (n, 2)),
        odometer_m=jax.random.uniform(ks[10], (n,), maxval=3000.0),
    )


def golden_dynamics(n: int = 2000, seed: int = 1) -> dict[str, Any]:
    key = jax.random.key(seed)
    k_car, k_cond, k_act, k_noise = jax.random.split(key, 4)
    cars = _random_car(k_car, n)
    conds = _random_conditions(k_cond, n, jnp.stack([cars.x_m, cars.y_m], axis=1))
    actions = jax.random.uniform(k_act, (n, 2), minval=-1.2, maxval=1.2)
    noise = jax.random.normal(k_noise, (n, 2))
    out = jax.jit(jax.vmap(lambda s, c, a, z: control_step(s, c, a, z, CONFIG, jnp)))(
        cars, conds, actions, noise
    )
    return {
        "n": n,
        "state": _soa(cars),
        "conditions": _soa(conds),
        "action": _f32(actions),
        "gust_noise": _f32(noise),
        "next_state": _soa(out),
    }


def _random_env_states(key: jax.Array, bank: TrackBank, n: int) -> EnvState:
    """Cars near the centerline of the golden tracks with random motion, conditions and history."""
    b = TrackBank(*(jnp.asarray(a) for a in bank))
    n_tracks = int(bank.width.shape[0])
    ks = jax.random.split(key, 14)
    tid = jax.random.randint(ks[0], (n,), 0, n_tracks)
    s = jax.random.uniform(ks[1], (n,)) * b.length[tid]
    conds = jax.vmap(lambda k, t: sample_conditions(k, 1.0, b.length[t], b.width[t]))(
        jax.random.split(ks[2], n), tid
    )
    speed = jax.random.uniform(ks[3], (n,), maxval=40.0)
    states = jax.vmap(lambda t, s0, v, c: start_state(b, t, s0, v, c, 1.0, CONFIG, jnp))(
        tid, s, speed, conds
    )
    # Puddles relative to the start line, as the browser places them.
    states = states._replace(
        cond=jax.vmap(lambda c, t: place_puddles(c, b, t, 0.0, jnp))(states.cond, tid),
    )
    pts, tangents = jax.vmap(lambda t, s0: point_at(b, t, s0, jnp))(tid, s)
    normals = jnp.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    lateral = jax.random.uniform(ks[4], (n,), minval=-1.0, maxval=1.0) * (b.width[tid] / 2 + 2.0)
    xy = pts + lateral[:, None] * normals
    yaw_err = jax.random.uniform(ks[5], (n,), minval=-0.6, maxval=0.6)
    car = states.car._replace(
        x_m=xy[:, 0],
        y_m=xy[:, 1],
        yaw_rad=states.car.yaw_rad + yaw_err,
        vy_mps=jax.random.uniform(ks[6], (n,), minval=-3, maxval=3),
        yaw_rate_radps=jax.random.uniform(ks[7], (n,), minval=-1, maxval=1),
        steer_rad=jax.random.uniform(ks[8], (n,), minval=-0.3, maxval=0.3),
        odometer_m=jax.random.uniform(ks[9], (n,), maxval=2000.0),
    )
    return states._replace(
        car=car,
        prev_action=jax.random.uniform(ks[10], (n, 2), minval=-1, maxval=1),
        last_proprio=jax.random.uniform(ks[11], (n, 8), minval=-1, maxval=1),
        history=jax.random.uniform(ks[12], (n, CONFIG.obs.history_len, 10), minval=-1, maxval=1)
        * (jnp.arange(n) < HISTORY_CASES)[:, None, None],
        history_len=jnp.full((n,), CONFIG.obs.history_len, jnp.int32),
        progress_m=jax.random.uniform(ks[13], (n,), minval=-10.0, maxval=500.0),
    )


def golden_env(
    bank: TrackBank, n_obs: int = 1000, n_step: int = 1000, n_scripted: int = 500
) -> dict[str, Any]:
    """Observations, core env steps and scripted actions on random states of the golden tracks."""
    b = TrackBank(*(jnp.asarray(a) for a in bank))
    zero_noise = jnp.zeros(CONFIG.obs.policy_size - 1)
    out = {}

    states = _random_env_states(jax.random.key(2), bank, n_obs)
    obs, _ = jax.jit(jax.vmap(lambda st: observe(st, b, zero_noise, CONFIG, jnp)))(states)
    obs_soa = _soa(obs)
    obs_soa["history"] = obs_soa["history"][:HISTORY_CASES]
    out["obs"] = {
        "n": n_obs,
        "history_cases": HISTORY_CASES,
        "state": _compact_env(_soa(states)),
        "obs": obs_soa,
    }

    states = _random_env_states(jax.random.key(3), bank, n_step)
    actions = jax.random.uniform(jax.random.key(4), (n_step, 2), minval=-1, maxval=1)
    gust = jax.random.normal(jax.random.key(5), (n_step, 2))
    new, reward, info = jax.jit(jax.vmap(lambda st, a, z: core_step(st, a, z, b, CONFIG, jnp)))(
        states, actions, gust
    )
    out["env_step"] = {
        "n": n_step,
        "history_cases": HISTORY_CASES,
        "state": _compact_env(_soa(states)),
        "action": _f32(actions),
        "gust_noise": _f32(gust),
        "next_state": _compact_env(_soa(new), drop_cond=True),
        "reward": _f32(reward),
        "info": _soa(info),
    }

    states = _random_env_states(jax.random.key(6), bank, n_scripted)
    actions = jax.jit(jax.vmap(lambda st: scripted_action(st, b, CONFIG, jnp)))(states)
    out["scripted"] = {
        "n": n_scripted,
        "history_cases": HISTORY_CASES,
        "state": _compact_env(_soa(states)),
        "action": _f32(actions),
    }
    return out


def golden_scripted_closed_loop(bank: TrackBank, steps: int = 400) -> dict[str, Any]:
    """The scripted driver from the start line, no noise: progress and crash flag after `steps`."""
    b = TrackBank(*(jnp.asarray(a) for a in bank))
    n_tracks = int(bank.width.shape[0])
    cases = []
    for suite in ("dry", "wet"):
        for tid in range(n_tracks):
            cond = suite_conditions(jax.random.key(10 + tid), suite, b.length[tid], b.width[tid])
            state = start_state(
                b, jnp.int32(tid), 0.0, CONFIG.episode.eval_speed_mps, cond, 1.0, CONFIG, jnp
            )

            def body(
                carry: tuple[EnvState, jax.Array], _: Any
            ) -> tuple[tuple[EnvState, jax.Array], jax.Array]:
                st, crashed = carry
                a = scripted_action(st, b, CONFIG, jnp)
                new, _, info = core_step(st, a, jnp.zeros(2), b, CONFIG, jnp)
                new = jax.tree.map(lambda x, y: jnp.where(crashed, y, x), new, st)
                return (new, crashed | info.terminated & ~info.lap_completed), new.progress_m

            (final, crashed), progress = jax.lax.scan(
                body, (state, jnp.array(False)), None, length=steps
            )
            cases.append(
                {
                    "suite": suite,
                    "track_index": tid,
                    "conditions": _soa(cond),
                    "progress_m": _f32(progress[jnp.array([99, 199, steps - 1])]),
                    "crashed": bool(crashed),
                    "final_xy": _f32(jnp.stack([final.car.x_m, final.car.y_m])),
                }
            )
    return {"steps": steps, "checkpoints": [100, 200, steps], "cases": cases}


def export_golden(out_dir: Path = GOLDEN) -> None:
    """Write every parity fixture."""
    jax.config.update("jax_default_matmul_precision", "highest")
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = make_bank([generate_track(s) for s in GOLDEN_TRACK_SEEDS])
    meta = {
        "git_sha": _git_sha(),
        "config_hash": config_hash(),
        "physics_hash": physics_hash(),
        "jax_version": jax.__version__,
    }
    _write_json(out_dir / "rng.json", {**meta, **golden_rng()})
    _write_json(
        out_dir / "tracks.json", {**meta, "seeds": list(GOLDEN_TRACK_SEEDS), **golden_tracks()}
    )
    _write_json(out_dir / "dynamics.json", {**meta, **golden_dynamics()})
    env = golden_env(bank)
    for name, data in env.items():
        _write_json(
            out_dir / f"{name}.json", {**meta, "track_seeds": list(GOLDEN_TRACK_SEEDS), **data}
        )
    closed = golden_scripted_closed_loop(bank)
    _write_json(
        out_dir / "scripted_closed_loop.json",
        {**meta, "track_seeds": list(GOLDEN_TRACK_SEEDS), **closed},
    )
    for p in sorted(out_dir.glob("*.json")):
        print(f"golden → {p.relative_to(REPO)} ({p.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--golden", action="store_true", help="also write parity fixtures")
    parser.add_argument("--only-golden", action="store_true", help="write only parity fixtures")
    args = parser.parse_args()
    if not args.only_golden:
        export_artifacts()
    if args.golden or args.only_golden:
        export_golden()


if __name__ == "__main__":
    main()
