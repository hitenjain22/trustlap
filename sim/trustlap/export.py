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
import types
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
# Policy weights (SPEC 9.2)
# ---------------------------------------------------------------------------------------------


def _dense_layers(
    params: dict[str, Any], prefix: str, out_activation: str
) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """[(name, kernel [in, out], bias [out], activation)] for `{prefix}_0…`, then `{prefix}_out`."""
    params = params[prefix]  # Flax nests each MLP's layers under the module name
    layers = []
    i = 0
    while f"{prefix}_{i}" in params:
        layer = params[f"{prefix}_{i}"]
        layers.append(
            (f"{prefix}_{i}", np.asarray(layer["kernel"]), np.asarray(layer["bias"]), "elu")
        )
        i += 1
    out = params[f"{prefix}_out"]
    layers.append(
        (f"{prefix}_out", np.asarray(out["kernel"]), np.asarray(out["bias"]), out_activation)
    )
    return layers


def _stats_json(stats: dict[str, Any]) -> dict[str, list[float]]:
    var = np.asarray(stats["var"], np.float64)
    return {
        "mean": np.asarray(stats["mean"], np.float32).astype(float).tolist(),
        "std": np.sqrt(var + 1e-8).astype(np.float32).astype(float).tolist(),
    }


def write_network(
    out_dir: Path,
    name: str,
    nets: dict[str, list[tuple[str, np.ndarray, np.ndarray, str]]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Write `name.bin` (little-endian float32: per layer kernel [in, out] row-major, then bias)
    and `name.json` (layer table with byte offsets, plus `extra`). Returns the json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[bytes] = []
    offset = 0
    table: dict[str, list[dict[str, Any]]] = {}
    for net_name, layers in nets.items():
        table[net_name] = []
        for layer_name, kernel, bias, activation in layers:
            k = np.ascontiguousarray(kernel, dtype="<f4")
            b = np.ascontiguousarray(bias, dtype="<f4")
            table[net_name].append(
                {
                    "name": layer_name,
                    "in": int(k.shape[0]),
                    "out": int(k.shape[1]),
                    "activation": activation,
                    "byte_offset": offset,
                }
            )
            chunks += [k.tobytes(), b.tobytes()]
            offset += k.nbytes + b.nbytes
    blob = b"".join(chunks)
    (out_dir / f"{name}.bin").write_bytes(blob)
    meta = {
        "schema_version": 1,
        "name": name,
        "bin": f"{name}.bin",
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "networks": table,
        "obs_clip": 10.0,
        **extra,
    }
    _write_json(out_dir / f"{name}.json", meta, indent=1)
    return meta


def export_actor_policy(
    policy_path: Path, name: str, out_dir: Path = ARTIFACTS / "policies"
) -> dict[str, Any]:
    """Export a blind or teacher policy saved by `train.save_policy`: actor (+ encoder) and the
    normalization statistics. The critic stays behind (training only)."""
    from trustlap.train import load_policy

    pol = load_policy(policy_path)
    params = pol["params"]["params"]
    nets = {"actor": _dense_layers(params, "actor", "linear")}
    extra: dict[str, Any] = {
        "kind": "teacher" if "encoder" in params else "blind",
        "obs_norm": _stats_json(pol["obs_stats"]),
        "log_std": np.asarray(params["log_std"], np.float32).astype(float).tolist(),
        "training": pol["info"],
    }
    if "encoder" in params:
        nets["encoder"] = _dense_layers(params, "encoder", "tanh")
        extra["privileged_norm"] = _stats_json(pol["priv_stats"])
    return write_network(out_dir, name, nets, extra)


def export_students(
    students_path: Path, out_dir: Path = ARTIFACTS / "policies"
) -> list[dict[str, Any]]:
    """Export each student's adaptation module as `student_<k>`: a shared trunk (200 → 256 → 128,
    ELU), the ẑ head (128 → 8, tanh) and the grip head (128 → 2, sigmoid × grip_scale)."""
    from trustlap.student import load_students

    data = load_students(students_path)
    params = data["params"]["params"]
    k_members = int(np.asarray(params["adapt_out"]["kernel"]).shape[0])
    metas = []
    for k in range(k_members):
        take = lambda layer, k=k: (np.asarray(layer["kernel"][k]), np.asarray(layer["bias"][k]))  # noqa: E731
        trunk = []
        i = 0
        while f"adapt_{i}" in params:
            kern, bias = take(params[f"adapt_{i}"])
            trunk.append((f"adapt_{i}", kern, bias, "elu"))
            i += 1
        z_k, z_b = take(params["adapt_out"])
        g_k, g_b = take(params["grip_out"])
        nets = {
            "trunk": trunk,
            "z_head": [("adapt_out", z_k, z_b, "tanh")],
            "grip_head": [("grip_out", g_k, g_b, "sigmoid")],
        }
        extra = {
            "kind": "student",
            "member": k,
            "grip_scale": CONFIG.network.grip_scale,
            "training": data["info"],
        }
        metas.append(write_network(out_dir, f"student_{k}", nets, extra))
    return metas


def golden_students(
    students_path: Path, teacher_path: Path, n: int = 300, seed: int = 12
) -> dict[str, Any]:
    """Random (obs, history) → ensemble action, uncertainty, mean ẑ and grip."""
    from trustlap.confidence import ensemble_act
    from trustlap.env import Observation
    from trustlap.student import load_students, load_teacher

    teacher = load_teacher(teacher_path)
    params = jax.tree.map(jnp.asarray, load_students(students_path)["params"])
    k1, k2 = jax.random.split(jax.random.key(seed))
    obs = teacher.obs_stats.mean + 1.5 * jnp.sqrt(teacher.obs_stats.var) * jax.random.normal(
        k1, (n, CONFIG.obs.policy_size)
    )
    history = jax.random.uniform(k2, (n, CONFIG.obs.history_size), minval=-1.0, maxval=1.0)
    batch = Observation(
        policy=obs, privileged=jnp.zeros((n, CONFIG.obs.privileged_size)), history=history
    )
    out = jax.vmap(lambda o: ensemble_act(teacher, params, o))(batch)
    return {
        "n": n,
        "obs": _f32(obs),
        "history": _f32(history),
        "action": _f32(out.action),
        "uncertainty": _f32(out.uncertainty),
        "z_mean": _f32(out.z_mean),
        "grip_mean": _f32(out.grip_mean),
    }


# ---------------------------------------------------------------------------------------------
# Training records and metrics (SPEC 9.1, 9.3)
# ---------------------------------------------------------------------------------------------


def _read_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _downsample(rows: list[Any], limit: int = 500) -> list[Any]:
    if len(rows) <= limit:
        return rows
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, limit)).astype(int))
    return [rows[i] for i in idx]


def export_training(
    teacher_run: Path | None, blind_run: Path | None, out_dir: Path = ARTIFACTS
) -> None:
    """curriculum.json, population_curves.json, lineage.json and the checkpoint timeline."""
    from trustlap.train import load_policy

    if teacher_run and (teacher_run / "log.jsonl").exists():
        rows = _read_log(teacher_run)
        curriculum = [
            {
                "update": r["update"],
                "env_steps": r["env_steps_per_member"],
                "d": r["difficulty"],
                "success": r["curriculum_success"],
            }
            for r in rows
        ]
        _write_json(
            out_dir / "curriculum.json", {"run": "teacher", "points": _downsample(curriculum)}
        )
        points = [
            {
                "update": r["update"],
                "env_steps": r["env_steps_per_member"],
                "fitness": [round(f, 3) for f in r["fitness"]],
                "ids": r["member_ids"],
                "lr": r["lr"],
                "ent": r["ent_coef"],
                "completion": r["completion"],
            }
            for r in rows
        ]
        curves = {"members": len(rows[0]["fitness"]), "points": _downsample(points)}
        _write_json(out_dir / "population_curves.json", curves)
        lineage_path = teacher_run / "lineage.json"
        if lineage_path.exists():
            lineage = json.loads(lineage_path.read_text())
            lineage["winner_id"] = load_policy(teacher_run / "final.msgpack")["info"].get(
                "member_id"
            )
            _write_json(out_dir / "lineage.json", lineage)
        snaps = sorted((teacher_run / "snapshots").glob("update_*.msgpack"))
        index = []
        for i, snap in enumerate(snaps):
            info = export_actor_policy(snap, f"teacher_{i:02d}", out_dir / "checkpoints")[
                "training"
            ]
            index.append(
                {
                    "i": i,
                    "name": f"teacher_{i:02d}",
                    "update": info["update"],
                    "env_steps": info["env_steps"],
                    "difficulty": info["difficulty"],
                    "mean_return": info["return"],
                    "completion": info["completion"],
                }
            )
        if index:
            _write_json(out_dir / "checkpoints" / "index.json", {"checkpoints": index}, indent=1)
    if blind_run and (blind_run / "log.jsonl").exists():
        rows = _read_log(blind_run)
        curve = [
            {"update": r["update"], "env_steps": r["env_steps_per_member"], "d": r["difficulty"]}
            for r in rows
        ]
        _write_json(
            out_dir / "curriculum_blind.json", {"run": "blind", "points": _downsample(curve)}
        )


def _training_totals(runs: list[Path]) -> tuple[int, float]:
    """Total env steps (all members) and wall-clock minutes across training runs."""
    steps, minutes = 0, 0.0
    for run in runs:
        rows = _read_log(run)
        if not rows:
            continue
        last = rows[-1]
        if "env_steps" in last:  # PPO runs log the population total
            steps += last["env_steps"]
        else:  # student runs log per member
            steps += last["env_steps_per_member"] * len(last["z_loss"])
        minutes += last["wall_s"] / 60
    return int(steps), round(minutes, 1)


def readme_block(metrics: dict[str, Any]) -> str:
    """README headline bullets (SPEC 16.1), generated from metrics.json: never hand-typed."""
    h = metrics["headline"]
    s = metrics.get("suites", {})

    def comp(suite: str, driver: str) -> str:
        return f"{s[suite][driver]['completion']['value']:.0%}"

    lines = []
    if "max_parallel_cars" in h and "gpu_speedup_vs_cpu" in h:
        lines.append(
            f"- {h['max_parallel_cars']:,} cars simulated in parallel on one T4 GPU, "
            f"{h['gpu_speedup_vs_cpu']:,.0f}× faster than the same JAX code on its CPU"
        )
    if "train_env_steps_total" in h:
        steps = h["train_env_steps_total"]
        steps_text = f"{steps / 1e9:.1f}B" if steps >= 1e9 else f"{steps / 1e6:,.0f}M"
        lines.append(
            f"- PPO drivers with population-based training and an automatic curriculum: "
            f"{steps_text} env steps in {h['train_wall_minutes']:.0f} min · {h['train_hardware']}"
        )
    if {"puddles", "wet", "worn"} <= s.keys():
        pl = h.get("puddle_latency_ms", {})
        lat, noticed, trials = pl.get("median"), pl.get("noticed"), pl.get("trials")
        # Claim a reaction time only if most trials noticed (as on the site).
        mostly = lat is not None and (noticed is None or not trials or noticed >= trials / 2)
        lines.append(
            "- A learned adaptation module estimates road grip from 1 s of motion"
            + (f" (notices a puddle in ~{lat:.0f} ms)" if mostly else "")
            + ": laps completed vs a non-adaptive driver on unseen tracks: "
            f"puddles {comp('puddles', 'student')} vs {comp('puddles', 'blind')}, "
            f"wet {comp('wet', 'student')} vs {comp('wet', 'blind')}, "
            f"worn tires {comp('worn', 'student')} vs {comp('worn', 'blind')}"
            + (
                f". It does not notice a single puddle while driving through it "
                f"({noticed} of {trials} controlled trials)"
                if noticed is not None and trials and not mostly
                else ""
            )
        )
    red = h.get("student_crash_reduction_vs_blind_combined")
    if red and red.get("value") is not None:
        v, (lo, hi) = red["value"], red["ci"]
        word = "fewer" if v >= 0 else "more"
        lines.append(
            f"- Honest headline: in the everything-at-once conditions the student crashes "
            f"{abs(v):.0%} {word} than the non-adaptive driver "
            + (
                f"(95% CI {lo:.0%} to {hi:.0%})"
                if v >= 0
                else f"(95% CI {-hi:.0%} to {-lo:.0%} more)"
            )
        )
    cal = metrics.get("calibration", {})
    if "ece" in cal:
        lines.append(
            f"- Ensemble crash-risk estimate: calibrated (ECE {cal['ece']:.3f}) "
            "but weakly predictive "
            f"(AUROC {cal['auroc']:.2f})"
        )
    return "\n".join(lines)


def resume_bullets(metrics: dict[str, Any]) -> str:
    """Resume bullets (SPEC 18), generated from metrics.json so they match the site exactly."""
    h = metrics["headline"]
    s = metrics.get("suites", {})
    pct = lambda v: f"{v:.0%}"  # noqa: E731
    comp = lambda suite, d: pct(s[suite][d]["completion"]["value"])  # noqa: E731
    steps = h["train_env_steps_total"]
    steps_text = f"{steps / 1e9:.1f}B" if steps >= 1e9 else f"{steps / 1e6:,.0f}M"
    red, (lo, hi) = (
        h["student_crash_reduction_vs_blind_combined"]["value"],
        h["student_crash_reduction_vs_blind_combined"]["ci"],
    )
    cal = metrics["calibration"]
    pl = h.get("puddle_latency_ms", {})
    out = [
        f"- Built TrustLap, a JAX vehicle-dynamics simulator that runs {h['max_parallel_cars']:,} "
        f"cars in parallel on one T4 GPU ({h['gpu_speedup_vs_cpu']:,.0f}x its CPU), with a "
        "TypeScript port held to the Python physics by golden-file parity tests and served as "
        "a live in-browser demo (Web Worker, Canvas 2D, hand-written MLP runtime).",
        f"- Trained PPO driving agents with population-based training and an automatic "
        f"curriculum ({steps_text} environment steps in {h['train_wall_minutes']:.0f} min on "
        f"{h['train_hardware']}); an RMA-style adaptation module that infers road grip raised "
        f"lap completion over a non-adaptive baseline on unseen tracks in puddles "
        f"({comp('puddles', 'student')} vs {comp('puddles', 'blind')}) and rain "
        f"({comp('wet', 'student')} vs {comp('wet', 'blind')}).",
        f"- Evaluated on {metrics['protocol']['test_tracks']} held-out tracks across "
        f"{len(s)} condition suites with paired cluster-bootstrap CIs and reported the losses "
        "too: "
        + (
            f"{pct(red)} fewer crashes than the baseline in combined conditions "
            f"(95% CI {pct(lo)} to {pct(hi)})"
            if red >= 0
            else f"{pct(-red)} more crashes than the baseline in combined conditions"
        )
        + f", worse on dry roads ({comp('dry', 'student')} vs {comp('dry', 'blind')}), "
        f"a calibrated but weakly predictive ensemble risk estimate (ECE {cal['ece']:.3f}, "
        f"AUROC {cal['auroc']:.2f})"
        + (
            f", and no detection of single puddles ({pl['noticed']} of {pl['trials']})."
            if pl.get("noticed") == 0 and pl.get("trials")
            else "."
        ),
    ]
    return "\n".join(out)


def update_resume(metrics: dict[str, Any], path: Path = REPO / "docs" / "RESUME.md") -> None:
    path.write_text(
        "# Resume bullets\n\n"
        "Generated by `sim/trustlap/export.py` from `metrics.json` (don't edit by hand; "
        "re-run the export). Numbers match the site and README.\n\n"
        f"{resume_bullets(metrics)}\n"
    )


def update_readme(metrics: dict[str, Any], readme: Path = REPO / "README.md") -> None:
    text = readme.read_text()
    start, end = "<!-- metrics:start -->", "<!-- metrics:end -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        readme.write_text(f"{head}{start}\n{readme_block(metrics)}\n{end}{tail}")


def export_metrics(
    partial: Path, runs: list[Path], train_hardware: str, out_dir: Path = ARTIFACTS
) -> dict[str, Any]:
    """Assemble metrics.json from the evaluation output, the benchmark and the training logs.

    `train_hardware` describes where the runs trained (shown on the model card); the logs don't
    record the device, so the caller states it.
    """
    metrics = json.loads(partial.read_text())
    head = metrics["headline"]
    bench_path = RESULTS / "bench_t4.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        gpu = [r for r in bench["rows"] if r["impl"].startswith("jax-") and r["impl"] != "jax-cpu"]
        cpu = {r["n_cars"]: r for r in bench["rows"] if r["impl"] == "jax-cpu"}
        shared = [r for r in gpu if r["n_cars"] in cpu]
        if gpu:
            head["max_parallel_cars"] = max(r["n_cars"] for r in gpu)
        if shared:
            top = max(shared, key=lambda r: r["n_cars"])
            head["gpu_speedup_vs_cpu"] = round(
                top["steps_per_s"] / cpu[top["n_cars"]]["steps_per_s"], 1
            )
    steps, minutes = _training_totals(runs)
    head["train_env_steps_total"] = steps
    head["train_wall_minutes"] = minutes
    head["train_hardware"] = train_hardware
    metrics["git_sha"] = _git_sha()
    _write_json(out_dir / "metrics.json", metrics)
    cal = metrics["calibration"]
    # The browser's crash-risk model: the logistic + isotonic model when the evaluation fitted one,
    # else the earlier disagreement-only table.
    if "model" in cal:
        table = {"model": cal["model"], "tau_default": cal["tau_default"]}
    else:
        table = {"u": cal["table"]["u"], "p": cal["table"]["p"], "tau_default": cal["tau_default"]}
    _write_json(out_dir / "calibration.json", table)
    update_readme(metrics)
    update_resume(metrics)
    return metrics


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


def golden_policy(policy_path: Path, n: int = 500, seed: int = 11) -> dict[str, Any]:
    """Random raw observations (around the policy's own statistics) → mean action."""
    from trustlap.networks import ActorCritic
    from trustlap.ppo import RunningStats, policy_action
    from trustlap.train import load_policy

    pol = load_policy(policy_path)
    teacher = "encoder" in pol["params"]["params"]
    net = ActorCritic(privileged_actor=teacher)
    obs_stats = RunningStats(**{k: jnp.asarray(v) for k, v in pol["obs_stats"].items()})
    priv_stats = RunningStats(**{k: jnp.asarray(v) for k, v in pol["priv_stats"].items()})
    k1, k2 = jax.random.split(jax.random.key(seed))
    obs = obs_stats.mean + 1.5 * jnp.sqrt(obs_stats.var) * jax.random.normal(
        k1, (n, CONFIG.obs.policy_size)
    )
    priv = priv_stats.mean + 1.5 * jnp.sqrt(priv_stats.var) * jax.random.normal(
        k2, (n, CONFIG.obs.privileged_size)
    )
    from trustlap.env import Observation

    batch = Observation(
        policy=obs, privileged=priv, history=jnp.zeros((n, CONFIG.obs.history_size))
    )
    actions = jax.vmap(lambda o: policy_action(net, pol["params"], obs_stats, priv_stats, o))(batch)
    return {"n": n, "obs": _f32(obs), "privileged": _f32(priv), "action": _f32(actions)}


def golden_policy_closed_loop(
    policy_path: Path, bank: TrackBank, steps: int = 400
) -> dict[str, Any]:
    """The policy drives from the start line on the golden tracks (no noise): 20 cases ×
    `steps` steps → progress at checkpoints and whether it crashed (SPEC 9.4)."""
    from trustlap.networks import ActorCritic
    from trustlap.ppo import RunningStats, policy_action
    from trustlap.train import load_policy

    pol = load_policy(policy_path)
    teacher = "encoder" in pol["params"]["params"]
    net = ActorCritic(privileged_actor=teacher)
    obs_stats = RunningStats(**{k: jnp.asarray(v) for k, v in pol["obs_stats"].items()})
    priv_stats = RunningStats(**{k: jnp.asarray(v) for k, v in pol["priv_stats"].items()})
    b = TrackBank(*(jnp.asarray(a) for a in bank))
    n_tracks = int(bank.width.shape[0])
    zero_noise = jnp.zeros(CONFIG.obs.policy_size - 1)

    @jax.jit
    def run(cond: Conditions, tid: jax.Array) -> tuple[jax.Array, jax.Array]:
        state = start_state(b, tid, 0.0, CONFIG.episode.eval_speed_mps, cond, 1.0, CONFIG, jnp)

        def body(
            carry: tuple[EnvState, jax.Array], _: Any
        ) -> tuple[tuple[EnvState, jax.Array], jax.Array]:
            st, crashed = carry
            obs, sensed = observe(st, b, zero_noise, CONFIG, jnp)
            st = st._replace(last_proprio=sensed)
            a = policy_action(net, pol["params"], obs_stats, priv_stats, obs)
            new, _, info = core_step(st, a, jnp.zeros(2), b, CONFIG, jnp)
            new = jax.tree.map(lambda x, y: jnp.where(crashed, y, x), new, st)
            return (new, crashed | info.terminated & ~info.lap_completed), new.progress_m

        (final, crashed), progress = jax.lax.scan(
            body, (state, jnp.array(False)), None, length=steps
        )
        return progress, crashed

    cases = []
    for i in range(20):
        suite = ("dry", "wet", "puddles", "dry")[i % 4]
        tid = i % n_tracks
        cond = suite_conditions(jax.random.key(100 + i), suite, b.length[tid], b.width[tid])
        progress, crashed = run(cond, jnp.int32(tid))
        cases.append(
            {
                "suite": suite,
                "track_index": tid,
                "conditions": _soa(cond),
                "progress_m": _f32(progress[jnp.array([99, 199, steps - 1])]),
                "crashed": bool(crashed),
            }
        )
    return {"steps": steps, "checkpoints": [100, 200, steps], "cases": cases}


def export_policy_golden(
    policies: dict[str, Path], closed_loop: str | None, out_dir: Path = GOLDEN
) -> None:
    """Parity fixtures for exported networks: `policy_<name>.json` and `closed_loop.json`."""
    jax.config.update("jax_default_matmul_precision", "highest")
    meta = {"git_sha": _git_sha(), "physics_hash": physics_hash(), "jax_version": jax.__version__}
    for name, path in policies.items():
        _write_json(
            out_dir / f"policy_{name}.json", {**meta, "policy": name, **golden_policy(path)}
        )
    if closed_loop:
        bank = make_bank([generate_track(s) for s in GOLDEN_TRACK_SEEDS])
        data = golden_policy_closed_loop(policies[closed_loop], bank)
        _write_json(
            out_dir / "closed_loop.json",
            {**meta, "policy": closed_loop, "track_seeds": list(GOLDEN_TRACK_SEEDS), **data},
        )
    for p in sorted(out_dir.glob("policy_*.json")) + [out_dir / "closed_loop.json"]:
        if p.exists():
            print(f"golden → {p.relative_to(REPO)} ({p.stat().st_size / 1024:.0f} KB)")


def golden_risk(bank: TrackBank, n: int = 64, seed: int = 7) -> dict[str, Any]:
    """Crash-risk parity: a model fit on synthetic data, its probabilities for random features,
    and the features computed from car states on the golden tracks."""
    from trustlap.confidence import fit_risk, risk_features, risk_predict, risk_to_dict

    rng = np.random.default_rng(seed)
    m = 4000
    x = np.stack(
        [
            rng.uniform(0, 0.3, m),
            rng.uniform(0, 60, m),
            rng.uniform(0, 0.05, m),
            rng.uniform(0, 2.0, m),
        ],
        axis=1,
    )
    logit = -4 + 6 * x[:, 0] + 0.02 * x[:, 1] + 2.5 * x[:, 3]
    y = (rng.uniform(size=m) < 1 / (1 + np.exp(-logit))).astype(float)
    model = fit_risk(x, y)
    feats = x[:n]
    probs = np.asarray(risk_predict(model, feats.astype(np.float32), np))
    # Features from states: random track, centerline index, velocity, disagreement and felt grip.
    states = []
    for _ in range(n):
        t = int(rng.integers(len(bank.seeds)))
        idx = int(rng.integers(int(bank.n_points[t])))
        vx, vy = float(rng.uniform(0, 60)), float(rng.uniform(-3, 3))
        u, mu = float(rng.uniform(0, 0.3)), float(rng.uniform(0.05, 1.2))
        car = types.SimpleNamespace(vx_mps=np.float32(vx), vy_mps=np.float32(vy))
        st = types.SimpleNamespace(car=car, track_id=t, idx=idx)
        f = risk_features(st, bank, np.float32(u), np.float32(mu), np)
        states.append(
            {"track": t, "idx": idx, "vx": vx, "vy": vy, "u": u, "mu": mu, "f": f.tolist()}
        )
    return {
        "model": risk_to_dict(model),
        "cases": [{"f": f.tolist(), "p": float(p)} for f, p in zip(feats, probs, strict=True)],
        "states": states,
    }


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
    _write_json(
        out_dir / "risk.json",
        {**meta, "track_seeds": list(GOLDEN_TRACK_SEEDS), **golden_risk(bank)},
    )
    for p in sorted(out_dir.glob("*.json")):
        print(f"golden → {p.relative_to(REPO)} ({p.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--golden", action="store_true", help="also write parity fixtures")
    parser.add_argument("--only-golden", action="store_true", help="write only parity fixtures")
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="export a trained policy (e.g. blind=runs/blind/final.msgpack); repeatable",
    )
    parser.add_argument("--closed-loop", default=None, help="policy NAME for closed_loop.json")
    parser.add_argument("--students", type=Path, default=None, help="students.msgpack to export")
    parser.add_argument(
        "--teacher", type=Path, default=None, help="teacher policy for the students"
    )
    parser.add_argument("--teacher-run", type=Path, default=None, help="teacher run dir")
    parser.add_argument("--blind-run", type=Path, default=None, help="blind run dir")
    parser.add_argument("--metrics", type=Path, default=None, help="evaluate.py output to publish")
    parser.add_argument(
        "--train-hardware",
        default=None,
        help='where the runs trained, e.g. "NVIDIA T4 GPU (Kaggle)"; required with --metrics',
    )
    args = parser.parse_args()
    policies = dict(item.split("=", 1) for item in args.policy)
    policies = {k: Path(v) for k, v in policies.items()}
    for name, path in policies.items():
        meta = export_actor_policy(path, name)
        print(f"policy → policies/{name}.json ({meta['bytes'] / 1024:.0f} KB weights)")
    if args.students:
        metas = export_students(args.students)
        print(f"students → {len(metas)} × policies/student_<k>.json")
    if args.teacher_run or args.blind_run:
        export_training(args.teacher_run, args.blind_run)
    if args.metrics and not args.train_hardware:
        parser.error("--metrics needs --train-hardware (the training logs don't record the device)")
    if args.metrics:
        student_run = args.students.parent if args.students else None
        runs = [p for p in (args.blind_run, args.teacher_run, student_run) if p]
        export_metrics(args.metrics, runs, args.train_hardware)
        print("metrics → metrics.json, calibration.json")
    if not args.only_golden:
        export_artifacts()
    if args.golden or args.only_golden:
        export_golden()
    if policies and (args.golden or args.only_golden):
        export_policy_golden(policies, args.closed_loop)
    if args.students and args.teacher and (args.golden or args.only_golden):
        meta = {
            "git_sha": _git_sha(),
            "physics_hash": physics_hash(),
            "jax_version": jax.__version__,
        }
        _write_json(
            GOLDEN / "students.json", {**meta, **golden_students(args.students, args.teacher)}
        )
        print("golden → web/tests/parity/golden/students.json")


if __name__ == "__main__":
    main()
