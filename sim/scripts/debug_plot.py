"""Plot one scripted lap: track, puddles, path coloured by speed, per-step trace (SPEC M1.8, 15).

    python scripts/debug_plot.py --track-seed 200000 --rain 0.8 --out runs/debug_lap.png

Also writes a CSV next to the PNG with every step (state, action, reward, μ per axle, events).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from trustlap.conditions import nominal_conditions, with_rain  # noqa: E402
from trustlap.config import CONFIG  # noqa: E402
from trustlap.env import core_step, curvature_at, start_state  # noqa: E402
from trustlap.scripted import scripted_action  # noqa: E402
from trustlap.track import boundaries, generate_track, make_bank  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--track-seed", type=int, default=200000)
    parser.add_argument("--rain", type=float, default=0.8)
    parser.add_argument("--puddles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0, help="puddle placement seed")
    parser.add_argument("--out", type=Path, default=Path("runs/debug_lap.png"))
    args = parser.parse_args()

    track = generate_track(args.track_seed)
    bank = make_bank([track])
    c = CONFIG.conditions
    rng = np.random.default_rng(args.seed)
    p = c.puddle_slots
    cond = with_rain(nominal_conditions(), args.rain)._replace(
        puddle_s_m=rng.uniform(
            c.puddle_start_margin_m, track.length_m - c.puddle_end_margin_m, p
        ).astype(np.float32),
        puddle_lateral_m=rng.uniform(-0.35, 0.35, p).astype(np.float32) * np.float32(track.width_m),
        puddle_mult=np.full(p, 0.4, np.float32),
        puddle_radius_m=np.full(p, 8.0, np.float32),
        puddle_active=(np.arange(p) < args.puddles).astype(np.float32),
    )
    state = start_state(
        bank, np.int32(0), np.float32(0.0), np.float32(CONFIG.episode.eval_speed_mps), cond, 1.0
    )
    cond = state.cond  # with puddle world positions filled in

    rows = []
    for _ in range(CONFIG.episode.max_steps):
        action = scripted_action(state, bank)
        state, reward, info = core_step(state, action, np.zeros(2, np.float32), bank)
        car = state.car
        slip_rear = math.degrees(
            math.atan2(car.vy_mps - CONFIG.car.lr_m * car.yaw_rate_radps, max(car.vx_mps, 0.5))
        )
        rows.append(
            {
                "step": int(info.steps),
                "s_m": float(state.s_m),
                "progress_m": float(state.progress_m),
                "x_m": float(car.x_m),
                "y_m": float(car.y_m),
                "speed_mps": float(info.speed_mps),
                "yaw_rate_radps": float(car.yaw_rate_radps),
                "rear_slip_deg": slip_rear,
                "a_steer": float(action[0]),
                "a_long": float(action[1]),
                "mu_front": float(info.mu_front),
                "mu_rear": float(info.mu_rear),
                "curvature": float(curvature_at(bank, state.track_id, state.s_m)),
                "reward": float(reward),
                "event": "lap"
                if info.lap_completed
                else "crash"
                if info.crashed
                else "stall"
                if info.stalled
                else "",
            }
        )
        if info.terminated:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    col = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "event"}
    outcome = rows[-1]["event"] or "time limit"
    lap_time = len(rows) * CONFIG.time.control_dt_s

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.15, 1])
    ax_map = fig.add_subplot(grid[:, 0])
    left, right = boundaries(track)
    for b in (left, right):
        closed = np.vstack([b, b[:1]])
        ax_map.plot(closed[:, 0], closed[:, 1], color="0.35", lw=0.8)
    for j in range(p):
        if cond.puddle_active[j] > 0:
            ax_map.add_patch(
                plt.Circle(
                    cond.puddle_xy_m[j],
                    float(cond.puddle_radius_m[j]),
                    color="tab:blue",
                    alpha=0.25,
                )
            )
    pts = np.stack([col["x_m"], col["y_m"]], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=2.2)
    lc.set_array(col["speed_mps"][:-1] * 3.6)
    ax_map.add_collection(lc)
    fig.colorbar(lc, ax=ax_map, label="speed (km/h)", shrink=0.6)
    ax_map.plot(*track.centerline[0], "k>", ms=9)
    ax_map.set_aspect("equal")
    ax_map.set_title(
        f"Track seed {args.track_seed} ({track.length_m:.0f} m), "
        f"rain {args.rain:.0%}, {args.puddles} puddles\n"
        f"scripted driver: {outcome} in {lap_time:.1f} s"
    )
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")

    s = col["progress_m"]
    ax_v = fig.add_subplot(grid[0, 1])
    ax_v.plot(s, col["speed_mps"] * 3.6, color="tab:green")
    ax_v.set_ylabel("speed (km/h)")
    # Curvature gets its own panel rather than a second y-axis (SPEC 11.6: never a dual axis).
    ax_c = fig.add_subplot(grid[1, 1], sharex=ax_v)
    ax_c.plot(s, np.abs(col["curvature"]), color="0.4")
    ax_c.set_ylabel("|curvature| (1/m)")
    ax_a = fig.add_subplot(grid[2, 1], sharex=ax_v)
    ax_a.plot(s, col["a_long"], color="tab:orange", label="throttle (+) / brake (−)")
    ax_a.plot(
        s,
        col["mu_rear"] / CONFIG.conditions.mu_dry.nominal,
        color="tab:blue",
        label="rear grip / dry grip",
    )
    ax_a.plot(
        s,
        np.clip(np.abs(col["rear_slip_deg"]) / 10, 0, 1.5),
        color="tab:red",
        lw=0.8,
        label="|rear slip| / 10°",
    )
    ax_a.set_xlabel("distance along track (m)")
    ax_a.legend(loc="lower right", fontsize=8)
    fig.savefig(args.out, dpi=130)
    print(f"{outcome} in {lap_time:.1f} s; wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
