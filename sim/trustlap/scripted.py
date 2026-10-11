"""Scripted baseline driver: a "competent human" that knows the true grip (SPEC M1.6, 8.1).

* Steering: pure pursuit toward a centerline point `clip(T · speed, min, max)` m ahead, plus
  yaw-rate damping (countersteer when the car rotates faster than the pursued arc).
* Speed: a braking-aware profile. For each point up to 160 m ahead, the fastest speed that
  corner allows is sqrt(grip_use · μ · g / |κ|); the target now is the lowest of
  sqrt(v_corner² + 2 · a_brake · distance) over those points.
* Longitudinal: P-controller on speed (a requested acceleration, converted to throttle or brake
  through the car's force limits), capped like traction control / ABS so the tires keep
  some sideways grip.

It reads true μ (base grip, tire wear, puddles), so it is a strong baseline on grip changes
but has no answer to steering lag or action delay. Ported to `web/src/sim/scripted.ts`.
One car, written against `xp` like `dynamics.py`.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from trustlap.conditions import puddle_grip
from trustlap.config import CONFIG, Config
from trustlap.dynamics import axle_grip, tire_grip_factor
from trustlap.env import EnvState, curvature_at, point_at
from trustlap.track import TrackBank


def scripted_action(
    state: EnvState, bank: TrackBank, cfg: Config = CONFIG, xp: ModuleType = np
) -> Any:
    """Action [2] = (steer, longitudinal) for one car."""
    sc, car_cfg = cfg.scripted, cfg.car
    car, cond = state.car, state.cond
    g = car_cfg.gravity_mps2
    tid = state.track_id
    f = car.x_m.dtype
    speed = xp.sqrt(car.vx_mps * car.vx_mps + car.vy_mps * car.vy_mps)

    # Steering: pure pursuit.
    look = xp.clip(speed * sc.lookahead_time_s, sc.lookahead_min_m, sc.lookahead_max_m)
    target, _ = point_at(bank, tid, state.s_m + look, xp)
    dx, dy = target[0] - car.x_m, target[1] - car.y_m
    c, s = xp.cos(car.yaw_rad), xp.sin(car.yaw_rad)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    alpha = xp.arctan2(local_y, local_x)
    path_curvature = 2.0 * xp.sin(alpha) / look
    steer = xp.arctan(car_cfg.wheelbase_m * path_curvature)
    # Yaw damping: countersteer when the car rotates faster than the pursued arc asks for.
    steer = steer + sc.yaw_damping_s * (speed * path_curvature - car.yaw_rate_radps)
    a_steer = xp.clip(steer / car_cfg.steer_max_rad, -1.0, 1.0)

    # Speed profile from true grip along the road ahead.
    n_samples = int(sc.profile_horizon_m / sc.profile_step_m) + 1
    dists = xp.arange(n_samples, dtype=f) * sc.profile_step_m
    ahead_s = state.s_m + dists
    kappa = xp.abs(curvature_at(bank, tid, ahead_s, xp))
    pts, _ = point_at(bank, tid, ahead_s, xp)
    puddles = xp.stack([puddle_grip(pts[k], cond, xp) for k in range(n_samples)])
    mu_ahead = cond.mu_base * tire_grip_factor(car.odometer_m, cond, cfg, xp) * puddles
    v_corner = xp.sqrt(sc.grip_use * mu_ahead * g / xp.maximum(kappa, 1e-4))
    v_allow = xp.sqrt(v_corner * v_corner + 2.0 * sc.brake_use * mu_ahead * g * dists)
    v_target = xp.minimum(xp.min(v_allow), sc.max_speed_mps)

    # Longitudinal: P-control, then traction control (throttle) and ABS (brakes) that only use
    # the grip cornering leaves free (friction circle): long ≤ sqrt(1 − (lateral use)²).
    mu_f, mu_r = axle_grip(car, cond, cfg, xp)
    mass = car_cfg.mass_kg * cond.mass_mult
    fz_f = mass * g * car_cfg.lr_m / car_cfg.wheelbase_m
    fz_r = mass * g * car_cfg.lf_m / car_cfg.wheelbase_m
    kappa_here = xp.abs(curvature_at(bank, tid, state.s_m, xp))
    lateral_accel = xp.maximum(xp.abs(car.vx_mps * car.yaw_rate_radps), speed * speed * kappa_here)
    lat_use_f = xp.minimum(lateral_accel / (mu_f * g), 1.0)
    lat_use_r = xp.minimum(lateral_accel / (mu_r * g), 1.0)
    free_f = sc.traction_use * xp.sqrt(1.0 - lat_use_f * lat_use_f)
    free_r = sc.traction_use * xp.sqrt(1.0 - lat_use_r * lat_use_r)
    drive_max = xp.minimum(
        car_cfg.drive_force_max_n, car_cfg.power_w * cond.power_mult / xp.maximum(car.vx_mps, 1.0)
    )
    brake_force = car_cfg.brake_force_max_g * car_cfg.mass_kg * g * cond.brake_mult
    # Requested acceleration (m/s²) → throttle or brake fraction through the actual force limits.
    accel = sc.speed_gain * (v_target - car.vx_mps)
    a_long = xp.where(accel >= 0.0, accel * mass / drive_max, accel * mass / brake_force)
    throttle_cap = free_r * mu_r * fz_r / drive_max
    brake_cap = xp.minimum(
        free_f * mu_f * fz_f / (brake_force * car_cfg.brake_front_frac),
        free_r * mu_r * fz_r / (brake_force * (1.0 - car_cfg.brake_front_frac)),
    )
    a_long = xp.clip(a_long, -xp.minimum(brake_cap, 1.0), xp.minimum(throttle_cap, 1.0))
    return xp.stack([a_steer, a_long]).astype(f)
