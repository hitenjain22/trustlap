"""Vehicle dynamics (SPEC 6.3), ported operation by operation to `web/src/sim/dynamics.ts`.

A dynamic bicycle model: one front and one rear "axle" with simplified Pacejka tires, a
friction ellipse, drag, rolling resistance, wind with gusts, a first-order steering actuator
and an action delay line. One control step (50 ms) = 4 physics substeps (12.5 ms).

Every function here handles ONE car and is written against an array module `xp`
(`numpy` or `jax.numpy`): JAX runs many cars with `jax.vmap`, and the NumPy benchmark baseline
calls the very same code in a Python loop. Keep it free of Python control flow on values.

Units: meters, seconds, kilograms, newtons, radians. Body frame: x forward, y left.
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

from trustlap.conditions import Conditions, puddle_grip
from trustlap.config import CONFIG, Config


class CarState(NamedTuple):
    """Physical state of one car."""

    x_m: Any  # [] world position
    y_m: Any  # []
    yaw_rad: Any  # [] heading, wrapped to (−π, π]
    vx_mps: Any  # [] body-frame forward speed, ≥ 0 (no reverse gear)
    vy_mps: Any  # [] body-frame lateral speed (+ = left)
    yaw_rate_radps: Any  # []
    steer_rad: Any  # [] actual front-wheel angle (lags the command)
    action_buffer: Any  # [4, 2] previous actions, newest first
    gust_mps: Any  # [2] world-frame gust (Ornstein–Uhlenbeck)
    odometer_m: Any  # [] distance driven this episode (tire wear)


def initial_car_state(
    x_m: Any, y_m: Any, yaw_rad: Any, speed_mps: Any, cfg: Config = CONFIG, xp: ModuleType = np
) -> CarState:
    """A car at rest on the given pose, rolling forward at `speed_mps` with wheels straight."""
    f = xp.float32
    zero = xp.zeros((), f)
    return CarState(
        x_m=zero + x_m,
        y_m=zero + y_m,
        yaw_rad=zero + yaw_rad,
        vx_mps=zero + speed_mps,
        vy_mps=zero,
        yaw_rate_radps=zero,
        steer_rad=zero,
        action_buffer=xp.zeros((cfg.conditions.delay_buffer_len, 2), f),
        gust_mps=xp.zeros(2, f),
        odometer_m=zero,
    )


def wrap_angle(a: Any, xp: ModuleType = np) -> Any:
    """Wrap an angle to (−π, π]."""
    two_pi = 2.0 * np.pi
    return a - two_pi * xp.ceil((a - np.pi) / two_pi)


def axle_positions(state: CarState, cfg: Config = CONFIG, xp: ModuleType = np) -> tuple[Any, Any]:
    """World positions [2] of the front and rear axle centers."""
    c, s = xp.cos(state.yaw_rad), xp.sin(state.yaw_rad)
    front = xp.stack([state.x_m + cfg.car.lf_m * c, state.y_m + cfg.car.lf_m * s])
    rear = xp.stack([state.x_m - cfg.car.lr_m * c, state.y_m - cfg.car.lr_m * s])
    return front, rear


def tire_grip_factor(
    odometer_m: Any, cond: Conditions, cfg: Config = CONFIG, xp: ModuleType = np
) -> Any:
    """μ_tire = max(0.6, 1 − w · km driven)."""
    return xp.maximum(
        cfg.conditions.tire_grip_floor, 1.0 - cond.wear_per_km * (odometer_m / 1000.0)
    )


def axle_grip(
    state: CarState, cond: Conditions, cfg: Config = CONFIG, xp: ModuleType = np
) -> tuple[Any, Any]:
    """Friction coefficient under the front and rear axle: μ_base · μ_tire · puddle field."""
    front, rear = axle_positions(state, cfg, xp)
    base = cond.mu_base * tire_grip_factor(state.odometer_m, cond, cfg, xp)
    return base * puddle_grip(front, cond, xp), base * puddle_grip(rear, cond, xp)


def executed_action(state: CarState, action: Any, cond: Conditions, xp: ModuleType = np) -> Any:
    """The action that actually reaches the car: the current one if D = 0, else buffer[D − 1]."""
    slots = xp.arange(state.action_buffer.shape[0])
    pick = (slots == (cond.delay_steps - 1)).astype(action.dtype)
    delayed = xp.sum(state.action_buffer * pick[:, None], axis=0)
    return xp.where(cond.delay_steps == 0, action, delayed)


def lateral_tire_force(
    alpha_rad: Any, mu: Any, fz_n: Any, cfg: Config = CONFIG, xp: ModuleType = np
) -> Any:
    """Simplified Pacejka: Fy = −μ·Fz·sin(C·atan(Bα − E(Bα − atan(Bα)))). Opposes the slip."""
    b, c, e = cfg.car.tire_b, cfg.car.tire_c, cfg.car.tire_e
    ba = b * alpha_rad
    return -mu * fz_n * xp.sin(c * xp.arctan(ba - e * (ba - xp.arctan(ba))))


def substep(
    state: CarState,
    cond: Conditions,
    a_steer: Any,
    a_long: Any,
    cfg: Config = CONFIG,
    xp: ModuleType = np,
) -> CarState:
    """Advance one car by one physics substep (dt = 12.5 ms). Actions arrive delayed and clipped."""
    car = cfg.car
    dt = cfg.time.physics_dt_s
    g = car.gravity_mps2
    wheelbase = car.wheelbase_m
    mass = car.mass_kg * cond.mass_mult
    iz = car.yaw_inertia_kgm2 * cond.mass_mult
    vx, vy, r, delta, yaw = (
        state.vx_mps,
        state.vy_mps,
        state.yaw_rate_radps,
        state.steer_rad,
        state.yaw_rad,
    )

    # 1–2. Steering actuator: first-order lag toward the command, rate-limited.
    steer_cmd = a_steer * car.steer_max_rad
    steer_rate = xp.clip(
        (steer_cmd - delta) / cond.steer_tau_s, -car.steer_rate_max_radps, car.steer_rate_max_radps
    )

    # 3. Static axle loads.
    fz_f = mass * g * car.lr_m / wheelbase
    fz_r = mass * g * car.lf_m / wheelbase
    mu_f, mu_r = axle_grip(state, cond, cfg, xp)

    # 4. Longitudinal forces: rear-wheel drive (power-limited), brakes split 60/40.
    #    Drive and brake hardware limits are fixed (nominal mass), so a heavier car is slower.
    throttle = xp.maximum(a_long, 0.0)
    brake = xp.maximum(-a_long, 0.0) * (vx > car.brake_min_speed_mps)
    f_drive = throttle * xp.minimum(
        car.drive_force_max_n, car.power_w * cond.power_mult / xp.maximum(vx, 1.0)
    )
    f_brake = brake * (car.brake_force_max_g * car.mass_kg * g) * cond.brake_mult
    fx_f = -f_brake * car.brake_front_frac
    fx_r = f_drive - f_brake * (1.0 - car.brake_front_frac)
    fx_f_max = mu_f * fz_f
    fx_r_max = mu_r * fz_r
    fx_f = xp.clip(fx_f, -fx_f_max, fx_f_max)
    fx_r = xp.clip(fx_r, -fx_r_max, fx_r_max)

    # 5–6. Slip angles and lateral tire forces, reduced by the friction ellipse.
    vx_slip = xp.maximum(vx, car.min_slip_speed_mps)
    alpha_f = xp.arctan2(vy + car.lf_m * r, vx_slip) - delta
    alpha_r = xp.arctan2(vy - car.lr_m * r, vx_slip)
    use_f = fx_f / fx_f_max
    use_r = fx_r / fx_r_max
    fy_f = lateral_tire_force(alpha_f, mu_f, fz_f, cfg, xp) * xp.sqrt(
        xp.maximum(0.0, 1.0 - use_f * use_f)
    )
    fy_r = lateral_tire_force(alpha_r, mu_r, fz_r, cfg, xp) * xp.sqrt(
        xp.maximum(0.0, 1.0 - use_r * use_r)
    )

    # 7. Aerodynamics: air velocity relative to the car, in the body frame.
    cos_y, sin_y = xp.cos(yaw), xp.sin(yaw)
    wind_x = cond.wind_mps[0] + state.gust_mps[0]
    wind_y = cond.wind_mps[1] + state.gust_mps[1]
    wind_bx = cos_y * wind_x + sin_y * wind_y
    wind_by = -sin_y * wind_x + cos_y * wind_y
    u = vx - wind_bx
    v = vy - wind_by
    f_drag = -0.5 * car.air_density_kgpm3 * cond.drag_area_m2 * u * xp.abs(u)
    f_side = -0.5 * car.air_density_kgpm3 * car.side_area_coef_m2 * v * xp.abs(v)

    # 8. Rolling resistance (tanh makes it vanish smoothly at rest).
    f_roll = -cond.rolling_coef * mass * g * xp.tanh(vx / car.rolling_tanh_speed_mps)

    # 9. Body-frame equations of motion.
    cos_d, sin_d = xp.cos(delta), xp.sin(delta)
    vx_dot = (fx_r + fx_f * cos_d - fy_f * sin_d + f_drag + f_roll) / mass + vy * r
    vy_dot = (fy_r + fy_f * cos_d + fx_f * sin_d + f_side) / mass - vx * r
    r_dot = (
        car.lf_m * (fy_f * cos_d + fx_f * sin_d)
        - car.lr_m * fy_r
        - car.center_of_pressure_offset_m * f_side
    ) / iz

    # 10. Semi-implicit Euler: velocities first (no reverse gear) ...
    vx = xp.maximum(vx + vx_dot * dt, 0.0)
    vy = vy + vy_dot * dt
    r = r + r_dot * dt
    delta = xp.clip(delta + steer_rate * dt, -car.steer_max_rad, car.steer_max_rad)

    # 6.3.4 Low-speed blending toward the kinematic bicycle (avoids the slip-angle blow-up).
    speed = xp.sqrt(vx * vx + vy * vy)
    lam = xp.clip((speed - car.low_speed_blend_start_mps) / car.low_speed_blend_width_mps, 0.0, 1.0)
    r_kin = vx * xp.tan(delta) / wheelbase
    vy_kin = r_kin * car.lr_m
    vy = lam * vy + (1.0 - lam) * vy_kin
    r = lam * r + (1.0 - lam) * r_kin

    # ... then positions with the new velocities.
    x = state.x_m + (vx * cos_y - vy * sin_y) * dt
    y = state.y_m + (vx * sin_y + vy * cos_y) * dt
    yaw = wrap_angle(yaw + r * dt, xp)
    odometer = state.odometer_m + xp.sqrt(vx * vx + vy * vy) * dt

    return state._replace(
        x_m=x,
        y_m=y,
        yaw_rad=yaw,
        vx_mps=vx,
        vy_mps=vy,
        yaw_rate_radps=r,
        steer_rad=delta,
        odometer_m=odometer,
    )


def control_step(
    state: CarState,
    cond: Conditions,
    action: Any,
    gust_noise: Any,
    cfg: Config = CONFIG,
    xp: ModuleType = np,
) -> CarState:
    """Advance one car by one control step (50 ms = 4 substeps).

    `action` [2] = (steer, longitudinal) in [−1, 1] (clipped here). `gust_noise` [2] is a
    standard-normal draw for the gust update; passing it in keeps this function deterministic.
    """
    action = xp.clip(action, -1.0, 1.0)
    applied = executed_action(state, action, cond, xp)
    for _ in range(cfg.time.substeps):
        state = substep(state, cond, applied[0], applied[1], cfg, xp)
    buffer = xp.concatenate([action[None, :], state.action_buffer[:-1]], axis=0)
    dt = cfg.time.control_dt_s
    theta = cfg.conditions.gust_theta_per_s
    gust = (
        state.gust_mps
        - theta * state.gust_mps * dt
        + cond.gust_sigma_mps * math.sqrt(dt) * gust_noise
    )
    return state._replace(action_buffer=buffer, gust_mps=gust)
