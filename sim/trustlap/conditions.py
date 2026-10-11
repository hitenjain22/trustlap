"""Per-episode driving conditions (SPEC 6.2): weather, puddles, car changes, lag, noise, wind.

`Conditions` is a NamedTuple of per-car values (a JAX pytree). Sampling uses `jax.random` and
is curriculum-aware: at difficulty d each parameter is drawn from bounds that move linearly
from its nominal value (d = 0) to its full training range (d = 1).

Puddles are sampled in track coordinates (arclength from the episode start, lateral offset);
`env.py` converts them to world positions because that needs the track geometry.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.config import CONFIG, Config, Range


class Conditions(NamedTuple):
    """Conditions for one car (add a leading batch axis for many cars)."""

    rain: Any  # [] 0..1
    mu_dry: Any  # [] dry-road friction coefficient
    mu_base: Any  # [] μ_dry · (1 − 0.45 · rain)
    wear_per_km: Any  # [] tire grip lost per km driven
    mass_mult: Any  # []
    drag_area_m2: Any  # [] CdA
    rolling_coef: Any  # [] Crr
    power_mult: Any  # []
    brake_mult: Any  # []
    steer_tau_s: Any  # [] steering actuator time constant
    delay_steps: Any  # [] int32, action delay in control steps (0..4)
    obs_noise: Any  # [] Gaussian σ added to normalized observations
    wind_mps: Any  # [2] mean wind, world frame
    gust_sigma_mps: Any  # [] OU gust strength
    puddle_s_m: Any  # [P] arclength ahead of the episode start
    puddle_lateral_m: Any  # [P] signed offset from the centerline (+ = left)
    puddle_mult: Any  # [P] grip multiplier at the puddle center
    puddle_radius_m: Any  # [P]
    puddle_active: Any  # [P] float 0/1
    puddle_xy_m: Any  # [P, 2] world position (filled by env.place_puddles)


def _uniform(key: jax.Array, r: Range, d: jax.Array) -> jax.Array:
    lo, hi = r.at_difficulty(d)  # works elementwise on arrays: affine in d
    return jax.random.uniform(key, (), minval=lo, maxval=hi)


# Parameters a "stretch" reset can push to the harder edge of their range (SPEC 7.4).
STRETCH_FIELDS = (
    ("rain", "rain"),
    ("mu_dry", "mu_dry"),
    ("wear_per_km", "tire_wear_per_km"),
    ("mass_mult", "mass_mult"),
    ("drag_area_m2", "drag_area_m2"),
    ("rolling_coef", "rolling_coef"),
    ("power_mult", "power_mult"),
    ("brake_mult", "brake_mult"),
    ("steer_tau_s", "steer_tau_s"),
    ("obs_noise", "obs_noise"),
    ("gust_sigma_mps", "gust_sigma_mps"),
    ("wind_mps", "wind_speed_mps"),
    ("delay_steps", None),
)


def _hard_bound(r: Range, d: jax.Array) -> jax.Array:
    """The bound at difficulty d that is farther from nominal (the harder one)."""
    lo, hi = r.at_difficulty(d)
    return lo if abs(r.lo - r.nominal) > abs(r.hi - r.nominal) else hi


def sample_conditions(
    key: jax.Array,
    difficulty: jax.Array,
    track_length_m: jax.Array,
    track_width_m: jax.Array,
    cfg: Config = CONFIG,
    stretch: jax.Array | bool = False,
) -> Conditions:
    """Draw conditions for one car at curriculum difficulty `difficulty` ∈ [0, 1].

    With `stretch`, one randomly chosen parameter is set to the harder edge of its range at
    this difficulty ("stretch exposure", SPEC 7.4).
    """
    cond = _sample(key, difficulty, track_length_m, track_width_m, cfg)
    c = cfg.conditions
    d = jnp.asarray(difficulty, jnp.float32)
    pick = jax.random.randint(jax.random.fold_in(key, 7919), (), 0, len(STRETCH_FIELDS))
    updates = {}
    for i, (field, range_name) in enumerate(STRETCH_FIELDS):
        on = jnp.asarray(stretch) & (pick == i)
        current = getattr(cond, field)
        if field == "delay_steps":
            edge = jnp.floor(c.delay_max_steps * d + 0.5).astype(jnp.int32)
        elif field == "wind_mps":
            norm = jnp.sqrt(jnp.sum(current * current))
            direction = jnp.where(
                norm > 1e-3, current / jnp.maximum(norm, 1e-3), jnp.array([1.0, 0.0])
            )
            edge = _hard_bound(c.wind_speed_mps, d) * direction
        else:
            edge = _hard_bound(getattr(c, range_name), d)
        updates[field] = jnp.where(on, edge, current)
    cond = cond._replace(**updates)
    # Keep the derived base grip consistent with a stretched rain level or dry grip.
    return cond._replace(mu_base=cond.mu_dry * (1.0 - c.rain_grip_drop * cond.rain))


def _sample(
    key: jax.Array,
    difficulty: jax.Array,
    track_length_m: jax.Array,
    track_width_m: jax.Array,
    cfg: Config,
) -> Conditions:
    c = cfg.conditions
    p = c.puddle_slots
    keys = jax.random.split(key, 20)
    d = jnp.asarray(difficulty, jnp.float32)
    rain = _uniform(keys[0], c.rain, d)
    mu_dry = _uniform(keys[1], c.mu_dry, d)
    wind_speed = _uniform(keys[2], c.wind_speed_mps, d)
    wind_dir = jax.random.uniform(keys[3], (), minval=0.0, maxval=2 * jnp.pi)
    delay_max = jnp.floor(c.delay_max_steps * d + 0.5).astype(jnp.int32)
    delay = jax.random.randint(keys[4], (), 0, delay_max + 1)
    count_jitter = jax.random.uniform(keys[5], (), minval=c.puddle_count_jitter_min, maxval=1.0)
    n_puddles = jnp.floor(rain * c.puddles_per_rain * count_jitter + 0.5)
    lat = c.puddle_lateral_frac * track_width_m
    return Conditions(
        rain=rain,
        mu_dry=mu_dry,
        mu_base=mu_dry * (1.0 - c.rain_grip_drop * rain),
        wear_per_km=_uniform(keys[6], c.tire_wear_per_km, d),
        mass_mult=_uniform(keys[7], c.mass_mult, d),
        drag_area_m2=_uniform(keys[8], c.drag_area_m2, d),
        rolling_coef=_uniform(keys[9], c.rolling_coef, d),
        power_mult=_uniform(keys[10], c.power_mult, d),
        brake_mult=_uniform(keys[11], c.brake_mult, d),
        steer_tau_s=_uniform(keys[12], c.steer_tau_s, d),
        delay_steps=delay,
        obs_noise=_uniform(keys[13], c.obs_noise, d),
        wind_mps=wind_speed * jnp.stack([jnp.cos(wind_dir), jnp.sin(wind_dir)]),
        gust_sigma_mps=_uniform(keys[14], c.gust_sigma_mps, d),
        puddle_s_m=jax.random.uniform(
            keys[15],
            (p,),
            minval=c.puddle_start_margin_m,
            maxval=track_length_m - c.puddle_end_margin_m,
        ),
        puddle_lateral_m=jax.random.uniform(keys[16], (p,), minval=-lat, maxval=lat),
        puddle_mult=jax.random.uniform(
            keys[17], (p,), minval=c.puddle_mult_min, maxval=c.puddle_mult_max
        ),
        puddle_radius_m=jax.random.uniform(
            keys[18], (p,), minval=c.puddle_radius_min_m, maxval=c.puddle_radius_max_m
        ),
        puddle_active=(jnp.arange(p) < n_puddles).astype(jnp.float32),
        puddle_xy_m=jnp.zeros((p, 2), jnp.float32),
    )


def nominal_conditions(cfg: Config = CONFIG, xp: ModuleType = np) -> Conditions:
    """Difficulty-0 conditions (dry, no puddles, nominal car, no lag/noise/wind) for one car.

    Use `._replace(...)` to build test and evaluation suites.
    """
    c = cfg.conditions
    p = c.puddle_slots
    f = xp.float32
    return Conditions(
        rain=f(0.0),
        mu_dry=f(c.mu_dry.nominal),
        mu_base=f(c.mu_dry.nominal),
        wear_per_km=f(c.tire_wear_per_km.nominal),
        mass_mult=f(c.mass_mult.nominal),
        drag_area_m2=f(c.drag_area_m2.nominal),
        rolling_coef=f(c.rolling_coef.nominal),
        power_mult=f(c.power_mult.nominal),
        brake_mult=f(c.brake_mult.nominal),
        steer_tau_s=f(c.steer_tau_s.nominal),
        delay_steps=xp.int32(0),
        obs_noise=f(0.0),
        wind_mps=xp.zeros(2, f),
        gust_sigma_mps=f(0.0),
        puddle_s_m=xp.zeros(p, f),
        puddle_lateral_m=xp.zeros(p, f),
        puddle_mult=xp.ones(p, f),
        puddle_radius_m=xp.full(p, c.puddle_radius_min_m, f),
        puddle_active=xp.zeros(p, f),
        puddle_xy_m=xp.zeros((p, 2), f),
    )


def with_rain(cond: Conditions, rain: float, cfg: Config = CONFIG) -> Conditions:
    """Set the rain level and the derived base grip consistently."""
    return cond._replace(
        rain=cond.rain * 0 + rain,
        mu_base=cond.mu_dry * (1.0 - cfg.conditions.rain_grip_drop * rain),
    )


def puddle_grip(pos_xy: Any, cond: Conditions, xp: ModuleType = np) -> Any:
    """Grip multiplier field m(p) = Π_j [1 − (1 − m_j) · exp(−d_j² / (2 (ρ_j/2)²))] at a point.

    Inactive puddle slots contribute a factor of 1.
    """
    dx = pos_xy[0] - cond.puddle_xy_m[:, 0]
    dy = pos_xy[1] - cond.puddle_xy_m[:, 1]
    sigma = cond.puddle_radius_m * 0.5
    bump = xp.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    factor = 1.0 - (1.0 - cond.puddle_mult) * bump * cond.puddle_active
    return xp.prod(factor)


def suite_conditions(
    key: jax.Array,
    suite: str,
    track_length_m: jax.Array,
    track_width_m: jax.Array,
    cfg: Config = CONFIG,
) -> Conditions:
    """Conditions for one car in a named evaluation suite (SPEC 8.2). Everything a suite
    doesn't name stays nominal. `SUITES` lists them; `OOD_SUITES` were never trained on.
    """
    c = cfg.conditions
    k_place, k_rain, k_count = jax.random.split(key, 3)
    # Draw puddle placement/shape at full range, then decide how many are active.
    full = sample_conditions(k_place, 1.0, track_length_m, track_width_m, cfg)
    base = nominal_conditions(cfg, jnp)._replace(
        puddle_s_m=full.puddle_s_m,
        puddle_lateral_m=full.puddle_lateral_m,
        puddle_mult=full.puddle_mult,
        puddle_radius_m=full.puddle_radius_m,
    )
    slots = jnp.arange(c.puddle_slots)
    if suite == "dry":
        return base
    if suite == "wet":
        rain = jax.random.uniform(k_rain, (), minval=0.6, maxval=1.0)
        jitter = jax.random.uniform(k_count, (), minval=c.puddle_count_jitter_min, maxval=1.0)
        count = jnp.floor(rain * c.puddles_per_rain * jitter + 0.5)
        return with_rain(base, rain, cfg)._replace(
            puddle_active=(slots < count).astype(jnp.float32)
        )
    if suite == "puddles":
        return with_rain(base, 0.8, cfg)._replace(
            puddle_active=jnp.ones(c.puddle_slots, jnp.float32)
        )
    f = jnp.float32
    if suite == "wind":
        angle = jax.random.uniform(k_rain, (), maxval=2 * jnp.pi)
        wind = 10.0 * jnp.stack([jnp.cos(angle), jnp.sin(angle)])
        return base._replace(wind_mps=wind, gust_sigma_mps=f(3.0))
    if suite == "worn":
        return base._replace(wear_per_km=f(0.25))
    if suite == "laggy":
        return base._replace(steer_tau_s=f(0.15), delay_steps=jnp.int32(2))
    if suite == "heavy":
        return base._replace(mass_mult=f(1.25), power_mult=f(0.75))
    if suite == "combined":
        return sample_conditions(k_place, 1.0, track_length_m, track_width_m, cfg)
    if suite == "ice":
        return base._replace(mu_dry=f(c.mu_dry.ood), mu_base=f(c.mu_dry.ood))
    if suite == "extreme_lag":
        return base._replace(
            steer_tau_s=f(c.steer_tau_s.ood), delay_steps=jnp.int32(c.delay_ood_steps)
        )
    if suite == "noisy_sensors":
        return base._replace(obs_noise=f(c.obs_noise.ood))
    if suite == "very_heavy":
        return base._replace(mass_mult=f(c.mass_mult.ood))
    raise ValueError(f"unknown suite {suite!r}")


IN_DISTRIBUTION_SUITES = ("dry", "wet", "puddles", "wind", "worn", "laggy", "heavy", "combined")
OOD_SUITES = ("ice", "extreme_lag", "noisy_sensors", "very_heavy")
SUITES = IN_DISTRIBUTION_SUITES + OOD_SUITES
