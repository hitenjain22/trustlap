"""Every constant in TrustLap, in one place (SPEC Section 2, rule 1).

Physics constants, condition ranges, observation layout, reward weights and training
hyperparameters live here as frozen dataclasses. `export.py` writes `to_dict()` to
`web/public/artifacts/<version>/config.json` and the TypeScript code reads it from there,
so nothing is ever hand-copied between languages.

Units are SI throughout: meters, seconds, kilograms, newtons, radians.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TimeConfig:
    """Simulation clock."""

    control_dt_s: float = 0.05  # one policy decision every 50 ms (20 Hz)
    substeps: int = 4  # physics substeps per control step

    @property
    def physics_dt_s(self) -> float:
        """Physics integration step in seconds (0.0125 s)."""
        return self.control_dt_s / self.substeps


@dataclass(frozen=True)
class TrackConfig:
    """Procedural track generation and validation (SPEC 6.1)."""

    ds_target_m: float = 2.0  # resampled centerline spacing (exact spacing is length / n_points)
    n_max: int = 2048  # padded points per track in the bank
    n_ctrl_min: int = 8  # control points, inclusive range
    n_ctrl_max: int = 14
    angle_jitter: float = 0.35  # fraction of the angular spacing 2π/n
    radius_min_m: float = 110.0
    radius_max_m: float = 260.0
    pull_prob: float = 0.3  # "character" pass: chance to pull a control point inward
    pull_min: float = 0.15  # pulled by U(pull_min, pull_max) · r_i
    pull_max: float = 0.4
    catmull_rom_alpha: float = 0.5  # centripetal
    samples_per_segment: int = 64
    curvature_smooth_window: int = 5  # circular moving average
    width_min_m: float = 11.0
    width_max_m: float = 14.0
    length_min_m: float = 900.0
    length_max_m: float = 2600.0
    min_radius_m: float = 18.0
    validate_stride: int = 4  # boundary intersection check uses every 4th point
    clearance_index_gap: int = 40  # points further apart than this (circularly) ...
    clearance_extra_m: float = 8.0  # ... must be at least width + this apart
    max_attempts: int = 50
    autofix_lambda: float = 0.3  # Laplacian smoothing of control points for too-tight corners
    autofix_max_iters: int = 10


@dataclass(frozen=True)
class BankConfig:
    """Seed ranges for the track banks, half-open [start, stop)."""

    train: tuple[int, int] = (0, 2000)
    validation: tuple[int, int] = (100000, 100100)
    test: tuple[int, int] = (200000, 200200)
    showcase_count: int = 24  # test-bank tracks exported to the website


@dataclass(frozen=True)
class CarConfig:
    """Nominal vehicle constants (SPEC 6.3.2). Mass-dependent values scale with mass_mult."""

    mass_kg: float = 800.0
    yaw_inertia_kgm2: float = 1100.0
    lf_m: float = 1.2  # center of gravity to front axle
    lr_m: float = 1.4  # center of gravity to rear axle
    gravity_mps2: float = 9.81
    steer_max_rad: float = 0.42
    steer_rate_max_radps: float = 3.0
    power_w: float = 220_000.0
    drive_force_max_n: float = 7000.0
    brake_force_max_g: float = 1.3  # F_brake_max = 1.3 · m · g
    brake_front_frac: float = 0.6  # 60% front / 40% rear; rear-wheel drive
    air_density_kgpm3: float = 1.225
    side_area_coef_m2: float = 2.5  # C_s · A_s
    center_of_pressure_offset_m: float = 0.2  # behind the center of gravity
    tire_b: float = 9.0  # simplified Pacejka "magic formula"
    tire_c: float = 1.6
    tire_e: float = 0.95
    min_slip_speed_mps: float = 0.5  # vx floor inside the slip-angle atan2
    brake_min_speed_mps: float = 0.05  # brakes act only above this forward speed
    low_speed_blend_start_mps: float = 1.0  # below: fully kinematic
    low_speed_blend_width_mps: float = 2.0  # above start + width: fully dynamic
    rolling_tanh_speed_mps: float = 0.5
    body_length_m: float = 4.4  # rendering only
    body_width_m: float = 1.9

    @property
    def wheelbase_m(self) -> float:
        """Distance between axles (2.6 m)."""
        return self.lf_m + self.lr_m


@dataclass(frozen=True)
class Range:
    """A condition parameter: its value at difficulty 0, its full training range, and its
    out-of-distribution test value."""

    nominal: float
    lo: float
    hi: float
    ood: float

    def at_difficulty(self, d: float) -> tuple[float, float]:
        """Sampling bounds at curriculum difficulty d ∈ [0, 1].

        Each bound moves linearly from the nominal value (d = 0) to the full range (d = 1).
        See DECISIONS.md (2026-10-11) for why this replaces "nominal ± d · half-width".
        """
        return (
            self.nominal + d * (self.lo - self.nominal),
            self.nominal + d * (self.hi - self.nominal),
        )


@dataclass(frozen=True)
class ConditionsConfig:
    """Per-episode condition ranges (SPEC 6.2)."""

    rain: Range = Range(0.0, 0.0, 1.0, 1.0)
    mu_dry: Range = Range(1.08, 1.0, 1.15, 0.25)  # OOD "ice"
    mass_mult: Range = Range(1.0, 0.85, 1.25, 1.5)
    drag_area_m2: Range = Range(0.9, 0.6, 1.4, 1.8)
    rolling_coef: Range = Range(0.015, 0.010, 0.020, 0.030)
    power_mult: Range = Range(1.0, 0.75, 1.15, 0.6)
    brake_mult: Range = Range(1.0, 0.7, 1.1, 0.5)
    steer_tau_s: Range = Range(0.05, 0.02, 0.15, 0.25)
    obs_noise: Range = Range(0.0, 0.0, 0.05, 0.15)
    wind_speed_mps: Range = Range(0.0, 0.0, 10.0, 15.0)
    gust_sigma_mps: Range = Range(0.0, 0.0, 3.0, 5.0)
    tire_wear_per_km: Range = Range(0.0, 0.0, 0.25, 0.5)
    delay_max_steps: int = 2  # training: D ∈ {0, …, round(2d)}
    delay_ood_steps: int = 4
    delay_buffer_len: int = 4  # stores the 4 previous actions; D = k executes buffer[k − 1]
    rain_grip_drop: float = 0.45  # μ_base = μ_dry · (1 − 0.45 · R)
    puddle_slots: int = 8  # P_MAX, fixed for static shapes
    puddles_per_rain: float = 8.0  # count = round(R · 8 · U(0.5, 1))
    puddle_count_jitter_min: float = 0.5
    puddle_mult_min: float = 0.35  # grip multiplier at a puddle's center
    puddle_mult_max: float = 0.6
    puddle_mult_ood: float = 0.25
    puddle_radius_min_m: float = 4.0
    puddle_radius_max_m: float = 10.0
    puddle_start_margin_m: float = 100.0  # s_j ~ U(100, L − 50), from the episode start
    puddle_end_margin_m: float = 50.0
    puddle_lateral_frac: float = 0.35  # lateral offset ~ U(−0.35 W, 0.35 W)
    tire_grip_floor: float = 0.6  # μ_tire = max(0.6, 1 − w · km)
    gust_theta_per_s: float = 0.5  # Ornstein–Uhlenbeck mean reversion


@dataclass(frozen=True)
class ObsConfig:
    """Observation layout and scaling (SPEC 6.4)."""

    lookahead_m: tuple[float, ...] = (5, 10, 15, 20, 30, 40, 55, 70, 90, 110, 135, 160)
    vx_scale_mps: float = 30.0
    vy_scale_mps: float = 10.0
    yaw_rate_scale_radps: float = 2.0
    lookahead_scale_m: float = 100.0
    width_scale_m: float = 14.0
    wind_scale_mps: float = 10.0
    puddle_ahead_m: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)  # privileged puddle probe
    history_len: int = 20  # 1 s of [proprioception, action]

    @property
    def proprio_size(self) -> int:
        return 8

    @property
    def policy_size(self) -> int:
        """33 = proprioception 8 + lookahead 24 + width 1."""
        return self.proprio_size + 2 * len(self.lookahead_m) + 1

    @property
    def privileged_size(self) -> int:
        return 14

    @property
    def history_row_size(self) -> int:
        return self.proprio_size + 2

    @property
    def history_size(self) -> int:
        return self.history_len * self.history_row_size


@dataclass(frozen=True)
class RewardConfig:
    """Per-step reward weights (SPEC 6.6)."""

    progress_per_m: float = 0.1
    center: float = 0.02
    smooth: float = 0.01
    lap_bonus: float = 10.0
    crash_penalty: float = 10.0
    stall_penalty: float = 2.0


@dataclass(frozen=True)
class EpisodeConfig:
    """Termination, truncation, progress tracking and resets (SPEC 6.5, 6.7)."""

    max_steps: int = 2400  # 120 s, truncation (bootstrap)
    offtrack_margin_m: float = 1.0  # crash when |e_y| > W/2 + margin
    spin_angle_rad: float = math.radians(100.0)
    spin_time_s: float = 1.0
    stall_speed_mps: float = 1.0
    stall_time_s: float = 3.0
    wrong_way_m: float = 15.0  # cumulative progress below −15 m terminates
    search_back: int = 5  # nearest-point search window [idx − 5, idx + 25]
    search_ahead: int = 25
    delta_s_clip_m: float = 5.0
    train_speed_min_mps: float = 5.0
    train_speed_max_mps: float = 20.0
    eval_speed_mps: float = 5.0


@dataclass(frozen=True)
class ScriptedConfig:
    """Pure-pursuit + speed-profile baseline driver (SPEC M1.6).

    Tuned on the validation bank (never the test bank); see PLAN.md M1 notes for the sweep.
    """

    lookahead_min_m: float = 8.0
    lookahead_max_m: float = 40.0
    lookahead_time_s: float = 1.0  # pure-pursuit lookahead = clip(speed · T, min, max)
    yaw_damping_s: float = 0.1  # steer += k · (speed · path curvature − yaw rate)
    grip_use: float = 0.85  # fraction of the available friction the corner-speed plan uses
    brake_use: float = 0.45  # fraction of μ·g the braking plan assumes
    profile_horizon_m: float = 160.0
    profile_step_m: float = 4.0
    speed_gain: float = 3.0  # requested acceleration (m/s²) per m/s of speed error, 1/s
    traction_use: float = 0.8  # throttle/brakes use at most 80% of the grip cornering leaves free
    max_speed_mps: float = 75.0


@dataclass(frozen=True)
class NetworkConfig:
    """Network shapes (SPEC 7.1)."""

    hidden: tuple[int, ...] = (128, 128, 128)
    latent_size: int = 8
    encoder_hidden: tuple[int, ...] = (64,)
    adaptation_hidden: tuple[int, ...] = (256, 128)
    grip_scale: float = 1.3
    log_std_init: float = -0.5


@dataclass(frozen=True)
class PPOConfig:
    """PureJaxRL-style PPO defaults (SPEC 7.2)."""

    envs_per_agent: int = 1024
    rollout_len: int = 64
    minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class PBTConfig:
    """Population-based training (SPEC 7.3)."""

    population: int = 16
    ready_interval_updates: int = 25
    truncation_frac: float = 0.25
    perturb_factors: tuple[float, float] = (0.8, 1.25)
    resample_prob: float = 0.25
    lr_range: tuple[float, float] = (1e-4, 1e-3)  # log-uniform prior
    ent_range: tuple[float, float] = (1e-4, 1e-2)


@dataclass(frozen=True)
class CurriculumConfig:
    """Automatic difficulty (SPEC 7.4)."""

    check_interval_updates: int = 10
    up_threshold: float = 0.8
    down_threshold: float = 0.5
    step: float = 0.05
    consecutive_up: int = 2
    easy_frac: float = 0.2
    boundary_frac: float = 0.1


@dataclass(frozen=True)
class StudentConfig:
    """Rapid motor adaptation student (SPEC 7.6)."""

    lr: float = 3e-4
    grad_steps_per_rollout: int = 4
    rollout_len: int = 32
    grip_loss_weight: float = 0.5
    warm_start_frac: float = 0.1


@dataclass(frozen=True)
class EnsembleConfig:
    """Student ensemble and confidence (SPEC 7.7)."""

    members: int = 5
    crash_horizon_s: float = 2.0
    governor_taus: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


@dataclass(frozen=True)
class WebConfig:
    """Browser defaults (Appendix A)."""

    hero_cars: int = 48
    playground_cars_max: int = 5
    lab_cars: int = 16
    mobile_cars_max: int = 16


@dataclass(frozen=True)
class Config:
    """The whole configuration tree."""

    time: TimeConfig = field(default_factory=TimeConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    banks: BankConfig = field(default_factory=BankConfig)
    car: CarConfig = field(default_factory=CarConfig)
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    scripted: ScriptedConfig = field(default_factory=ScriptedConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    pbt: PBTConfig = field(default_factory=PBTConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    web: WebConfig = field(default_factory=WebConfig)


CONFIG = Config()


# Observation layout: the exact order of every element, exported so TypeScript can assert it.
def _obs_layout(cfg: Config) -> dict[str, list[str]]:
    lookahead = [f"look_{int(d)}m_{axis}" for d in cfg.obs.lookahead_m for axis in ("x", "y")]
    return {
        "policy": [
            "vx",
            "vy",
            "yaw_rate",
            "steer",
            "prev_a_steer",
            "prev_a_long",
            "e_y",
            "e_psi",
            *lookahead,
            "width",
        ],
        "privileged": [
            "mu_front",
            "mu_rear",
            "mu_base",
            "mu_tire",
            "puddle_min_ahead",
            "mass_mult",
            "drag_area",
            "rolling_coef",
            "power_mult",
            "brake_mult",
            "steer_tau",
            "delay",
            "wind_body_x",
            "wind_body_y",
        ],
        "history_row": [
            "vx",
            "vy",
            "yaw_rate",
            "steer",
            "prev_a_steer",
            "prev_a_long",
            "e_y",
            "e_psi",
            "a_steer",
            "a_long",
        ],
    }


OBS_LAYOUT = _obs_layout(CONFIG)


def privileged_scaling(cfg: Config = CONFIG) -> tuple[list[float], list[float]]:
    """(center, half_range) per privileged element; normalized = (value − center) / half_range.

    Condition-type elements use their full training range, so training values land in [−1, 1]
    and out-of-distribution values fall outside it.
    """
    c = cfg.conditions
    mu_lo = c.puddle_mult_min * c.mu_dry.lo * (1 - c.rain_grip_drop) * c.tire_grip_floor
    mu_hi = c.mu_dry.hi

    def span(lo: float, hi: float) -> tuple[float, float]:
        return (lo + hi) / 2, (hi - lo) / 2

    pairs = [
        span(mu_lo, mu_hi),  # mu_front
        span(mu_lo, mu_hi),  # mu_rear
        span(mu_lo, mu_hi),  # mu_base
        span(c.tire_grip_floor, 1.0),  # mu_tire
        span(c.puddle_mult_min, 1.0),  # puddle_min_ahead
        span(c.mass_mult.lo, c.mass_mult.hi),
        span(c.drag_area_m2.lo, c.drag_area_m2.hi),
        span(c.rolling_coef.lo, c.rolling_coef.hi),
        span(c.power_mult.lo, c.power_mult.hi),
        span(c.brake_mult.lo, c.brake_mult.hi),
        span(c.steer_tau_s.lo, c.steer_tau_s.hi),
        (0.0, float(c.delay_buffer_len)),  # delay: D / 4
        (0.0, cfg.obs.wind_scale_mps),  # wind_body_x
        (0.0, cfg.obs.wind_scale_mps),  # wind_body_y
    ]
    assert len(pairs) == cfg.obs.privileged_size
    return [p[0] for p in pairs], [p[1] for p in pairs]


def to_dict(cfg: Config = CONFIG) -> dict[str, Any]:
    """Plain-JSON view of the config, including derived values the web needs."""
    out: dict[str, Any] = dataclasses.asdict(cfg)
    out["time"]["physics_dt_s"] = cfg.time.physics_dt_s
    out["car"]["wheelbase_m"] = cfg.car.wheelbase_m
    out["obs"].update(
        policy_size=cfg.obs.policy_size,
        privileged_size=cfg.obs.privileged_size,
        history_size=cfg.obs.history_size,
        history_row_size=cfg.obs.history_row_size,
    )
    center, half = privileged_scaling(cfg)
    out["obs"]["privileged_center"] = center
    out["obs"]["privileged_half_range"] = half
    out["obs_layout"] = OBS_LAYOUT
    return out


def config_hash(cfg: Config = CONFIG) -> str:
    """Stable sha256 of the config; checkpoints refuse to resume if it changes."""
    blob = json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
