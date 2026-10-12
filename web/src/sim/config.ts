/**
 * Schema for `config.json`, the constants exported from `sim/trustlap/config.py` (SPEC rule 1).
 * Nothing here is a hand-copied number: every value comes from the file, validated at load.
 * Only the sections the browser uses are typed; the rest pass through untouched.
 */
import { z } from 'zod'

// Every schema here parses a handful of small files once; zod's JIT (it compiles a fast path
// per object schema on first parse) costs more than it saves and lands in the load-time long
// tasks. Global setting, so it covers every schema in the app and the worker.
z.config({ jitless: true })

const num = z.number()
const int = z.number().int()
const range = z.object({ nominal: num, lo: num, hi: num, ood: num })

export const simConfigSchema = z.object({
  time: z.object({ control_dt_s: num, substeps: int, physics_dt_s: num }),
  track: z.object({
    ds_target_m: num,
    n_max: int,
    n_ctrl_min: int,
    n_ctrl_max: int,
    angle_jitter: num,
    radius_min_m: num,
    radius_max_m: num,
    pull_prob: num,
    pull_min: num,
    pull_max: num,
    catmull_rom_alpha: num,
    samples_per_segment: int,
    curvature_smooth_window: int,
    width_min_m: num,
    width_max_m: num,
    length_min_m: num,
    length_max_m: num,
    min_radius_m: num,
    validate_stride: int,
    clearance_index_gap: int,
    clearance_extra_m: num,
    max_attempts: int,
    autofix_lambda: num,
    autofix_max_iters: int,
  }),
  car: z.object({
    mass_kg: num,
    yaw_inertia_kgm2: num,
    lf_m: num,
    lr_m: num,
    gravity_mps2: num,
    steer_max_rad: num,
    steer_rate_max_radps: num,
    power_w: num,
    drive_force_max_n: num,
    brake_force_max_g: num,
    brake_front_frac: num,
    air_density_kgpm3: num,
    side_area_coef_m2: num,
    center_of_pressure_offset_m: num,
    tire_b: num,
    tire_c: num,
    tire_e: num,
    min_slip_speed_mps: num,
    brake_min_speed_mps: num,
    low_speed_blend_start_mps: num,
    low_speed_blend_width_mps: num,
    rolling_tanh_speed_mps: num,
    body_length_m: num,
    body_width_m: num,
    wheelbase_m: num,
  }),
  conditions: z.object({
    rain: range,
    mu_dry: range,
    mass_mult: range,
    drag_area_m2: range,
    rolling_coef: range,
    power_mult: range,
    brake_mult: range,
    steer_tau_s: range,
    obs_noise: range,
    wind_speed_mps: range,
    gust_sigma_mps: range,
    tire_wear_per_km: range,
    delay_max_steps: int,
    delay_ood_steps: int,
    delay_buffer_len: int,
    rain_grip_drop: num,
    puddle_slots: int,
    puddles_per_rain: num,
    puddle_count_jitter_min: num,
    puddle_mult_min: num,
    puddle_mult_max: num,
    puddle_mult_ood: num,
    puddle_radius_min_m: num,
    puddle_radius_max_m: num,
    puddle_start_margin_m: num,
    puddle_end_margin_m: num,
    puddle_lateral_frac: num,
    tire_grip_floor: num,
    gust_theta_per_s: num,
  }),
  obs: z.object({
    lookahead_m: z.array(num),
    vx_scale_mps: num,
    vy_scale_mps: num,
    yaw_rate_scale_radps: num,
    lookahead_scale_m: num,
    width_scale_m: num,
    wind_scale_mps: num,
    puddle_ahead_m: z.array(num),
    history_len: int,
    proprio_size: int,
    policy_size: int,
    privileged_size: int,
    history_size: int,
    history_row_size: int,
    privileged_center: z.array(num),
    privileged_half_range: z.array(num),
  }),
  reward: z.object({
    progress_per_m: num,
    center: num,
    smooth: num,
    lap_bonus: num,
    crash_penalty: num,
    stall_penalty: num,
  }),
  episode: z.object({
    max_steps: int,
    offtrack_margin_m: num,
    spin_angle_rad: num,
    spin_time_s: num,
    stall_speed_mps: num,
    stall_time_s: num,
    wrong_way_m: num,
    search_back: int,
    search_ahead: int,
    delta_s_clip_m: num,
    train_speed_min_mps: num,
    train_speed_max_mps: num,
    eval_speed_mps: num,
  }),
  scripted: z.object({
    lookahead_min_m: num,
    lookahead_max_m: num,
    lookahead_time_s: num,
    yaw_damping_s: num,
    grip_use: num,
    brake_use: num,
    profile_horizon_m: num,
    profile_step_m: num,
    speed_gain: num,
    traction_use: num,
    max_speed_mps: num,
  }),
  web: z.object({ hero_cars: int, playground_cars_max: int, lab_cars: int, mobile_cars_max: int }),
  // Risk-feature constants arrived with the logistic risk model; older exports don't have them.
  ensemble: z
    .object({ risk_lookahead_points: int.optional(), risk_min_grip: num.optional() })
    .optional(),
  obs_layout: z.object({
    policy: z.array(z.string()),
    privileged: z.array(z.string()),
    history_row: z.array(z.string()),
  }),
})

export type SimConfig = z.infer<typeof simConfigSchema>
export type ConditionRange = z.infer<typeof range>

/** Parse and sanity-check a config; throws with the zod error on mismatch (fail loudly). */
export function parseSimConfig(raw: unknown): SimConfig {
  const cfg = simConfigSchema.parse(raw)
  if (cfg.obs.policy_size !== cfg.obs_layout.policy.length) {
    throw new Error(`obs layout mismatch: policy_size ${cfg.obs.policy_size}`)
  }
  if (cfg.obs.privileged_center.length !== cfg.obs.privileged_size) {
    throw new Error('privileged scaling has the wrong length')
  }
  return cfg
}
