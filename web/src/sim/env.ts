/**
 * The driving environment for one car (SPEC 6.4–6.7), ported from `core_step` / `observe` in
 * `sim/trustlap/env.py`: progress tracking, observations, reward and termination.
 * State is mutated in place; outputs go into caller-owned buffers.
 */
import type { SimConfig } from './config.ts'
import { puddleGrip, type Conditions } from './conditions.ts'
import {
  axleGrip,
  controlStep,
  resetCarState,
  tireGripFactor,
  wrapAngle,
  type CarState,
  createCarState,
} from './dynamics.ts'
import { mod, roundHalfEven, type TrackData } from './track.ts'

const fr = Math.fround
const clip = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

export interface EnvState {
  car: CarState
  cond: Conditions
  idx: number // nearest centerline point
  sM: number // arclength position in [0, L)
  progressM: number // signed distance along the track since the episode started
  steps: number
  spinSteps: number
  stallSteps: number
  prevAction: Float32Array // [2]
  lastProprio: Float32Array // [8] what the policy last observed (with noise)
  history: Float32Array // [H × 10] oldest first
  historyLen: number
}

export interface StepInfo {
  deltaSM: number
  lapCompleted: boolean
  crashed: boolean
  stalled: boolean
  wrongWay: boolean
  terminated: boolean
  truncated: boolean
  speedMps: number
  muFront: number
  muRear: number
  steps: number
  progressM: number
}

export interface Observation {
  policy: Float32Array // [33]
  privileged: Float32Array // [14]
}

export function createEnvState(cfg: SimConfig, cond: Conditions): EnvState {
  return {
    car: createCarState(cfg),
    cond,
    idx: 0,
    sM: 0,
    progressM: 0,
    steps: 0,
    spinSteps: 0,
    stallSteps: 0,
    prevAction: new Float32Array(2),
    lastProprio: new Float32Array(cfg.obs.proprio_size),
    history: new Float32Array(cfg.obs.history_size),
    historyLen: 0,
  }
}

export function createStepInfo(): StepInfo {
  return {
    deltaSM: 0,
    lapCompleted: false,
    crashed: false,
    stalled: false,
    wrongWay: false,
    terminated: false,
    truncated: false,
    speedMps: 0,
    muFront: 0,
    muRear: 0,
    steps: 0,
    progressM: 0,
  }
}

export function createObservation(cfg: SimConfig): Observation {
  return {
    policy: new Float32Array(cfg.obs.policy_size),
    privileged: new Float32Array(cfg.obs.privileged_size),
  }
}

// ---------------------------------------------------------------------------------------------
// Track geometry
// ---------------------------------------------------------------------------------------------

/** Wrap an arclength into [0, L). */
export function wrapS(s: number, length: number): number {
  return s - length * Math.floor(s / length)
}

/** Centerline point and unit tangent at arclength s (linear interpolation) → out[0..3]. */
export function pointAt(track: TrackData, s: number, out: Float64Array): void {
  const u = wrapS(s, track.lengthM) / track.dsM
  const i0f = Math.floor(u)
  const f = u - i0f
  const i0 = mod(i0f, track.n)
  const i1 = (i0 + 1) % track.n
  const c = track.centerline
  const t = track.tangent
  out[0] = c[2 * i0]! + f * (c[2 * i1]! - c[2 * i0]!)
  out[1] = c[2 * i0 + 1]! + f * (c[2 * i1 + 1]! - c[2 * i0 + 1]!)
  const tx = t[2 * i0]! + f * (t[2 * i1]! - t[2 * i0]!)
  const ty = t[2 * i0 + 1]! + f * (t[2 * i1 + 1]! - t[2 * i0 + 1]!)
  const norm = Math.sqrt(tx * tx + ty * ty)
  out[2] = tx / norm
  out[3] = ty / norm
}

/** Centerline curvature at arclength s (linear interpolation). */
export function curvatureAt(track: TrackData, s: number): number {
  const u = wrapS(s, track.lengthM) / track.dsM
  const i0f = Math.floor(u)
  const f = u - i0f
  const i0 = mod(i0f, track.n)
  const k0 = track.curvature[i0]!
  return k0 + f * (track.curvature[(i0 + 1) % track.n]! - k0)
}

const scratch = new Float64Array(4)

/**
 * Nearest centerline index within [idx − 5, idx + 25] and the projected arclength (SPEC 6.5).
 * Returns [idx, s] in `out`.
 */
export function locate(
  track: TrackData,
  x: number,
  y: number,
  idx: number,
  cfg: SimConfig,
  out: Float64Array,
): void {
  const ep = cfg.episode
  const c = track.centerline
  const n = track.n
  let best = -1
  let bestD = Infinity
  for (let o = -ep.search_back; o <= ep.search_ahead; o++) {
    const cand = mod(idx + o, n)
    const dx = c[2 * cand]! - x
    const dy = c[2 * cand + 1]! - y
    const d = dx * dx + dy * dy
    if (d < bestD) {
      bestD = d
      best = cand
    }
  }
  const i = best
  const project = (a: number): number => {
    const b = (a + 1) % n
    const ex = c[2 * b]! - c[2 * a]!
    const ey = c[2 * b + 1]! - c[2 * a + 1]!
    return ((x - c[2 * a]!) * ex + (y - c[2 * a + 1]!) * ey) / (ex * ex + ey * ey)
  }
  const tNext = project(i)
  const iPrev = (i - 1 + n) % n
  const tPrev = project(iPrev)
  const usePrev = tNext < 0.0
  const seg = usePrev ? iPrev : i
  const t = clip(usePrev ? tPrev : tNext, 0.0, 1.0)
  out[0] = i
  out[1] = wrapS((seg + t) * track.dsM, track.lengthM)
}

/** Signed lateral offset e_y (m, + = left) and heading error e_ψ (rad) → out[0..1]. */
export function lateralAndHeadingError(
  track: TrackData,
  s: number,
  x: number,
  y: number,
  yaw: number,
  out: Float64Array,
): void {
  pointAt(track, s, scratch)
  const tx = scratch[2]!
  const ty = scratch[3]!
  out[0] = -(x - scratch[0]!) * ty + (y - scratch[1]!) * tx
  out[1] = wrapAngle(yaw - Math.atan2(ty, tx))
}

// ---------------------------------------------------------------------------------------------
// Observations
// ---------------------------------------------------------------------------------------------

const err = new Float64Array(2)
const pt = new Float64Array(4)
const grips = new Float64Array(2)

/** [8] proprioception (no noise) → out[offset..offset+8]. */
export function proprioception(
  state: EnvState,
  track: TrackData,
  cfg: SimConfig,
  out: Float32Array,
  offset = 0,
): void {
  const o = cfg.obs
  const car = state.car
  lateralAndHeadingError(track, state.sM, car.xM, car.yM, car.yawRad, err)
  out[offset] = car.vxMps / o.vx_scale_mps
  out[offset + 1] = car.vyMps / o.vy_scale_mps
  out[offset + 2] = car.yawRateRadps / o.yaw_rate_scale_radps
  out[offset + 3] = car.steerRad / cfg.car.steer_max_rad
  out[offset + 4] = state.prevAction[0]!
  out[offset + 5] = state.prevAction[1]!
  out[offset + 6] = err[0]! / (track.widthM / 2.0)
  out[offset + 7] = err[1]! / Math.PI
}

/** [24] centerline points ahead in the car frame / 100, as (x, y) pairs → out[offset..]. */
export function lookahead(
  state: EnvState,
  track: TrackData,
  cfg: SimConfig,
  out: Float32Array,
  offset = 0,
): void {
  const car = state.car
  const c = Math.cos(car.yawRad)
  const s = Math.sin(car.yawRad)
  const scale = cfg.obs.lookahead_scale_m
  cfg.obs.lookahead_m.forEach((d, k) => {
    pointAt(track, state.sM + fr(d), pt)
    const dx = pt[0]! - car.xM
    const dy = pt[1]! - car.yM
    out[offset + 2 * k] = (c * dx + s * dy) / scale
    out[offset + 2 * k + 1] = (-s * dx + c * dy) / scale
  })
}

/** Lowest puddle grip at 0/10/20/30 m ahead on the car's current lateral line. */
export function puddleMinAhead(state: EnvState, track: TrackData, cfg: SimConfig): number {
  const car = state.car
  lateralAndHeadingError(track, state.sM, car.xM, car.yM, car.yawRad, err)
  const eY = err[0]!
  let lowest = Infinity
  for (const d of cfg.obs.puddle_ahead_m) {
    pointAt(track, state.sM + d, pt)
    const px = pt[0]! - eY * pt[3]!
    const py = pt[1]! + eY * pt[2]!
    lowest = Math.min(lowest, puddleGrip(px, py, state.cond))
  }
  return lowest
}

/** [14] true conditions, normalized over the training ranges (teacher/critic) → out. */
export function privileged(
  state: EnvState,
  track: TrackData,
  cfg: SimConfig,
  out: Float32Array,
): void {
  const car = state.car
  const cond = state.cond
  axleGrip(car, cond, cfg, grips)
  const windX = cond.windMps[0]! + car.gustMps[0]!
  const windY = cond.windMps[1]! + car.gustMps[1]!
  const c = Math.cos(car.yawRad)
  const s = Math.sin(car.yawRad)
  const raw = [
    grips[0]!,
    grips[1]!,
    cond.muBase,
    tireGripFactor(car.odometerM, cond, cfg),
    puddleMinAhead(state, track, cfg),
    cond.massMult,
    cond.dragAreaM2,
    cond.rollingCoef,
    cond.powerMult,
    cond.brakeMult,
    cond.steerTauS,
    cond.delaySteps,
    c * windX + s * windY,
    -s * windX + c * windY,
  ]
  const center = cfg.obs.privileged_center
  const half = cfg.obs.privileged_half_range
  for (let k = 0; k < raw.length; k++) out[k] = (raw[k]! - fr(center[k]!)) / fr(half[k]!)
}

/**
 * All observations for one car. `noise` (32 standard-normal values, or null for none) is
 * scaled by the episode's sensor noise and added to proprioception + lookahead. The noisy
 * proprioception is written to `state.lastProprio` (it feeds the history buffer).
 */
export function observe(
  state: EnvState,
  track: TrackData,
  noise: Float32Array | null,
  cfg: SimConfig,
  out: Observation,
): void {
  const policy = out.policy
  proprioception(state, track, cfg, policy, 0)
  lookahead(state, track, cfg, policy, cfg.obs.proprio_size)
  const sensed = policy.length - 1
  if (noise)
    for (let k = 0; k < sensed; k++) policy[k] = policy[k]! + state.cond.obsNoise * noise[k]!
  policy[sensed] = track.widthM / cfg.obs.width_scale_m
  state.lastProprio.set(policy.subarray(0, cfg.obs.proprio_size))
  privileged(state, track, cfg, out.privileged)
}

// ---------------------------------------------------------------------------------------------
// Reset and step
// ---------------------------------------------------------------------------------------------

/** Convert puddle track coordinates (arclength from `startS`, lateral offset) to world xy. */
export function placePuddles(cond: Conditions, track: TrackData, startS: number): void {
  for (let j = 0; j < cond.puddleActive.length; j++) {
    pointAt(track, startS + cond.puddleSM[j]!, pt)
    const lat = cond.puddleLateralM[j]!
    cond.puddleXYM[2 * j] = pt[0]! + lat * -pt[3]!
    cond.puddleXYM[2 * j + 1] = pt[1]! + lat * pt[2]!
  }
}

/**
 * Start an episode: car on the centerline at `startS`, aligned with the track. Unlike the
 * Python `start_state`, puddles are not re-placed: in the browser they belong to the track.
 */
export function resetEnvState(
  state: EnvState,
  track: TrackData,
  startS: number,
  speedMps: number,
): void {
  const s0 = wrapS(fr(startS), track.lengthM)
  pointAt(track, s0, pt)
  resetCarState(state.car, pt[0]!, pt[1]!, Math.atan2(pt[3]!, pt[2]!), speedMps)
  state.idx = Math.floor(s0 / track.dsM)
  state.sM = fr(s0)
  state.progressM = 0
  state.steps = 0
  state.spinSteps = 0
  state.stallSteps = 0
  state.prevAction.fill(0)
  state.lastProprio.fill(0)
  state.history.fill(0)
  state.historyLen = 0
}

const loc = new Float64Array(2)

/**
 * Advance one car one control step: physics, progress, reward, termination (no reset).
 * Writes `info` and returns the reward.
 */
export function coreStep(
  state: EnvState,
  aSteerIn: number,
  aLongIn: number,
  gustNoiseX: number,
  gustNoiseY: number,
  track: TrackData,
  cfg: SimConfig,
  info: StepInfo,
): number {
  const ep = cfg.episode
  const rw = cfg.reward
  const aSteer = fr(clip(aSteerIn, -1.0, 1.0))
  const aLong = fr(clip(aLongIn, -1.0, 1.0))
  const length = track.lengthM
  const width = track.widthM
  const car = state.car

  controlStep(car, state.cond, aSteer, aLong, gustNoiseX, gustNoiseY, cfg)
  locate(track, car.xM, car.yM, state.idx, cfg, loc)
  const idx = loc[0]!
  const s = fr(loc[1]!)
  const rawDs = s - state.sM
  const deltaS = clip(
    rawDs - length * roundHalfEven(rawDs / length),
    -ep.delta_s_clip_m,
    ep.delta_s_clip_m,
  )
  const progress = fr(state.progressM + deltaS)
  lateralAndHeadingError(track, s, car.xM, car.yM, car.yawRad, err)
  const eY = err[0]!
  const ePsi = err[1]!
  const speed = Math.sqrt(car.vxMps * car.vxMps + car.vyMps * car.vyMps)

  const dt = cfg.time.control_dt_s
  const spinSteps = Math.abs(ePsi) > ep.spin_angle_rad ? state.spinSteps + 1 : 0
  const stallSteps = speed < ep.stall_speed_mps ? state.stallSteps + 1 : 0
  const steps = state.steps + 1

  const offTrack = Math.abs(eY) > width / 2.0 + ep.offtrack_margin_m
  const spun = spinSteps > Math.round(ep.spin_time_s / dt)
  const crashed = offTrack || spun
  const stalled = stallSteps > Math.round(ep.stall_time_s / dt)
  const wrongWay = progress < -ep.wrong_way_m
  const lap = progress >= length
  const terminated = crashed || stalled || wrongWay || lap
  const truncated = steps >= ep.max_steps && !terminated

  const da0 = aSteer - state.prevAction[0]!
  const da1 = aLong - state.prevAction[1]!
  const center = eY / (width / 2.0)
  const reward =
    rw.progress_per_m * deltaS -
    rw.center * center * center -
    rw.smooth * (da0 * da0 + da1 * da1) +
    (lap ? rw.lap_bonus : 0) -
    (crashed ? rw.crash_penalty : 0) -
    (stalled ? rw.stall_penalty : 0)

  // History row = [proprioception the policy saw, action it took], oldest first.
  const h = state.history
  const row = cfg.obs.history_row_size
  h.copyWithin(0, row)
  h.set(state.lastProprio, h.length - row)
  h[h.length - 2] = aSteer
  h[h.length - 1] = aLong

  state.idx = idx
  state.sM = s
  state.progressM = progress
  state.steps = steps
  state.spinSteps = spinSteps
  state.stallSteps = stallSteps
  state.prevAction[0] = aSteer
  state.prevAction[1] = aLong
  state.historyLen = Math.min(state.historyLen + 1, cfg.obs.history_len)

  axleGrip(car, state.cond, cfg, grips)
  info.deltaSM = deltaS
  info.lapCompleted = lap
  info.crashed = crashed
  info.stalled = stalled
  info.wrongWay = wrongWay
  info.terminated = terminated
  info.truncated = truncated
  info.speedMps = speed
  info.muFront = grips[0]!
  info.muRear = grips[1]!
  info.steps = steps
  info.progressM = progress
  return reward
}
