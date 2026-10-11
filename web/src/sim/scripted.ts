/**
 * Scripted baseline driver that knows the true grip, ported from `sim/trustlap/scripted.py`.
 * Pure pursuit + yaw damping for steering; a braking-aware speed plan from true μ; pedals
 * capped to the grip cornering leaves free (friction circle).
 */
import type { SimConfig } from './config.ts'
import { puddleGrip } from './conditions.ts'
import { axleGrip, tireGripFactor } from './dynamics.ts'
import { curvatureAt, pointAt, type EnvState } from './env.ts'
import type { TrackData } from './track.ts'

const clip = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)
const pt = new Float64Array(4)
const grip = new Float64Array(2)

/**
 * Action (steer, longitudinal) for one car → out[0..1]. `scriptedCfg` defaults to the config's
 * tuned values; the browser may pass a variant (e.g. a more cautious driver for variety).
 */
export function scriptedAction(
  state: EnvState,
  track: TrackData,
  cfg: SimConfig,
  out: Float32Array,
  sc: SimConfig['scripted'] = cfg.scripted,
): void {
  const k = cfg.car
  const car = state.car
  const cond = state.cond
  const g = k.gravity_mps2
  const speed = Math.sqrt(car.vxMps * car.vxMps + car.vyMps * car.vyMps)

  // Steering: pure pursuit + yaw damping.
  const look = clip(speed * sc.lookahead_time_s, sc.lookahead_min_m, sc.lookahead_max_m)
  pointAt(track, state.sM + look, pt)
  const dx = pt[0]! - car.xM
  const dy = pt[1]! - car.yM
  const c = Math.cos(car.yawRad)
  const s = Math.sin(car.yawRad)
  const localX = c * dx + s * dy
  const localY = -s * dx + c * dy
  const alpha = Math.atan2(localY, localX)
  const pathCurvature = (2.0 * Math.sin(alpha)) / look
  let steer = Math.atan(k.wheelbase_m * pathCurvature)
  steer = steer + sc.yaw_damping_s * (speed * pathCurvature - car.yawRateRadps)
  const aSteer = clip(steer / k.steer_max_rad, -1.0, 1.0)

  // Speed plan from true grip along the road ahead.
  const nSamples = Math.floor(sc.profile_horizon_m / sc.profile_step_m) + 1
  const tire = tireGripFactor(car.odometerM, cond, cfg)
  let vTarget = Infinity
  for (let i = 0; i < nSamples; i++) {
    const dist = i * sc.profile_step_m
    const sAhead = state.sM + dist
    const kappa = Math.abs(curvatureAt(track, sAhead))
    pointAt(track, sAhead, pt)
    const mu = cond.muBase * tire * puddleGrip(pt[0]!, pt[1]!, cond)
    const vCorner = Math.sqrt((sc.grip_use * mu * g) / Math.max(kappa, 1e-4))
    const vAllow = Math.sqrt(vCorner * vCorner + 2.0 * sc.brake_use * mu * g * dist)
    vTarget = Math.min(vTarget, vAllow)
  }
  vTarget = Math.min(vTarget, sc.max_speed_mps)

  // Pedals: requested acceleration through the force limits, capped by free grip.
  axleGrip(car, cond, cfg, grip)
  const muF = grip[0]!
  const muR = grip[1]!
  const mass = k.mass_kg * cond.massMult
  const fzF = (mass * g * k.lr_m) / k.wheelbase_m
  const fzR = (mass * g * k.lf_m) / k.wheelbase_m
  const kappaHere = Math.abs(curvatureAt(track, state.sM))
  const lateralAccel = Math.max(Math.abs(car.vxMps * car.yawRateRadps), speed * speed * kappaHere)
  const latUseF = Math.min(lateralAccel / (muF * g), 1.0)
  const latUseR = Math.min(lateralAccel / (muR * g), 1.0)
  const freeF = sc.traction_use * Math.sqrt(1.0 - latUseF * latUseF)
  const freeR = sc.traction_use * Math.sqrt(1.0 - latUseR * latUseR)
  const driveMax = Math.min(
    k.drive_force_max_n,
    (k.power_w * cond.powerMult) / Math.max(car.vxMps, 1.0),
  )
  const brakeForce = k.brake_force_max_g * k.mass_kg * g * cond.brakeMult
  const accel = sc.speed_gain * (vTarget - car.vxMps)
  let aLong = accel >= 0.0 ? (accel * mass) / driveMax : (accel * mass) / brakeForce
  const throttleCap = (freeR * muR * fzR) / driveMax
  const brakeCap = Math.min(
    (freeF * muF * fzF) / (brakeForce * k.brake_front_frac),
    (freeR * muR * fzR) / (brakeForce * (1.0 - k.brake_front_frac)),
  )
  aLong = clip(aLong, -Math.min(brakeCap, 1.0), Math.min(throttleCap, 1.0))
  out[0] = aSteer
  out[1] = aLong
}
