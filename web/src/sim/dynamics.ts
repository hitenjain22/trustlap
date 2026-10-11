/**
 * Vehicle dynamics (SPEC 6.3), ported operation by operation from `sim/trustlap/dynamics.py`.
 *
 * Python runs in float32; here intermediate math is float64 and every state variable is
 * rounded to float32 (`Math.fround`) after each substep, which keeps the two within the
 * golden-file tolerance. Functions mutate a preallocated `CarState` (no per-step allocation).
 *
 * Units: meters, seconds, kilograms, newtons, radians. Body frame: x forward, y left.
 */
import type { SimConfig } from './config.ts'
import { puddleGrip, type Conditions } from './conditions.ts'

const fr = Math.fround

export interface CarState {
  xM: number
  yM: number
  yawRad: number // wrapped to (−π, π]
  vxMps: number // ≥ 0, no reverse gear
  vyMps: number
  yawRateRadps: number
  steerRad: number
  actionBuffer: Float32Array // [4 × 2] previous actions, newest first
  gustMps: Float32Array // [2] world frame
  odometerM: number
}

export function createCarState(cfg: SimConfig): CarState {
  return {
    xM: 0,
    yM: 0,
    yawRad: 0,
    vxMps: 0,
    vyMps: 0,
    yawRateRadps: 0,
    steerRad: 0,
    actionBuffer: new Float32Array(2 * cfg.conditions.delay_buffer_len),
    gustMps: new Float32Array(2),
    odometerM: 0,
  }
}

/** Put a car at rest on a pose, rolling forward at `speedMps` with straight wheels. */
export function resetCarState(
  car: CarState,
  xM: number,
  yM: number,
  yawRad: number,
  speedMps: number,
): void {
  car.xM = fr(xM)
  car.yM = fr(yM)
  car.yawRad = fr(yawRad)
  car.vxMps = fr(speedMps)
  car.vyMps = 0
  car.yawRateRadps = 0
  car.steerRad = 0
  car.actionBuffer.fill(0)
  car.gustMps.fill(0)
  car.odometerM = 0
}

/** Wrap an angle to (−π, π]. */
export function wrapAngle(a: number): number {
  const twoPi = 2.0 * Math.PI
  return a - twoPi * Math.ceil((a - Math.PI) / twoPi)
}

/** μ_tire = max(0.6, 1 − w · km driven). */
export function tireGripFactor(odometerM: number, cond: Conditions, cfg: SimConfig): number {
  return Math.max(cfg.conditions.tire_grip_floor, 1.0 - cond.wearPerKm * (odometerM / 1000.0))
}

/** Friction coefficient under the front and rear axle: μ_base · μ_tire · puddle field. */
export function axleGrip(car: CarState, cond: Conditions, cfg: SimConfig, out: Float64Array): void {
  const c = Math.cos(car.yawRad)
  const s = Math.sin(car.yawRad)
  const base = cond.muBase * tireGripFactor(car.odometerM, cond, cfg)
  out[0] = base * puddleGrip(car.xM + cfg.car.lf_m * c, car.yM + cfg.car.lf_m * s, cond)
  out[1] = base * puddleGrip(car.xM - cfg.car.lr_m * c, car.yM - cfg.car.lr_m * s, cond)
}

/** Simplified Pacejka lateral force: −μ·Fz·sin(C·atan(Bα − E(Bα − atan(Bα)))). */
export function lateralTireForce(
  alphaRad: number,
  mu: number,
  fzN: number,
  cfg: SimConfig,
): number {
  const { tire_b: b, tire_c: c, tire_e: e } = cfg.car
  const ba = b * alphaRad
  return -mu * fzN * Math.sin(c * Math.atan(ba - e * (ba - Math.atan(ba))))
}

const clip = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)
const grip = new Float64Array(2)

/** Advance one car by one physics substep (12.5 ms). Actions arrive delayed and clipped. */
export function substep(
  car: CarState,
  cond: Conditions,
  aSteer: number,
  aLong: number,
  cfg: SimConfig,
): void {
  const k = cfg.car
  const dt = cfg.time.physics_dt_s
  const g = k.gravity_mps2
  const wheelbase = k.wheelbase_m
  const mass = k.mass_kg * cond.massMult
  const iz = k.yaw_inertia_kgm2 * cond.massMult
  let vx = car.vxMps
  let vy = car.vyMps
  let r = car.yawRateRadps
  let delta = car.steerRad
  const yaw = car.yawRad

  // 1–2. Steering actuator.
  const steerCmd = aSteer * k.steer_max_rad
  const steerRate = clip(
    (steerCmd - delta) / cond.steerTauS,
    -k.steer_rate_max_radps,
    k.steer_rate_max_radps,
  )

  // 3. Static loads.
  const fzF = (mass * g * k.lr_m) / wheelbase
  const fzR = (mass * g * k.lf_m) / wheelbase
  axleGrip(car, cond, cfg, grip)
  const muF = grip[0]!
  const muR = grip[1]!

  // 4. Longitudinal forces (drive/brake hardware fixed at nominal mass).
  const throttle = Math.max(aLong, 0.0)
  const brake = Math.max(-aLong, 0.0) * (vx > k.brake_min_speed_mps ? 1 : 0)
  const fDrive =
    throttle * Math.min(k.drive_force_max_n, (k.power_w * cond.powerMult) / Math.max(vx, 1.0))
  const fBrake = brake * (k.brake_force_max_g * k.mass_kg * g) * cond.brakeMult
  const fxFMax = muF * fzF
  const fxRMax = muR * fzR
  const fxF = clip(-fBrake * k.brake_front_frac, -fxFMax, fxFMax)
  const fxR = clip(fDrive - fBrake * (1.0 - k.brake_front_frac), -fxRMax, fxRMax)

  // 5–6. Slip angles and lateral forces with the friction ellipse.
  const vxSlip = Math.max(vx, k.min_slip_speed_mps)
  const alphaF = Math.atan2(vy + k.lf_m * r, vxSlip) - delta
  const alphaR = Math.atan2(vy - k.lr_m * r, vxSlip)
  const useF = fxF / fxFMax
  const useR = fxR / fxRMax
  const fyF = lateralTireForce(alphaF, muF, fzF, cfg) * Math.sqrt(Math.max(0.0, 1.0 - useF * useF))
  const fyR = lateralTireForce(alphaR, muR, fzR, cfg) * Math.sqrt(Math.max(0.0, 1.0 - useR * useR))

  // 7. Aerodynamics in the body frame.
  const cosY = Math.cos(yaw)
  const sinY = Math.sin(yaw)
  const windX = cond.windMps[0]! + car.gustMps[0]!
  const windY = cond.windMps[1]! + car.gustMps[1]!
  const windBx = cosY * windX + sinY * windY
  const windBy = -sinY * windX + cosY * windY
  const u = vx - windBx
  const v = vy - windBy
  const fDrag = -0.5 * k.air_density_kgpm3 * cond.dragAreaM2 * u * Math.abs(u)
  const fSide = -0.5 * k.air_density_kgpm3 * k.side_area_coef_m2 * v * Math.abs(v)

  // 8. Rolling resistance.
  const fRoll = -cond.rollingCoef * mass * g * Math.tanh(vx / k.rolling_tanh_speed_mps)

  // 9. Body-frame equations of motion.
  const cosD = Math.cos(delta)
  const sinD = Math.sin(delta)
  const vxDot = (fxR + fxF * cosD - fyF * sinD + fDrag + fRoll) / mass + vy * r
  const vyDot = (fyR + fyF * cosD + fxF * sinD + fSide) / mass - vx * r
  const rDot =
    (k.lf_m * (fyF * cosD + fxF * sinD) - k.lr_m * fyR - k.center_of_pressure_offset_m * fSide) / iz

  // 10. Semi-implicit Euler: velocities first …
  vx = Math.max(vx + vxDot * dt, 0.0)
  vy = vy + vyDot * dt
  r = r + rDot * dt
  delta = clip(delta + steerRate * dt, -k.steer_max_rad, k.steer_max_rad)

  // Low-speed blend toward the kinematic bicycle.
  const speed = Math.sqrt(vx * vx + vy * vy)
  const lam = clip((speed - k.low_speed_blend_start_mps) / k.low_speed_blend_width_mps, 0.0, 1.0)
  const rKin = (vx * Math.tan(delta)) / wheelbase
  const vyKin = rKin * k.lr_m
  vy = lam * vy + (1.0 - lam) * vyKin
  r = lam * r + (1.0 - lam) * rKin

  // … then positions with the new velocities. Round state to float32 like the Python side.
  car.xM = fr(car.xM + (vx * cosY - vy * sinY) * dt)
  car.yM = fr(car.yM + (vx * sinY + vy * cosY) * dt)
  car.yawRad = fr(wrapAngle(yaw + r * dt))
  car.vxMps = fr(vx)
  car.vyMps = fr(vy)
  car.yawRateRadps = fr(r)
  car.steerRad = fr(delta)
  car.odometerM = fr(car.odometerM + Math.sqrt(vx * vx + vy * vy) * dt)
}

/**
 * Advance one car by one control step (50 ms = 4 substeps). `action` = (steer, long), clipped
 * to [−1, 1] here. `gustNoiseX/Y` are standard-normal draws for the gust update.
 */
export function controlStep(
  car: CarState,
  cond: Conditions,
  aSteerIn: number,
  aLongIn: number,
  gustNoiseX: number,
  gustNoiseY: number,
  cfg: SimConfig,
): void {
  const aSteer = fr(clip(aSteerIn, -1.0, 1.0))
  const aLong = fr(clip(aLongIn, -1.0, 1.0))
  // Executed action: the current one if D = 0, else buffer[D − 1].
  const d = cond.delaySteps
  const execSteer = d === 0 ? aSteer : car.actionBuffer[2 * (d - 1)]!
  const execLong = d === 0 ? aLong : car.actionBuffer[2 * (d - 1) + 1]!
  for (let i = 0; i < cfg.time.substeps; i++) substep(car, cond, execSteer, execLong, cfg)
  // Shift the delay line (newest first).
  car.actionBuffer.copyWithin(2, 0, car.actionBuffer.length - 2)
  car.actionBuffer[0] = aSteer
  car.actionBuffer[1] = aLong
  const dt = cfg.time.control_dt_s
  const theta = cfg.conditions.gust_theta_per_s
  const sq = Math.sqrt(dt)
  const gx = car.gustMps[0]!
  const gy = car.gustMps[1]!
  car.gustMps[0] = gx - theta * gx * dt + cond.gustSigmaMps * sq * gustNoiseX
  car.gustMps[1] = gy - theta * gy * dt + cond.gustSigmaMps * sq * gustNoiseY
}
