/**
 * Driving conditions for one car (SPEC 6.2), mirroring `sim/trustlap/conditions.py`.
 *
 * Training samples conditions with jax.random; the browser instead builds them from slider
 * values (and seeded puddle placement), so only the shapes and the puddle field must match.
 */
import type { SimConfig } from './config.ts'
import { PCG32 } from './rng.ts'

export interface Conditions {
  rain: number
  muDry: number
  muBase: number // μ_dry · (1 − 0.45 · rain)
  wearPerKm: number
  massMult: number
  dragAreaM2: number
  rollingCoef: number
  powerMult: number
  brakeMult: number
  steerTauS: number
  delaySteps: number // integer 0..4
  obsNoise: number
  windMps: Float32Array // [2] world frame
  gustSigmaMps: number
  puddleSM: Float32Array // [P] arclength from the start line
  puddleLateralM: Float32Array // [P] + = left of the centerline
  puddleMult: Float32Array // [P] grip multiplier at the center
  puddleRadiusM: Float32Array // [P]
  puddleActive: Float32Array // [P] 0/1
  puddleXYM: Float32Array // [2P] world positions (filled by placePuddles)
}

/** Difficulty-0 conditions: dry, no puddles, nominal car, no lag, noise or wind. */
export function nominalConditions(cfg: SimConfig): Conditions {
  const c = cfg.conditions
  const p = c.puddle_slots
  return {
    rain: 0,
    muDry: Math.fround(c.mu_dry.nominal),
    muBase: Math.fround(c.mu_dry.nominal),
    wearPerKm: Math.fround(c.tire_wear_per_km.nominal),
    massMult: Math.fround(c.mass_mult.nominal),
    dragAreaM2: Math.fround(c.drag_area_m2.nominal),
    rollingCoef: Math.fround(c.rolling_coef.nominal),
    powerMult: Math.fround(c.power_mult.nominal),
    brakeMult: Math.fround(c.brake_mult.nominal),
    steerTauS: Math.fround(c.steer_tau_s.nominal),
    delaySteps: 0,
    obsNoise: 0,
    windMps: new Float32Array(2),
    gustSigmaMps: 0,
    puddleSM: new Float32Array(p),
    puddleLateralM: new Float32Array(p),
    puddleMult: new Float32Array(p).fill(1),
    puddleRadiusM: new Float32Array(p).fill(c.puddle_radius_min_m),
    puddleActive: new Float32Array(p),
    puddleXYM: new Float32Array(2 * p),
  }
}

/** Set the rain level and the derived base grip together. */
export function setRain(cond: Conditions, rain: number, cfg: SimConfig): void {
  cond.rain = Math.fround(rain)
  cond.muBase = Math.fround(cond.muDry * (1.0 - cfg.conditions.rain_grip_drop * cond.rain))
}

/**
 * Seeded puddle layout for a track (browser): positions, sizes and depths are fixed per
 * track; `activePuddles` decides how many are wet. Same ranges as training.
 */
export function layoutPuddles(
  cond: Conditions,
  seed: number,
  trackLengthM: number,
  trackWidthM: number,
  cfg: SimConfig,
): void {
  const c = cfg.conditions
  const rng = new PCG32(seed)
  const lat = c.puddle_lateral_frac * trackWidthM
  for (let j = 0; j < c.puddle_slots; j++) {
    cond.puddleSM[j] = rng.nextRange(c.puddle_start_margin_m, trackLengthM - c.puddle_end_margin_m)
    cond.puddleLateralM[j] = rng.nextRange(-lat, lat)
    cond.puddleMult[j] = rng.nextRange(c.puddle_mult_min, c.puddle_mult_max)
    cond.puddleRadiusM[j] = rng.nextRange(c.puddle_radius_min_m, c.puddle_radius_max_m)
  }
}

/** Activate the first `count` puddle slots (0..P). */
export function setActivePuddles(cond: Conditions, count: number): void {
  for (let j = 0; j < cond.puddleActive.length; j++) cond.puddleActive[j] = j < count ? 1 : 0
}

/** Puddles that rain level R would bring on average: round(R · 8 · 0.75). */
export function puddlesForRain(rain: number, cfg: SimConfig): number {
  const c = cfg.conditions
  return Math.min(
    c.puddle_slots,
    Math.round(rain * c.puddles_per_rain * (0.5 + c.puddle_count_jitter_min / 2)),
  )
}

/**
 * Grip multiplier field at a point: Π_j [1 − (1 − m_j) · exp(−d_j² / (2 (ρ_j/2)²))].
 * Inactive slots contribute 1.
 */
export function puddleGrip(x: number, y: number, cond: Conditions): number {
  let product = 1.0
  for (let j = 0; j < cond.puddleActive.length; j++) {
    const dx = x - cond.puddleXYM[2 * j]!
    const dy = y - cond.puddleXYM[2 * j + 1]!
    const sigma = cond.puddleRadiusM[j]! * 0.5
    const bump = Math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    product *= 1.0 - (1.0 - cond.puddleMult[j]!) * bump * cond.puddleActive[j]!
  }
  return product
}
