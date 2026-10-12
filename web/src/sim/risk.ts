/**
 * Crash-risk model, mirroring sim/trustlap/confidence.py (risk_features, risk_predict):
 * p = isotonic(sigmoid(w · (f − mean) / std + b)) over four features, or the earlier
 * disagreement-only table for artifacts exported before the model existed.
 */
import type { SimConfig } from './config.ts'
import type { EnvState } from './env.ts'
import type { TrackData } from './track.ts'

export interface RiskModel {
  features: string[]
  mean: number[]
  std: number[]
  w: number[]
  b: number
  table_s: number[]
  table_p: number[]
}

/** What calibration.json holds: the model, or the older table over disagreement u. */
export type RiskCalibration = { model: RiskModel } | { u: number[]; p: number[] }

/** numpy.interp: piecewise-linear, clamped at the ends. */
export function interp(x: number, xs: ArrayLike<number>, ps: ArrayLike<number>): number {
  const n = xs.length
  if (n === 0) return NaN
  if (x <= xs[0]!) return ps[0]!
  if (x >= xs[n - 1]!) return ps[n - 1]!
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs[mid]! <= x) lo = mid
    else hi = mid
  }
  const span = xs[hi]! - xs[lo]!
  return span > 0 ? ps[lo]! + ((ps[hi]! - ps[lo]!) * (x - xs[lo]!)) / span : ps[hi]!
}

/**
 * [disagreement, speed (m/s), sharpest curvature ahead (1/m), grip demand v²κ/(μ̂g)] for one car.
 * The lookahead and grip floor come from config.json (ensemble section).
 */
export function riskFeatures(
  state: EnvState,
  track: TrackData,
  uncertainty: number,
  muHatRear: number,
  cfg: SimConfig,
  out = new Float64Array(4),
): Float64Array {
  const look = cfg.ensemble?.risk_lookahead_points
  const minGrip = cfg.ensemble?.risk_min_grip
  if (look === undefined || minGrip === undefined) {
    throw new Error('config.json lacks ensemble.risk_* (re-export with the risk model)')
  }
  const car = state.car
  const speed = Math.sqrt(car.vxMps * car.vxMps + car.vyMps * car.vyMps)
  let kappa = 0
  for (let j = 0; j < look; j++) {
    kappa = Math.max(kappa, Math.abs(track.curvature[(state.idx + j) % track.n]!))
  }
  const demand = (speed * speed * kappa) / (Math.max(muHatRear, minGrip) * cfg.car.gravity_mps2)
  out[0] = uncertainty
  out[1] = speed
  out[2] = kappa
  out[3] = demand
  return out
}

/** Calibrated P(crash within 2 s) from the features (model) or from disagreement alone. */
export function riskProbability(cal: RiskCalibration, f: ArrayLike<number>): number {
  if ('u' in cal) return interp(f[0]!, cal.u, cal.p)
  const m = cal.model
  let z = m.b
  for (let i = 0; i < m.w.length; i++) z += (m.w[i]! * (f[i]! - m.mean[i]!)) / m.std[i]!
  return interp(1 / (1 + Math.exp(-z)), m.table_s, m.table_p)
}
