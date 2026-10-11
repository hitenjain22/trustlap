/** Shared helpers for parity tests: load golden fixtures and rebuild TS state from them. */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parseSimConfig, type SimConfig } from '../../src/sim/config.ts'
import type { Conditions } from '../../src/sim/conditions.ts'
import type { CarState } from '../../src/sim/dynamics.ts'
import type { EnvState } from '../../src/sim/env.ts'
import { generateTrack, toTrackData, type TrackData } from '../../src/sim/track.ts'

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url))

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any

export function loadGolden(name: string): Json {
  return JSON.parse(readFileSync(here(`./golden/${name}.json`), 'utf8'))
}

export function loadConfig(): SimConfig {
  return parseSimConfig(
    JSON.parse(readFileSync(here('../../public/artifacts/v1/config.json'), 'utf8')),
  )
}

export function goldenTracks(seeds: number[], cfg: SimConfig): TrackData[] {
  return seeds.map((s) => toTrackData(generateTrack(s, cfg.track)))
}

const f32 = (xs: number[] | number[][]) => Float32Array.from((xs as number[]).flat(2) as number[])

export function carFrom(soa: Json, i: number): CarState {
  return {
    xM: soa.x_m[i],
    yM: soa.y_m[i],
    yawRad: soa.yaw_rad[i],
    vxMps: soa.vx_mps[i],
    vyMps: soa.vy_mps[i],
    yawRateRadps: soa.yaw_rate_radps[i],
    steerRad: soa.steer_rad[i],
    actionBuffer: f32(soa.action_buffer[i]),
    gustMps: f32(soa.gust_mps[i]),
    odometerM: soa.odometer_m[i],
  }
}

export function condFrom(soa: Json, i: number): Conditions {
  return {
    rain: soa.rain[i],
    muDry: soa.mu_dry[i],
    muBase: soa.mu_base[i],
    wearPerKm: soa.wear_per_km[i],
    massMult: soa.mass_mult[i],
    dragAreaM2: soa.drag_area_m2[i],
    rollingCoef: soa.rolling_coef[i],
    powerMult: soa.power_mult[i],
    brakeMult: soa.brake_mult[i],
    steerTauS: soa.steer_tau_s[i],
    delaySteps: soa.delay_steps[i],
    obsNoise: soa.obs_noise[i],
    windMps: f32(soa.wind_mps[i]),
    gustSigmaMps: soa.gust_sigma_mps[i],
    puddleSM: f32(soa.puddle_s_m[i]),
    puddleLateralM: f32(soa.puddle_lateral_m[i]),
    puddleMult: f32(soa.puddle_mult[i]),
    puddleRadiusM: f32(soa.puddle_radius_m[i]),
    puddleActive: f32(soa.puddle_active[i]),
    puddleXYM: f32(soa.puddle_xy_m[i]),
  }
}

/** Env state for case i; histories beyond `historyCases` are zeros (as exported). */
export function envStateFrom(soa: Json, i: number, cfg: SimConfig, historyCases: number): EnvState {
  const history = new Float32Array(cfg.obs.history_size)
  if (i < historyCases) history.set(f32(soa.history[i]))
  return {
    car: carFrom(soa.car, i),
    cond: condFrom(soa.cond, i),
    idx: soa.idx[i],
    sM: soa.s_m[i],
    progressM: soa.progress_m[i],
    steps: soa.steps[i],
    spinSteps: soa.spin_steps[i],
    stallSteps: soa.stall_steps[i],
    prevAction: f32(soa.prev_action[i]),
    lastProprio: f32(soa.last_proprio[i]),
    history,
    historyLen: soa.history_len[i],
  }
}

/** Tolerance check |a − b| ≤ atol + rtol·|a| that collects the worst offenders. */
export class Tolerance {
  failures: string[] = []
  worst = 0
  readonly atol: number
  readonly rtol: number

  constructor(atol = 1e-4, rtol = 1e-4) {
    this.atol = atol
    this.rtol = rtol
  }

  check(label: string, expected: number, actual: number, angle = false): void {
    let diff = Math.abs(expected - actual)
    if (angle) diff = Math.min(diff, Math.abs(diff - 2 * Math.PI))
    const limit = this.atol + this.rtol * Math.abs(expected)
    this.worst = Math.max(this.worst, diff / limit)
    if (!(diff <= limit) && this.failures.length < 10) {
      this.failures.push(
        `${label}: expected ${expected}, got ${actual} (diff ${diff.toExponential(2)})`,
      )
    }
  }
}
