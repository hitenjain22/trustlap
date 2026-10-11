/**
 * Simulation worker (SPEC 10.6): runs the environment and the drivers off the main thread
 * at a fixed 20 Hz, and posts Float32Array snapshots for the renderer to interpolate.
 *
 * Browser driving is continuous: a finished lap rolls over, and a crashed car fades for
 * 0.6 s, then respawns on the centerline a little behind where it left the road.
 */
import * as Comlink from 'comlink'
import type { SimConfig } from '../sim/config.ts'
import {
  layoutPuddles,
  nominalConditions,
  puddlesForRain,
  setActivePuddles,
  setRain,
  type Conditions,
} from '../sim/conditions.ts'
import { axleGrip } from '../sim/dynamics.ts'
import {
  coreStep,
  createEnvState,
  createStepInfo,
  placePuddles,
  resetEnvState,
  type EnvState,
} from '../sim/env.ts'
import { PCG32 } from '../sim/rng.ts'
import { scriptedAction } from '../sim/scripted.ts'
import { generateTrack, toTrackData, type TrackData } from '../sim/track.ts'
import {
  CarField,
  CarFlag,
  HEADER,
  STRIDE,
  type ConditionValues,
  type SimApi,
  type SimInit,
  type TrackGeometry,
} from './protocol.ts'

const STEP_MS = 50
const MAX_CATCH_UP = 4
const RESPAWN_STEPS = 12 // 0.6 s fade before a crashed car comes back
const RESPAWN_BACK_M = 30
const RESPAWN_SPEED_MPS = 8

interface Driver {
  state: EnvState
  scripted: SimConfig['scripted']
  respawnIn: number // steps left while crashed, 0 = driving
  lapFlag: boolean
  laps: number
}

let cfg: SimConfig
let track: TrackData
let trackSeed = 0
let drivers: Driver[] = []
let shared: Conditions // puddle arrays and condition values shared by every car
let values: ConditionValues
let layout: SimInit['layout'] = 'spread'
let autoPuddles = false
let rng = new PCG32(0)
let speedMultiplier = 1
let paused = false
let simStep = 0
let crashes = 0
let listener: ((snapshot: Float32Array) => void) | null = null
const pool: Float32Array[] = []
const info = createStepInfo()
const action = new Float32Array(2)
const grip = new Float64Array(2)

const defaults = (c: SimConfig): ConditionValues => ({
  rain: 0,
  puddles: 0,
  windMps: 0,
  windDirDeg: 0,
  gustMps: 0,
  massMult: c.conditions.mass_mult.nominal,
  powerMult: c.conditions.power_mult.nominal,
  brakeMult: c.conditions.brake_mult.nominal,
  dragAreaM2: c.conditions.drag_area_m2.nominal,
  wearPerKm: c.conditions.tire_wear_per_km.nominal,
  steerTauS: c.conditions.steer_tau_s.nominal,
  delaySteps: 0,
  obsNoise: 0,
})

/** Copy the UI's condition values into a car's Conditions (puddle arrays are shared). */
function applyValues(cond: Conditions): void {
  setRain(cond, values.rain, cfg)
  const dir = (values.windDirDeg * Math.PI) / 180
  cond.windMps[0] = values.windMps * Math.cos(dir)
  cond.windMps[1] = values.windMps * Math.sin(dir)
  cond.gustSigmaMps = values.gustMps
  cond.massMult = values.massMult
  cond.powerMult = values.powerMult
  cond.brakeMult = values.brakeMult
  cond.dragAreaM2 = values.dragAreaM2
  cond.wearPerKm = values.wearPerKm
  cond.steerTauS = values.steerTauS
  cond.delaySteps = values.delaySteps
  cond.obsNoise = values.obsNoise
}

function buildTrack(seed: number): TrackGeometry {
  trackSeed = seed
  track = toTrackData(generateTrack(seed, cfg.track))
  shared = nominalConditions(cfg)
  layoutPuddles(shared, seed, track.lengthM, track.widthM, cfg)
  placePuddles(shared, track, 0)
  setActivePuddles(shared, values.puddles)
  return geometry()
}

function geometry(): TrackGeometry {
  const p = shared.puddleActive.length
  const puddles = new Float32Array(4 * p)
  for (let j = 0; j < p; j++) {
    puddles[4 * j] = shared.puddleXYM[2 * j]!
    puddles[4 * j + 1] = shared.puddleXYM[2 * j + 1]!
    puddles[4 * j + 2] = shared.puddleRadiusM[j]!
    puddles[4 * j + 3] = shared.puddleMult[j]!
  }
  return {
    seed: trackSeed,
    n: track.n,
    lengthM: track.lengthM,
    widthM: track.widthM,
    centerline: track.centerline.slice(),
    normal: track.normal.slice(),
    curvature: track.curvature.slice(),
    puddles,
    puddleActive: values.puddles,
  }
}

/** A per-car copy of the shared conditions that reuses the shared puddle arrays. */
function carConditions(): Conditions {
  const cond: Conditions = { ...shared, windMps: new Float32Array(2) }
  applyValues(cond)
  return cond
}

/** Slightly different driving styles so the cars spread out and overtake. */
function driverStyle(): SimConfig['scripted'] {
  const sc = cfg.scripted
  return {
    ...sc,
    grip_use: sc.grip_use * rng.nextRange(0.86, 1.0),
    brake_use: sc.brake_use * rng.nextRange(0.85, 1.05),
    lookahead_time_s: sc.lookahead_time_s * rng.nextRange(0.9, 1.15),
  }
}

function placeCars(n: number): void {
  drivers = []
  for (let i = 0; i < n; i++) {
    const state = createEnvState(cfg, carConditions())
    const startS = layout === 'spread' ? (i * track.lengthM) / n : -i * 8
    resetEnvState(state, track, startS, layout === 'spread' ? 15 : cfg.episode.eval_speed_mps)
    drivers.push({ state, scripted: driverStyle(), respawnIn: 0, lapFlag: false, laps: 0 })
  }
  simStep = 0
  crashes = 0
}

function respawn(d: Driver): void {
  const s = d.state.sM - RESPAWN_BACK_M
  resetEnvState(d.state, track, s, RESPAWN_SPEED_MPS)
}

function step(): void {
  for (const d of drivers) {
    d.lapFlag = false
    if (d.respawnIn > 0) {
      d.respawnIn--
      if (d.respawnIn === 0) respawn(d)
      continue
    }
    scriptedAction(d.state, track, cfg, action, d.scripted)
    const gx = d.state.cond.gustSigmaMps > 0 ? rng.nextNormal() : 0
    const gy = d.state.cond.gustSigmaMps > 0 ? rng.nextNormal() : 0
    coreStep(d.state, action[0]!, action[1]!, gx, gy, track, cfg, info)
    if (info.lapCompleted) {
      d.state.progressM -= track.lengthM // keep driving: the next lap starts now
      d.lapFlag = true
      d.laps++
    } else if (info.terminated) {
      d.respawnIn = RESPAWN_STEPS
      crashes++
    }
    if (d.state.steps >= cfg.episode.max_steps - 1) d.state.steps = 0 // no time limit in the browser
  }
  simStep++
}

function snapshot(): Float32Array {
  const size = HEADER + STRIDE * drivers.length
  let buf = pool.pop()
  if (!buf || buf.length !== size) buf = new Float32Array(size)
  buf[0] = simStep * (STEP_MS / 1000)
  buf[1] = simStep
  buf[2] = 1 + drivers.reduce((m, d) => Math.max(m, d.laps), 0)
  buf[3] = crashes
  drivers.forEach((d, i) => {
    const o = HEADER + i * STRIDE
    const car = d.state.car
    buf[o + CarField.X] = car.xM
    buf[o + CarField.Y] = car.yM
    buf[o + CarField.Yaw] = car.yawRad
    buf[o + CarField.Speed] = Math.sqrt(car.vxMps * car.vxMps + car.vyMps * car.vyMps)
    buf[o + CarField.RearSlip] = Math.atan2(
      car.vyMps - cfg.car.lr_m * car.yawRateRadps,
      Math.max(car.vxMps, 0.5),
    )
    axleGrip(car, d.state.cond, cfg, grip)
    buf[o + CarField.MuFront] = grip[0]!
    buf[o + CarField.MuRear] = grip[1]!
    buf[o + CarField.MuHatFront] = NaN
    buf[o + CarField.MuHatRear] = NaN
    buf[o + CarField.PCrash] = NaN
    buf[o + CarField.Flags] =
      (d.respawnIn > 0 ? CarFlag.Crashed : 0) | (d.lapFlag ? CarFlag.Lap : 0)
  })
  return buf
}

// Fixed-timestep loop: accumulate wall time × speed, step at 20 Hz, at most 4 catch-up steps.
let last = 0
let acc = 0
let timer: ReturnType<typeof setTimeout> | undefined

function tick(): void {
  const now = performance.now()
  acc += (now - last) * speedMultiplier
  last = now
  let stepped = 0
  while (acc >= STEP_MS && stepped < MAX_CATCH_UP) {
    step()
    acc -= STEP_MS
    stepped++
  }
  if (stepped === MAX_CATCH_UP) acc = Math.min(acc, STEP_MS) // no spiral of death
  if (stepped > 0 && listener) {
    const snap = snapshot()
    listener(Comlink.transfer(snap, [snap.buffer]))
  }
  timer = setTimeout(tick, 8)
}

function start(): void {
  clearTimeout(timer)
  last = performance.now()
  acc = 0
  if (!paused) tick()
}

const api: SimApi = {
  async init(opts: SimInit) {
    cfg = opts.config
    rng = new PCG32(opts.seed)
    layout = opts.layout
    autoPuddles = opts.autoPuddles
    values = { ...defaults(cfg), ...opts.conditions }
    if (autoPuddles) values.puddles = puddlesForRain(values.rain, cfg)
    const geo = buildTrack(opts.trackSeed)
    placeCars(opts.nCars)
    start()
    return geo
  },
  async setTrack(seed: number) {
    const geo = buildTrack(seed)
    placeCars(drivers.length)
    return geo
  },
  async setConditions(partial) {
    values = { ...values, ...partial }
    if (autoPuddles && partial.rain !== undefined) values.puddles = puddlesForRain(values.rain, cfg)
    setActivePuddles(shared, values.puddles)
    for (const d of drivers) applyValues(d.state.cond)
    return values.puddles
  },
  async reset() {
    placeCars(drivers.length)
  },
  async setSpeed(multiplier: number) {
    speedMultiplier = multiplier
  },
  async pause() {
    paused = true
    clearTimeout(timer)
  },
  async resume() {
    if (!paused) return
    paused = false
    start()
  },
  async onSnapshot(callback) {
    listener = callback
  },
  async returnBuffer(buffer) {
    if (pool.length < 4) pool.push(buffer)
  },
}

Comlink.expose(api)
