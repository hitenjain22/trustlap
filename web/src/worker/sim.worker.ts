/**
 * Simulation worker (SPEC 10.6): runs the environment and the drivers off the main thread
 * at a fixed 20 Hz, and posts Float32Array snapshots for the renderer to interpolate.
 *
 * Drivers: the scripted baseline, or exported neural networks (batched over all cars that
 * use the same network). Browser driving is continuous: a finished lap rolls over, and a
 * crashed car fades for 0.6 s, then respawns on the centerline a little behind.
 */
import * as Comlink from 'comlink'
import { ActorPolicy } from '../policy/actor.ts'
import { StudentEnsemble } from '../policy/ensemble.ts'
import { loadNetwork } from '../policy/loader.ts'
import type { SimConfig } from '../sim/config.ts'
import {
  layoutPuddles,
  nominalConditions,
  puddleGrip,
  puddlesForRain,
  setActivePuddles,
  setRain,
  type Conditions,
} from '../sim/conditions.ts'
import { axleGrip, lastTireForces } from '../sim/dynamics.ts'
import {
  coreStep,
  createEnvState,
  createObservation,
  createStepInfo,
  curvatureAt,
  observe,
  placePuddles,
  resetEnvState,
  type EnvState,
  type Observation,
} from '../sim/env.ts'
import { interp, riskFeatures, riskProbability } from '../sim/risk.ts'
import { PCG32 } from '../sim/rng.ts'
import { scriptedAction } from '../sim/scripted.ts'
import {
  buildTrack as splineTrack,
  generateTrack,
  stadiumTrack,
  toTrackData,
  type TrackData,
} from '../sim/track.ts'
import {
  CarField,
  CarFlag,
  CrashCause,
  PUDDLE_TEST,
  DRIVER_KINDS,
  HEADER,
  STRIDE,
  type ConditionValues,
  type DriverKind,
  type SimApi,
  type SimInit,
  type TrackGeometry,
} from './protocol.ts'

const STEP_MS = 50
const PUDDLE_FIELD = 0.85 // an axle is "in a puddle" where the puddle field cuts grip by 15%+
const PUDDLE_MEMORY_STEPS = 30 // a crash within 1.5 s of a puddle is blamed on it
const CORNER_RADIUS_M = 150 // tighter than this counts as "mid-corner"
const GOVERNOR_RAMP = 0.2 // same constants as sim/trustlap/confidence.py
const GOVERNOR_BRAKE = -0.4
const MAX_CATCH_UP = 4
const RESPAWN_STEPS = 12 // 0.6 s fade before a crashed car comes back
const RESPAWN_BACK_M = 30
const RESPAWN_SPEED_MPS = 8
const GRID_ROW_M = 9
const GRID_SIDE_M = 2.5

interface Driver {
  kind: DriverKind
  ghost: boolean
  state: EnvState
  scripted: SimConfig['scripted']
  respawnIn: number // steps left while crashed, 0 = driving
  lapFlag: boolean
  laps: number
  /** Lap timing (simulation steps) and the last action, for telemetry. */
  lapStart: number
  bestLap: number
  lastLap: number
  lastAction: Float32Array
  /** Student estimates for telemetry (NaN for other drivers). */
  muHat: Float32Array
  uncertainty: number
  /** Per-car spread of power and mass (a field of one driver) so identical drivers don't lockstep. */
  powerScale: number
  massScale: number
  /** Tire forces after the last step (body frame, N) and the most recent crash. */
  forces: Float32Array
  lastPuddleStep: number // last step with an axle in a puddle (grip field < 0.85)
  crashSpeed: number
  spinStartSpeed: number // speed when the current spin began (the car is slow by the time it counts)
  crashCause: number
  crashCount: number
}

let cfg: SimConfig
let track: TrackData
let trackSeed = 0
let drivers: Driver[] = []
let shared: Conditions // puddle arrays and condition values shared by every car
let ready = false // init finished (track built, cars placed)
let values: ConditionValues
let opts: SimInit
let rng = new PCG32(0)
let speedMultiplier = 1
let governor = { on: false, tau: 0.2 }
let pending: Partial<ConditionValues> = {} // changes that arrived before init
let pendingTrack: number | null = null
let customTrack: { points: Float64Array; widthM: number } | null = null
let puddleTest = false // cars start at 20 m/s at the stadium's s = 0
let paused = false
let simStep = 0
let crashes = 0
let listener: ((snapshot: Float32Array) => void) | null = null
let obs: Observation
const policies = new Map<DriverKind, ActorPolicy>()
let ensemble: StudentEnsemble | null = null
let histBatch = new Float32Array(0)
const pool: Float32Array[] = []
const info = createStepInfo()
const action = new Float32Array(2)
const grip = new Float64Array(2)
let obsBatch = new Float32Array(0)
let privBatch = new Float32Array(0)
let actBatch = new Float32Array(0)
let noise = new Float32Array(0)

const defaults = (c: SimConfig): ConditionValues => ({
  rain: 0,
  muDry: c.conditions.mu_dry.nominal,
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
function applyValues(cond: Conditions, d?: Pick<Driver, 'powerScale' | 'massScale'>): void {
  cond.muDry = Math.fround(values.muDry)
  setRain(cond, values.rain, cfg)
  const dir = (values.windDirDeg * Math.PI) / 180
  cond.windMps[0] = values.windMps * Math.cos(dir)
  cond.windMps[1] = values.windMps * Math.sin(dir)
  cond.gustSigmaMps = values.gustMps
  cond.massMult = values.massMult * (d?.massScale ?? 1)
  cond.powerMult = values.powerMult * (d?.powerScale ?? 1)
  cond.brakeMult = values.brakeMult
  cond.dragAreaM2 = values.dragAreaM2
  cond.wearPerKm = values.wearPerKm
  cond.steerTauS = values.steerTauS
  cond.delaySteps = values.delaySteps
  cond.obsNoise = values.obsNoise
}

async function ensurePolicies(kinds: DriverKind[]): Promise<void> {
  if (opts.checkpoint && !policies.has('teacher')) {
    policies.set(
      'teacher',
      new ActorPolicy(await loadNetwork(opts.artifactBase, opts.checkpoint, 'checkpoints')),
    )
  }
  const unique = [...new Set(kinds)]
  const needed = unique.filter((k) => (k === 'blind' || k === 'teacher') && !policies.has(k))
  const loaded = await Promise.all(needed.map((k) => loadNetwork(opts.artifactBase, k)))
  loaded.forEach((net, i) => policies.set(needed[i]!, new ActorPolicy(net)))
  if (unique.includes('student') && !ensemble) {
    const count = Math.max(1, opts.studentMembers)
    const [teacher, ...students] = await Promise.all([
      loadNetwork(opts.artifactBase, 'teacher'),
      ...Array.from({ length: count }, (_, k) => loadNetwork(opts.artifactBase, `student_${k}`)),
    ])
    ensemble = new StudentEnsemble(teacher!, students)
  }
}

/** Puddle grip field under the front and rear axle (1 = no puddle). */
function axlePuddle(state: EnvState): [number, number] {
  const car = state.car
  const c = Math.cos(car.yawRad)
  const s = Math.sin(car.yawRad)
  return [
    puddleGrip(car.xM + cfg.car.lf_m * c, car.yM + cfg.car.lf_m * s, state.cond),
    puddleGrip(car.xM - cfg.car.lr_m * c, car.yM - cfg.car.lr_m * s, state.cond),
  ]
}

const riskF = new Float64Array(4)
/** Calibrated P(crash within 2 s) for a student car (risk model, or the older table). */
function crashProbability(d: Driver): number {
  const cal = opts.calibration
  if (!cal || Number.isNaN(d.uncertainty)) return NaN
  if ('u' in cal) return interp(d.uncertainty, cal.u, cal.p)
  riskFeatures(d.state, track, d.uncertainty, d.muHat[1]!, cfg, riskF)
  return riskProbability(cal, riskF)
}

/** Build a generated track (seed ≥ 0) or a hand-drawn one (`custom` control points). */
function buildTrack(
  seed: number,
  custom?: { points: Float64Array; widthM: number },
): TrackGeometry {
  puddleTest = false
  trackSeed = custom ? -1 : seed
  track = toTrackData(
    custom
      ? splineTrack(custom.points, custom.widthM, -1, 0, 0, cfg.track)
      : generateTrack(seed, cfg.track),
  )
  shared = nominalConditions(cfg)
  // Drawn tracks have no seed of their own: puddle placement uses a fixed one.
  layoutPuddles(shared, custom ? 0 : seed, track.lengthM, track.widthM, cfg)
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
function carConditions(d: Pick<Driver, 'powerScale' | 'massScale'>): Conditions {
  const cond: Conditions = { ...shared, windMps: new Float32Array(2) }
  applyValues(cond, d)
  return cond
}

/** Slightly different scripted styles so a scripted field spreads out instead of lockstepping. */
function scriptedStyle(vary: boolean): SimConfig['scripted'] {
  const sc = cfg.scripted
  if (!vary) return sc
  return {
    ...sc,
    grip_use: sc.grip_use * rng.nextRange(0.86, 1.0),
    brake_use: sc.brake_use * rng.nextRange(0.85, 1.05),
    lookahead_time_s: sc.lookahead_time_s * rng.nextRange(0.9, 1.15),
  }
}

function placeCars(): void {
  const n = opts.nCars
  const roster: { kind: DriverKind; ghost: boolean }[] = []
  for (let i = 0; i < n; i++) roster.push({ kind: opts.driver, ghost: false })
  for (const g of opts.ghosts) if (g !== opts.driver) roster.push({ kind: g, ghost: true })
  const spread = opts.layout === 'spread'
  drivers = roster.map(({ kind, ghost }, i) => {
    // A field of identical drivers gets slightly different cars (within the training ranges).
    const field = n > 1 && !ghost
    const scales = {
      powerScale: field ? rng.nextRange(0.94, 1.0) : 1,
      massScale: field ? rng.nextRange(1.0, 1.06) : 1,
    }
    const state = createEnvState(cfg, carConditions(scales))
    // Ghosts start exactly with the first car so they race it like for like.
    const slot = ghost ? 0 : i
    const startS = spread ? (slot * track.lengthM) / n : -Math.floor(slot / 2) * GRID_ROW_M
    const startSpeed = puddleTest
      ? PUDDLE_TEST.startSpeedMps
      : spread
        ? 15
        : cfg.episode.eval_speed_mps
    resetEnvState(state, track, puddleTest ? 0 : startS, startSpeed)
    if (!spread && n > 1 && !ghost) {
      // Two-wide grid: even slots on the right, odd on the left.
      const side = slot % 2 === 0 ? -1 : 1
      state.car.xM = Math.fround(state.car.xM - side * GRID_SIDE_M * Math.sin(state.car.yawRad))
      state.car.yM = Math.fround(state.car.yM + side * GRID_SIDE_M * Math.cos(state.car.yawRad))
    }
    return {
      kind,
      ghost,
      state,
      scripted: scriptedStyle(field),
      respawnIn: 0,
      lapFlag: false,
      laps: 0,
      lapStart: 0,
      bestLap: NaN,
      lastLap: NaN,
      lastAction: new Float32Array(2),
      muHat: new Float32Array([NaN, NaN]),
      uncertainty: NaN,
      forces: new Float32Array(4),
      lastPuddleStep: -1e9,
      crashSpeed: NaN,
      spinStartSpeed: NaN,
      crashCause: 0,
      crashCount: 0,
      ...scales,
    }
  })
  simStep = 0
  crashes = 0
}

function respawn(d: Driver): void {
  resetEnvState(d.state, track, d.state.sM - RESPAWN_BACK_M, RESPAWN_SPEED_MPS)
  d.lapStart = simStep // a crashed lap doesn't count; the next one starts now
}

/** Standard normals for the sensor-noise slider (AI drivers only); null when noise is off. */
function sensorNoise(cond: Conditions): Float32Array | null {
  if (cond.obsNoise <= 0) return null
  const size = cfg.obs.policy_size - 1
  if (noise.length !== size) noise = new Float32Array(size)
  for (let k = 0; k < size; k++) noise[k] = rng.nextNormal()
  return noise
}

/** Actions for every network-driven car, batched per network → actBatch, indexed via `slot`. */
const slot = new Map<Driver, number>()
function networkActions(): void {
  const d = cfg.obs.policy_size
  const p = cfg.obs.privileged_size
  slot.clear()
  let offset = 0
  for (const [kind, policy] of policies) {
    const cars = drivers.filter((x) => x.kind === kind && x.respawnIn === 0)
    if (cars.length === 0) continue
    if (obsBatch.length < cars.length * d) obsBatch = new Float32Array(cars.length * d)
    if (privBatch.length < cars.length * p) privBatch = new Float32Array(cars.length * p)
    if (actBatch.length < 2 * (offset + cars.length)) {
      const grown = new Float32Array(2 * (offset + cars.length))
      grown.set(actBatch.subarray(0, 2 * offset))
      actBatch = grown
    }
    cars.forEach((car, b) => {
      observe(car.state, track, sensorNoise(car.state.cond), cfg, obs)
      obsBatch.set(obs.policy, b * d)
      privBatch.set(obs.privileged, b * p)
      slot.set(car, offset + b)
    })
    policy.act(obsBatch, cars.length, actBatch.subarray(2 * offset), privBatch)
    offset += cars.length
  }
  if (!ensemble) return
  const students = drivers.filter((x) => x.kind === 'student' && x.respawnIn === 0)
  if (students.length === 0) return
  const n = students.length
  const h = cfg.obs.history_size
  if (obsBatch.length < n * d) obsBatch = new Float32Array(n * d)
  if (histBatch.length < n * h) histBatch = new Float32Array(n * h)
  if (actBatch.length < 2 * (offset + n)) {
    const grown = new Float32Array(2 * (offset + n))
    grown.set(actBatch.subarray(0, 2 * offset))
    actBatch = grown
  }
  students.forEach((car, b) => {
    observe(car.state, track, sensorNoise(car.state.cond), cfg, obs)
    obsBatch.set(obs.policy, b * d)
    histBatch.set(car.state.history, b * h)
    slot.set(car, offset + b)
  })
  const members = opts.ensembleMembers ?? ensemble.size
  const out = ensemble.act(obsBatch, histBatch, n, members, members > 1)
  actBatch.set(out.actions.subarray(0, 2 * n), 2 * offset)
  students.forEach((car, b) => {
    car.muHat[0] = out.gripMean[2 * b]!
    car.muHat[1] = out.gripMean[2 * b + 1]!
    car.uncertainty = members > 1 ? out.uncertainty[b]! : NaN
  })
}

function step(): void {
  networkActions()
  for (const d of drivers) {
    d.lapFlag = false
    if (d.respawnIn > 0) {
      d.respawnIn--
      if (d.respawnIn === 0) respawn(d)
      continue
    }
    if (d.kind === 'scripted') {
      scriptedAction(d.state, track, cfg, action, d.scripted)
    } else {
      const b = slot.get(d) ?? -1
      action[0] = b >= 0 ? actBatch[2 * b]! : 0
      action[1] = b >= 0 ? actBatch[2 * b + 1]! : 0
    }
    if (d.kind === 'student' && governor.on && !Number.isNaN(d.uncertainty)) {
      // Safety governor: cap the throttle as crash risk rises past τ; brake above τ + 0.2.
      const p = crashProbability(d)
      if (p > governor.tau) {
        const cap = Math.min(1, Math.max(0, 1 - (p - governor.tau) / GOVERNOR_RAMP))
        action[1] = Math.min(action[1]!, cap)
        if (p > governor.tau + GOVERNOR_RAMP) action[1] = Math.min(action[1]!, GOVERNOR_BRAKE)
      }
    }
    const gusty = d.state.cond.gustSigmaMps > 0
    const gx = gusty ? rng.nextNormal() : 0
    const gy = gusty ? rng.nextNormal() : 0
    const speedBefore = Math.hypot(d.state.car.vxMps, d.state.car.vyMps)
    const spinningBefore = d.state.spinSteps > 0
    coreStep(d.state, action[0]!, action[1]!, gx, gy, track, cfg, info)
    d.forces.set(lastTireForces)
    if (d.state.spinSteps > 0 && !spinningBefore) d.spinStartSpeed = speedBefore
    if (Math.min(...axlePuddle(d.state)) < PUDDLE_FIELD) d.lastPuddleStep = simStep
    d.lastAction[0] = d.state.prevAction[0]!
    d.lastAction[1] = d.state.prevAction[1]!
    if (info.lapCompleted) {
      d.state.progressM -= track.lengthM // keep driving: the next lap starts now
      d.lapFlag = true
      d.laps++
      // A lap counts from the car's start (or respawn); time it in simulation steps.
      d.lastLap = ((simStep + 1 - d.lapStart) * STEP_MS) / 1000
      d.bestLap = Number.isNaN(d.bestLap) ? d.lastLap : Math.min(d.bestLap, d.lastLap)
      d.lapStart = simStep + 1
    } else if (info.terminated) {
      d.respawnIn = RESPAWN_STEPS
      if (!d.ghost) crashes++
      d.crashCount++
      d.crashSpeed =
        !info.stalled && d.state.spinSteps > 0 && Number.isFinite(d.spinStartSpeed)
          ? d.spinStartSpeed
          : speedBefore
      d.crashCause =
        (simStep - d.lastPuddleStep <= PUDDLE_MEMORY_STEPS ? CrashCause.Puddle : 0) |
        (Math.abs(curvatureAt(track, d.state.sM)) > 1 / CORNER_RADIUS_M ? CrashCause.Corner : 0) |
        (info.stalled ? CrashCause.Stalled : 0) |
        (!info.stalled && !info.wrongWay && d.state.spinSteps > 0 ? CrashCause.Spun : 0) |
        (values.rain >= 0.3 ? CrashCause.Wet : 0) |
        (d.kind === 'student' && governor.on ? CrashCause.GovernorOn : 0)
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
  buf[2] = 1 + drivers.reduce((m, d) => (d.ghost ? m : Math.max(m, d.laps)), 0)
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
    buf[o + CarField.MuHatFront] = d.muHat[0]!
    buf[o + CarField.MuHatRear] = d.muHat[1]!
    buf[o + CarField.PCrash] = crashProbability(d)
    buf[o + CarField.Flags] =
      (d.respawnIn > 0 ? CarFlag.Crashed : 0) |
      (d.lapFlag ? CarFlag.Lap : 0) |
      (d.ghost ? CarFlag.Ghost : 0) |
      (i === 0 ? CarFlag.Focused : 0)
    buf[o + CarField.Driver] = DRIVER_KINDS.indexOf(d.kind)
    buf[o + CarField.Steer] = d.lastAction[0]!
    buf[o + CarField.Long] = d.lastAction[1]!
    buf[o + CarField.LapTime] = ((simStep - d.lapStart) * STEP_MS) / 1000
    buf[o + CarField.BestLap] = d.bestLap
    buf[o + CarField.LastLap] = d.lastLap
    buf[o + CarField.FxFront] = d.forces[0]!
    buf[o + CarField.FyFront] = d.forces[1]!
    buf[o + CarField.FxRear] = d.forces[2]!
    buf[o + CarField.FyRear] = d.forces[3]!
    buf[o + CarField.CrashSpeed] = d.crashSpeed
    buf[o + CarField.CrashCause] = d.crashCause
    buf[o + CarField.Crashes] = d.crashCount
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
  async init(o: SimInit) {
    opts = o
    cfg = o.config
    obs = createObservation(cfg)
    rng = new PCG32(o.seed)
    values = { ...defaults(cfg), ...o.conditions, ...pending }
    pending = {}
    if (o.autoPuddles) values.puddles = puddlesForRain(values.rain, cfg)
    await ensurePolicies([opts.driver, ...opts.ghosts])
    // Anything the UI changed while the networks were loading wins over the initial values.
    values = { ...values, ...pending }
    pending = {}
    const geo = customTrack ? buildTrack(-1, customTrack) : buildTrack(pendingTrack ?? o.trackSeed)
    pendingTrack = null
    placeCars()
    ready = true
    start()
    return geo
  },
  async setTrack(seed: number) {
    customTrack = null
    if (!ready) {
      pendingTrack = seed // init builds it
      return null
    }
    const geo = buildTrack(seed)
    placeCars()
    return geo
  },
  async setCarCount(nCars) {
    // Drop cars where they are (every second one, from the back) instead of re-placing the
    // field: re-placing made the whole grid jump to a new layout mid-race.
    opts = { ...opts, nCars }
    if (!ready) return
    let kept = 0
    drivers = drivers.filter((d, i) => {
      if (d.ghost) return true
      const keep = i % 2 === 0 && kept < nCars
      if (keep) kept++
      return keep
    })
  },
  async startPuddleTest() {
    if (!ready) return null
    customTrack = null
    puddleTest = true
    opts = { ...opts, driver: 'student', ghosts: [] }
    values = { ...defaults(cfg), puddles: 1 }
    trackSeed = -1
    track = toTrackData(stadiumTrack(cfg.track))
    shared = nominalConditions(cfg)
    layoutPuddles(shared, 0, track.lengthM, track.widthM, cfg)
    shared.puddleSM[0] = PUDDLE_TEST.puddleSM
    shared.puddleLateralM[0] = 0
    shared.puddleMult[0] = PUDDLE_TEST.mult
    shared.puddleRadiusM[0] = PUDDLE_TEST.radiusM
    placePuddles(shared, track, 0)
    setActivePuddles(shared, 1)
    await ensurePolicies(['student'])
    placeCars()
    return geometry()
  },
  async setCustomTrack(points, widthM) {
    customTrack = { points, widthM }
    if (!ready) return null // init builds it
    const geo = buildTrack(-1, customTrack)
    placeCars()
    return geo
  },
  async setConditions(partial) {
    if (!ready) {
      // Not initialized yet (the UI can change sliders while artifacts load): remember it.
      pending = { ...pending, ...partial }
      return pending.puddles ?? 0
    }
    values = { ...values, ...partial }
    if (opts.autoPuddles && partial.rain !== undefined) {
      values.puddles = puddlesForRain(values.rain, cfg)
    }
    setActivePuddles(shared, values.puddles)
    for (const d of drivers) applyValues(d.state.cond, d)
    return values.puddles
  },
  async setDrivers(driver, ghosts) {
    if (!ready) {
      opts = { ...opts, driver, ghosts } // applied by init
      return
    }
    await ensurePolicies([driver, ...ghosts])
    opts = { ...opts, driver, ghosts }
    placeCars()
  },
  async setCheckpoint(name: string) {
    opts = { ...opts, checkpoint: name }
    policies.set(
      'teacher',
      new ActorPolicy(await loadNetwork(opts.artifactBase, name, 'checkpoints')),
    )
    placeCars()
  },
  async setGovernor(on: boolean, tau: number) {
    governor = { on, tau }
  },
  async reset() {
    placeCars()
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
