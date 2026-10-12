/**
 * Contract between the main thread and the simulation worker (SPEC 10.6).
 *
 * The worker steps the simulation at a fixed 20 Hz and posts a transferable Float32Array
 * snapshot after every tick. Layout: HEADER floats, then STRIDE floats per car.
 */
import type { SimConfig } from '../sim/config.ts'

export const HEADER = 4 // [simTimeS, stepIndex, leaderLap (1-based), crashesTotal]
export const STRIDE = 24 // per car, see CarField
export const CarField = {
  X: 0,
  Y: 1,
  Yaw: 2,
  Speed: 3,
  RearSlip: 4, // radians, signed
  MuFront: 5,
  MuRear: 6,
  MuHatFront: 7, // student estimate (NaN until M5)
  MuHatRear: 8,
  PCrash: 9, // NaN until M5
  Flags: 10, // bit flags, see CarFlag
  Driver: 11, // index into DRIVER_KINDS
  Steer: 12, // last action, steering [−1, 1]
  Long: 13, // last action, throttle (+) / brake (−) [−1, 1]
  LapTime: 14, // current lap so far (s)
  BestLap: 15, // best completed lap (s), NaN until one is done
  LastLap: 16, // last completed lap (s), NaN until one is done
  FxFront: 17, // tire forces, body frame (N), last substep
  FyFront: 18,
  FxRear: 19,
  FyRear: 20,
  CrashSpeed: 21, // speed at the most recent crash (m/s), NaN before the first
  CrashCause: 22, // bit flags, see CrashCause
  Crashes: 23, // this car's crash count
} as const

/** Why the most recent crash happened, as bit flags (for the plain-words crash toast). */
export const CrashCause = {
  Puddle: 1, // drove through a puddle in the last 1.5 s
  Corner: 2, // the track was curving (radius < 150 m) where it happened
  Spun: 4, // spun out (otherwise: left the track)
  Stalled: 8, // stopped moving
  Wet: 16, // rain ≥ 30%
  GovernorOn: 32, // student with the safety governor on
} as const

/**
 * Puddle-test setup, the same as `evaluate.puddle_step` (sim/trustlap/evaluate.py): one puddle
 * (grip × 0.4, radius 8 m) 200 m down the straight, the car starting at 20 m/s.
 */
export const PUDDLE_TEST = { puddleSM: 200, mult: 0.4, radiusM: 8, startSpeedMps: 20 } as const

/** Driver identities (SPEC 11.3). Order matters: it's the snapshot's Driver index. */
export const DRIVER_KINDS = ['scripted', 'blind', 'teacher', 'student'] as const
export type DriverKind = (typeof DRIVER_KINDS)[number]
/** Which trained drivers a build ships, from the artifact manifest's file list. */
export function availableDrivers(files: string[]): DriverKind[] {
  const has = (f: string) => files.includes(`policies/${f}.json`)
  const out: DriverKind[] = ['scripted']
  if (has('blind')) out.push('blind')
  if (has('teacher')) out.push('teacher')
  if (has('teacher') && has('student_0')) out.push('student')
  return out
}

/** Number of exported students (student_0 … student_{K−1}). */
export function studentCount(files: string[]): number {
  return files.filter((f) => /^policies\/student_\d+\.json$/.test(f)).length
}
export const CarFlag = {
  Crashed: 1, // fading out after a crash
  Lap: 2, // completed a lap this step
  Ghost: 4,
  Focused: 8,
} as const

/** Condition values the UI can set (all optional; omitted keys keep their value). */
export interface ConditionValues {
  rain: number // 0..1
  muDry: number // dry-road grip (training 1.0–1.15; "black ice" 0.25)
  puddles: number // active puddles 0..8
  windMps: number
  windDirDeg: number // direction the wind blows toward, 0 = +x
  gustMps: number
  massMult: number
  powerMult: number
  brakeMult: number
  dragAreaM2: number
  wearPerKm: number
  steerTauS: number
  delaySteps: number
  obsNoise: number
}

export interface SimInit {
  config: SimConfig
  trackSeed: number
  nCars: number
  seed: number
  conditions: Partial<ConditionValues>
  /** Spread cars evenly around the lap (hero) or start them together on the grid. */
  layout: 'spread' | 'grid'
  /** Set the number of puddles from the rain level (as training does on average). */
  autoPuddles: boolean
  /** The driver of the main cars (all `nCars` of them). */
  driver: DriverKind
  /** One ghost car per listed driver, on the same track and conditions. */
  ghosts: DriverKind[]
  /** Absolute URL of the artifact folder, e.g. https://host/artifacts/v1 (workers have no base). */
  artifactBase: string
  /** Students available (from the manifest) and how many to run (1 = light mode, no uncertainty). */
  studentMembers: number
  ensembleMembers?: number
  /** Isotonic table u → P(crash in 2 s), if computed (calibration.json). */
  calibration?: { u: number[]; p: number[] }
  /** Training Lab: use this checkpoint (checkpoints/<name>) as the teacher network. */
  checkpoint?: string
}

/** Track geometry the renderer needs (float32, interleaved xy). */
export interface TrackGeometry {
  /** Generator seed, or -1 for a hand-drawn track. */
  seed: number
  n: number
  lengthM: number
  widthM: number
  centerline: Float32Array
  normal: Float32Array
  curvature: Float32Array
  puddles: Float32Array // [P × 4] x, y, radius, grip multiplier
  puddleActive: number
}

export interface SimApi {
  init(opts: SimInit): Promise<TrackGeometry>
  /** Null if called before init finished (init then builds this track). */
  setTrack(trackSeed: number): Promise<TrackGeometry | null>
  /** A hand-drawn track from validated control points (meters); geometry seed is -1. */
  setCustomTrack(controlPoints: Float64Array, widthM: number): Promise<TrackGeometry | null>
  /**
   * The puddle test (SPEC 10.2): the evaluation's straight with a single puddle, one student
   * car, dry conditions otherwise. Null before init.
   */
  startPuddleTest(): Promise<TrackGeometry | null>
  /** Re-place the cars with a new count and layout on the current track (slow-device fallback). */
  setCarCount(nCars: number, layout: SimInit['layout']): Promise<void>
  setConditions(values: Partial<ConditionValues>): Promise<number> // returns active puddles
  setDrivers(driver: DriverKind, ghosts: DriverKind[]): Promise<void>
  /** Training Lab: drive the "teacher" cars with a timeline checkpoint (checkpoints/<name>). */
  setCheckpoint(name: string): Promise<void>
  /** Safety governor for student cars (SPEC 7.7): ease off above crash risk τ. */
  setGovernor(on: boolean, tau: number): Promise<void>
  reset(): Promise<void>
  setSpeed(multiplier: number): Promise<void>
  pause(): Promise<void>
  resume(): Promise<void>
  onSnapshot(callback: (snapshot: Float32Array) => void): Promise<void>
  returnBuffer(buffer: Float32Array): Promise<void>
}
