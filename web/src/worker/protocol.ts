/**
 * Contract between the main thread and the simulation worker (SPEC 10.6).
 *
 * The worker steps the simulation at a fixed 20 Hz and posts a transferable Float32Array
 * snapshot after every tick. Layout: HEADER floats, then STRIDE floats per car.
 */
import type { SimConfig } from '../sim/config.ts'

export const HEADER = 4 // [simTimeS, stepIndex, leaderLap (1-based), crashesTotal]
export const STRIDE = 11 // per car, see CarField
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
} as const
export const CarFlag = {
  Crashed: 1, // fading out after a crash
  Lap: 2, // completed a lap this step
  Ghost: 4,
  Focused: 8,
} as const

/** Condition values the UI can set (all optional; omitted keys keep their value). */
export interface ConditionValues {
  rain: number // 0..1
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
}

/** Track geometry the renderer needs (float32, interleaved xy). */
export interface TrackGeometry {
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
  setTrack(trackSeed: number): Promise<TrackGeometry>
  setConditions(values: Partial<ConditionValues>): Promise<number> // returns active puddles
  reset(): Promise<void>
  setSpeed(multiplier: number): Promise<void>
  pause(): Promise<void>
  resume(): Promise<void>
  onSnapshot(callback: (snapshot: Float32Array) => void): Promise<void>
  returnBuffer(buffer: Float32Array): Promise<void>
}
