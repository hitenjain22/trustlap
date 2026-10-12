/**
 * Runs a live simulation in a Web Worker and draws it on a canvas (SPEC 10.6).
 * Handles: loading the config, track-draw intro then launch, pausing on hidden tabs,
 * the slow-device fallback, and recycling snapshot buffers.
 */
import * as Comlink from 'comlink'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ARTIFACT_VERSION,
  loadCalibration,
  loadManifest,
  loadSimConfig,
  type Calibration,
} from '../data/artifacts.ts'
import { Renderer } from '../render/Renderer.ts'
import type { SimConfig } from '../sim/config.ts'
import {
  CarField,
  CarFlag,
  DRIVER_KINDS,
  HEADER,
  STRIDE,
  type ConditionValues,
  type DriverKind,
  availableDrivers,
  studentCount,
  type SimApi,
  type SimInit,
  type TrackGeometry,
} from '../worker/protocol.ts'

export interface LiveSimOptions {
  nCars: number
  /** A seed, or a promise of one (e.g. the first showcase track once it has loaded). */
  trackSeed: number | Promise<number>
  conditions: Partial<ConditionValues>
  layout: 'spread' | 'grid'
  driver: DriverKind
  ghosts?: DriverKind[]
  showFocus?: boolean
  /** Sample the focused car for a telemetry panel. */
  telemetry?: boolean
  /** Training Lab: drive the teacher cars with this timeline checkpoint. */
  checkpoint?: string
  /** How many students the student driver runs (1 = light, no crash-risk estimate). */
  ensembleMembers?: number
  intro: boolean
  seed?: number
  autoPuddles?: boolean
  fitInsetLeft?: () => number
  /** Cap on rain streaks (phones use fewer; default 1,200). */
  maxRainStreaks?: number
}

/** The focused car's live readings (Playground telemetry), sampled at 10 Hz. */
export interface Telemetry {
  kind: DriverKind
  speedMps: number
  lapTime: number
  bestLap: number
  lastLap: number
  ghostLastLap: number // first ghost's last lap (NaN without a ghost or a lap)
  muFront: number
  muRear: number
  muHatFront: number // NaN unless the driver estimates grip (student)
  muHatRear: number
  pCrash: number // NaN unless calibrated ensemble
  steer: number
  long: number
  crashed: boolean
  /** The focused car's crash count, speed (m/s) and cause flags at its latest crash. */
  crash: { count: number; speedMps: number; cause: number }
  /** Crashes per driver kind (focused car and ghosts). */
  crashesByDriver: Partial<Record<DriverKind, number>>
  /** Tire forces of the focused car, body frame (N): front x/y, rear x/y. */
  forces: [number, number, number, number]
  /** Last 10 s (100 samples): sim time (s), speed, true and estimated rear grip, crash risk. */
  history: {
    t: number[]
    speed: number[]
    muRear: number[]
    muHatRear: number[]
    pCrash: number[]
  }
}

const HISTORY = 100

export interface LiveSimStatus {
  phase: 'loading' | 'ready' | 'error'
  cars: number
  leaderLap: number
  crashes: number
  degraded: boolean
  track: TrackGeometry | null
  /** Drivers this build ships (from the artifact manifest). */
  available: DriverKind[]
}

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

export function useLiveSim(opts: LiveSimOptions) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const apiRef = useRef<Comlink.Remote<SimApi> | null>(null)
  const rendererRef = useRef<Renderer | null>(null)
  const [status, setStatus] = useState<LiveSimStatus>({
    phase: 'loading',
    cars: opts.nCars,
    leaderLap: 1,
    crashes: 0,
    degraded: false,
    track: null,
    available: ['scripted'],
  })
  const initial = useRef(opts)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const valuesRef = useRef<Partial<ConditionValues>>({ ...opts.conditions })
  const driversRef = useRef({ driver: opts.driver, ghosts: opts.ghosts ?? [] })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const o = initial.current
    const reduced = prefersReducedMotion()
    let disposed = false
    let lastStatus = 0
    let lastTelemetry = 0
    const worker = new Worker(new URL('../worker/sim.worker.ts', import.meta.url), {
      type: 'module',
    })
    const api = Comlink.wrap<SimApi>(worker)
    apiRef.current = api
    let students = 0
    let calibration: Calibration | null = null
    // Built from the latest conditions and drivers, so a restart (slow device) keeps them.
    const initArgs = (
      cfg: SimConfig,
      seed: number,
      nCars: number,
      layout: SimInit['layout'],
    ): SimInit => ({
      config: cfg,
      trackSeed: seed,
      nCars,
      seed: o.seed ?? 1,
      conditions: valuesRef.current,
      layout,
      autoPuddles: o.autoPuddles ?? false,
      driver: driversRef.current.driver,
      ghosts: driversRef.current.ghosts,
      artifactBase: `${window.location.origin}/artifacts/${ARTIFACT_VERSION}`,
      studentMembers: students,
      ensembleMembers: o.ensembleMembers,
      checkpoint: o.checkpoint,
      calibration: calibration
        ? 'model' in calibration
          ? { model: calibration.model }
          : { u: calibration.u, p: calibration.p }
        : undefined,
    })
    const renderer = new Renderer(canvas, {
      reducedMotion: reduced,
      showFocus: o.showFocus ?? false,
      fitInsetLeft: o.fitInsetLeft,
      maxRainStreaks: o.maxRainStreaks,
      onSlow: () => {
        // Halve the work on slow devices (SPEC 10.3) by restarting with half the cars.
        // Keeps whatever track is showing (a new, drawn or puddle-test track included).
        renderer.degrade()
        const half = Math.max(1, Math.floor(o.nCars / 2))
        if (half === o.nCars) return // nothing to drop (e.g. the Playground's single car)
        void api.setCarCount(half).then(() => {
          setStatus((s) => ({ ...s, cars: half, degraded: true }))
        })
      },
    })
    rendererRef.current = renderer
    renderer.rain = o.conditions.rain ?? 0
    renderer.onRecycle = (buf) => void api.returnBuffer(Comlink.transfer(buf, [buf.buffer]))

    const resize = new ResizeObserver(() => renderer.resize())
    resize.observe(canvas)
    renderer.resize()
    renderer.start()

    const onVisibility = () => void (document.hidden ? api.pause() : api.resume())
    document.addEventListener('visibilitychange', onVisibility)

    Promise.all([loadSimConfig(), Promise.resolve(o.trackSeed), loadManifest(), loadCalibration()])
      .then(async ([cfg, seed, manifest, cal]) => {
        calibration = cal
        const files = manifest.files.map((f) => f.path)
        students = studentCount(files)
        const available = availableDrivers(files)
        // Fall back to the best shipped driver if the requested one isn't in this build.
        const pick = (k: DriverKind) =>
          available.includes(k) ? k : available[available.length - 1]!
        driversRef.current = {
          driver: pick(driversRef.current.driver),
          ghosts: driversRef.current.ghosts.filter((g) => available.includes(g)),
        }
        setStatus((s) => ({ ...s, available }))
        await api.onSnapshot(
          Comlink.proxy((snap: Float32Array) => {
            renderer.pushSnapshot(snap)
            const now = performance.now()
            if (o.telemetry && now - lastTelemetry >= 100) {
              lastTelemetry = now
              setTelemetry((prev) => readTelemetry(snap, prev))
            }
            if (now - lastStatus > 250) {
              lastStatus = now
              // Mean speed of cars on track, published for tests and the a11y summary.
              let sum = 0
              let count = 0
              for (let o = HEADER; o < snap.length; o += STRIDE) {
                if (snap[o + CarField.Flags]! & CarFlag.Crashed) continue
                sum += snap[o + CarField.Speed]!
                count++
              }
              canvas.dataset.meanSpeed = count ? (sum / count).toFixed(2) : '0'
              canvas.dataset.step = String(snap[1])
              setStatus((s) => ({ ...s, leaderLap: snap[2]!, crashes: snap[3]! }))
            }
          }),
        )
        if (o.intro && !reduced) await api.pause()
        const geo = await api.init(initArgs(cfg, seed, o.nCars, o.layout))
        if (disposed) return
        renderer.setTrack(geo, o.intro)
        setStatus((s) => ({ ...s, phase: 'ready', track: geo }))
        if (o.intro && !reduced) setTimeout(() => void api.resume(), 950)
      })
      .catch((error) => {
        console.error(error)
        if (!disposed) setStatus((s) => ({ ...s, phase: 'error' }))
      })

    return () => {
      disposed = true
      renderer.stop()
      resize.disconnect()
      document.removeEventListener('visibilitychange', onVisibility)
      api[Comlink.releaseProxy]()
      worker.terminate()
    }
  }, [])

  const setConditions = useCallback(async (values: Partial<ConditionValues>) => {
    const api = apiRef.current
    const renderer = rendererRef.current
    if (!api || !renderer) return
    valuesRef.current = { ...valuesRef.current, ...values }
    if (values.rain !== undefined) renderer.rain = values.rain
    if (values.windMps !== undefined || values.windDirDeg !== undefined) {
      const dir = ((values.windDirDeg ?? 0) * Math.PI) / 180
      renderer.windX = (values.windMps ?? 0) * Math.cos(dir)
    }
    renderer.activePuddles = await api.setConditions(values)
  }, [])

  const setTrack = useCallback(async (seed: number) => {
    const api = apiRef.current
    const renderer = rendererRef.current
    if (!api || !renderer) return
    const geo = await api.setTrack(seed)
    if (!geo) return // still loading: init will build this track
    renderer.setTrack(geo, false)
    setStatus((s) => ({ ...s, track: geo, crashes: 0, leaderLap: 1 }))
  }, [])

  const setCustomTrack = useCallback(async (points: Float64Array, widthM: number) => {
    const api = apiRef.current
    const renderer = rendererRef.current
    if (!api || !renderer) return
    const geo = await api.setCustomTrack(points, widthM)
    if (!geo) return
    renderer.setTrack(geo, false)
    setStatus((s) => ({ ...s, track: geo, crashes: 0, leaderLap: 1 }))
  }, [])

  const startPuddleTest = useCallback(async () => {
    const api = apiRef.current
    const renderer = rendererRef.current
    if (!api || !renderer) return
    const geo = await api.startPuddleTest()
    if (!geo) return
    renderer.setTrack(geo, false)
    setTelemetry(null) // fresh history for the test chart
    setStatus((s) => ({ ...s, track: geo, crashes: 0, leaderLap: 1 }))
  }, [])

  /** Renderer-only view options: camera, tire-force arrows, confidence halo, chase labels. */
  const setView = useCallback(
    (v: Partial<Pick<Renderer, 'camera' | 'showForces' | 'haloTau' | 'labels'>>) => {
      const renderer = rendererRef.current
      if (renderer) Object.assign(renderer, v)
    },
    [],
  )

  const control = useCallback(async (action: 'pause' | 'resume' | 'reset') => {
    const api = apiRef.current
    if (!api) return
    if (action === 'pause') await api.pause()
    else if (action === 'resume') await api.resume()
    else await api.reset()
  }, [])

  const setSpeed = useCallback(async (multiplier: number) => {
    if (rendererRef.current) rendererRef.current.speed = multiplier
    await apiRef.current?.setSpeed(multiplier)
  }, [])

  const setDrivers = useCallback(async (driver: DriverKind, ghosts: DriverKind[]) => {
    driversRef.current = { driver, ghosts }
    await apiRef.current?.setDrivers(driver, ghosts)
  }, [])

  const setGovernor = useCallback(async (on: boolean, tau: number) => {
    await apiRef.current?.setGovernor(on, tau)
  }, [])

  const setCheckpoint = useCallback(async (name: string) => {
    await apiRef.current?.setCheckpoint(name)
  }, [])

  return {
    canvasRef,
    status,
    telemetry,
    setConditions,
    setTrack,
    setCustomTrack,
    setView,
    startPuddleTest,
    control,
    setSpeed,
    setDrivers,
    setCheckpoint,
    setGovernor,
  }
}

function push(arr: number[], v: number): number[] {
  const out = arr.length >= HISTORY ? arr.slice(arr.length - HISTORY + 1) : arr.slice()
  out.push(v)
  return out
}

/** Focused car = car 0; first ghost = the first car flagged as a ghost. */
function readTelemetry(snap: Float32Array, prev: Telemetry | null): Telemetry {
  const n = (snap.length - HEADER) / STRIDE
  let ghostLast = NaN
  for (let i = 1; i < n; i++) {
    const g = HEADER + i * STRIDE
    if (snap[g + CarField.Flags]! & CarFlag.Ghost) {
      ghostLast = snap[g + CarField.LastLap]!
      break
    }
  }
  const f = (k: number) => snap[HEADER + k]!
  const crashesByDriver: Partial<Record<DriverKind, number>> = {}
  for (let i = 0; i < n; i++) {
    const o = HEADER + i * STRIDE
    const kind = DRIVER_KINDS[snap[o + CarField.Driver]!] ?? 'scripted'
    crashesByDriver[kind] = (crashesByDriver[kind] ?? 0) + snap[o + CarField.Crashes]!
  }
  const h = prev?.history ?? { t: [], speed: [], muRear: [], muHatRear: [], pCrash: [] }
  return {
    kind: DRIVER_KINDS[f(CarField.Driver)] ?? 'scripted',
    speedMps: f(CarField.Speed),
    lapTime: f(CarField.LapTime),
    bestLap: f(CarField.BestLap),
    lastLap: f(CarField.LastLap),
    ghostLastLap: ghostLast,
    muFront: f(CarField.MuFront),
    muRear: f(CarField.MuRear),
    muHatFront: f(CarField.MuHatFront),
    muHatRear: f(CarField.MuHatRear),
    pCrash: f(CarField.PCrash),
    steer: f(CarField.Steer),
    long: f(CarField.Long),
    crashed: (f(CarField.Flags) & CarFlag.Crashed) !== 0,
    crash: {
      count: f(CarField.Crashes),
      speedMps: f(CarField.CrashSpeed),
      cause: f(CarField.CrashCause),
    },
    crashesByDriver,
    forces: [f(CarField.FxFront), f(CarField.FyFront), f(CarField.FxRear), f(CarField.FyRear)],
    history: {
      t: push(h.t, snap[0]!),
      speed: push(h.speed, f(CarField.Speed)),
      muRear: push(h.muRear, f(CarField.MuRear)),
      muHatRear: push(h.muHatRear, f(CarField.MuHatRear)),
      pCrash: push(h.pCrash, f(CarField.PCrash)),
    },
  }
}
