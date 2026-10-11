/**
 * Runs a live simulation in a Web Worker and draws it on a canvas (SPEC 10.6).
 * Handles: loading the config, track-draw intro then launch, pausing on hidden tabs,
 * the slow-device fallback, and recycling snapshot buffers.
 */
import * as Comlink from 'comlink'
import { useCallback, useEffect, useRef, useState } from 'react'
import { loadSimConfig } from '../data/artifacts.ts'
import { Renderer } from '../render/Renderer.ts'
import type { DriverKind } from '../render/layers/cars.ts'
import {
  CarField,
  CarFlag,
  HEADER,
  STRIDE,
  type ConditionValues,
  type SimApi,
  type TrackGeometry,
} from '../worker/protocol.ts'

export interface LiveSimOptions {
  nCars: number
  /** A seed, or a promise of one (e.g. the first showcase track once it has loaded). */
  trackSeed: number | Promise<number>
  conditions: Partial<ConditionValues>
  layout: 'spread' | 'grid'
  driver: DriverKind
  intro: boolean
  seed?: number
  autoPuddles?: boolean
  fitInsetLeft?: () => number
}

export interface LiveSimStatus {
  phase: 'loading' | 'ready' | 'error'
  cars: number
  leaderLap: number
  crashes: number
  degraded: boolean
  track: TrackGeometry | null
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
  })
  const initial = useRef(opts)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const o = initial.current
    const reduced = prefersReducedMotion()
    let disposed = false
    let lastStatus = 0
    const worker = new Worker(new URL('../worker/sim.worker.ts', import.meta.url), {
      type: 'module',
    })
    const api = Comlink.wrap<SimApi>(worker)
    apiRef.current = api
    let cfgRef: Awaited<ReturnType<typeof loadSimConfig>> | null = null
    let trackSeed = 0
    const renderer = new Renderer(canvas, {
      reducedMotion: reduced,
      driver: o.driver,
      fitInsetLeft: o.fitInsetLeft,
      onSlow: () => {
        // Halve the work on slow devices (SPEC 10.3) by restarting with half the cars.
        if (!cfgRef) return
        renderer.degrade()
        const half = Math.max(1, Math.floor(o.nCars / 2))
        void api
          .init({
            config: cfgRef,
            trackSeed: trackSeed,
            nCars: half,
            seed: o.seed ?? 1,
            conditions: o.conditions,
            layout: 'spread',
            autoPuddles: o.autoPuddles ?? false,
          })
          .then((geo) => {
            renderer.setTrack(geo, false)
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

    Promise.all([loadSimConfig(), Promise.resolve(o.trackSeed)])
      .then(async ([cfg, seed]) => {
        cfgRef = cfg
        trackSeed = seed
        await api.onSnapshot(
          Comlink.proxy((snap: Float32Array) => {
            renderer.pushSnapshot(snap)
            const now = performance.now()
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
        const geo = await api.init({
          config: cfg,
          trackSeed: seed,
          nCars: o.nCars,
          seed: o.seed ?? 1,
          conditions: o.conditions,
          layout: o.layout,
          autoPuddles: o.autoPuddles ?? false,
        })
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
    renderer.setTrack(geo, false)
    setStatus((s) => ({ ...s, track: geo, crashes: 0, leaderLap: 1 }))
  }, [])

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

  return { canvasRef, status, setConditions, setTrack, control, setSpeed }
}
