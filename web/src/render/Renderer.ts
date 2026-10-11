/**
 * Canvas 2D renderer (SPEC 11.7, 10.6). Draws at the display rate, interpolating between the
 * last two 20 Hz snapshots from the worker. Per frame it blits ONE cached background (static
 * layers + wet sheen + skid marks), then draws puddles, cars and rain on top.
 */
import { CarField, CarFlag, HEADER, STRIDE, type TrackGeometry } from '../worker/protocol.ts'
import { fitCamera, type Camera } from './camera.ts'
import { carScale, drawCars, drawHeadlights, type CarPose, type DriverKind } from './layers/cars.ts'
import { drawPuddles } from './layers/puddles.ts'
import { RainLayer } from './layers/rain.ts'
import { SkidLayer } from './layers/skids.ts'
import { drawStatic, edgePath, trackPath } from './layers/static.ts'
import { readPalette, withAlpha, type Palette } from './palette.ts'

const STEP_MS = 50
const INTRO_MS = 900
const SKID_LIFETIME_MS = 60_000

export interface RendererOptions {
  reducedMotion: boolean
  driver: DriverKind
  /** Called once if the first 2 s average more than 20 ms per frame. */
  onSlow?: () => void
  /** Keep the track clear of overlaid UI: CSS px to leave free on the left (e.g. hero text). */
  fitInsetLeft?: () => number
}

/** Shortest-arc angle interpolation. */
export function lerpAngle(a: number, b: number, t: number): number {
  let d = b - a
  d -= 2 * Math.PI * Math.round(d / (2 * Math.PI))
  return a + d * t
}

export class Renderer {
  readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private readonly staticLayer = document.createElement('canvas')
  private readonly background = document.createElement('canvas')
  private bgRain = -1
  private bgBuiltAt = 0
  private readonly skids = new SkidLayer()
  private readonly rainLayer = new RainLayer()
  private readonly pal: Palette = readPalette()
  private readonly opts: RendererOptions
  private geo: TrackGeometry | null = null
  private cam: Camera = { scale: 1, ox: 0, oy: 0 }
  private prev: Float32Array | null = null
  private curr: Float32Array | null = null
  private currAt = 0
  private lastFrame = 0
  private introStart = -1
  private frameTimes: number[] = []
  private slowReported = false
  private poses: CarPose[] = []
  private lastSkidStep = -1
  private raf = 0
  rain = 0
  windX = 0
  activePuddles = 0
  speed = 1
  /** Recycled snapshot buffers go back to the worker through this. */
  onRecycle: ((buf: Float32Array) => void) | null = null

  constructor(canvas: HTMLCanvasElement, opts: RendererOptions) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d', { alpha: false })!
    this.opts = opts
  }

  /** Match the canvas to its CSS size × min(devicePixelRatio, 2) and rebuild cached layers. */
  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr))
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr))
    if (w === this.canvas.width && h === this.canvas.height && this.geo) return
    this.canvas.width = w
    this.canvas.height = h
    this.staticLayer.width = this.background.width = w
    this.staticLayer.height = this.background.height = h
    this.rainLayer.resize(w, h, dpr)
    this.rebuildStatic()
  }

  setTrack(geo: TrackGeometry, intro: boolean): void {
    this.geo = geo
    this.activePuddles = geo.puddleActive
    this.prev = this.curr = null
    this.rebuildStatic()
    this.introStart = intro && !this.opts.reducedMotion ? performance.now() : -1
  }

  /** True while the track-drawing intro is playing. */
  get introPlaying(): boolean {
    return this.introStart >= 0 && performance.now() - this.introStart < INTRO_MS
  }

  /** Halve the rain for slow devices. */
  degrade(): void {
    this.rainLayer.maxStreaks = 600
  }

  private rebuildStatic(): void {
    if (!this.geo) return
    const w = this.canvas.width
    const h = this.canvas.height
    const dpr = w / Math.max(1, this.canvas.clientWidth)
    const inset = Math.min(w * 0.45, (this.opts.fitInsetLeft?.() ?? 0) * dpr)
    this.cam = fitCamera(this.geo.centerline, this.geo.widthM, w, h, 0.06, {
      left: inset,
      top: 0,
      right: w,
      bottom: h,
    })
    drawStatic(
      this.staticLayer.getContext('2d')!,
      this.geo,
      this.cam,
      this.pal,
      this.canvas.width,
      this.canvas.height,
    )
    this.bgRain = -1 // force a background rebuild
  }

  /** Static layers + wet sheen for the current rain level; clears skid marks. */
  private rebuildBackground(now: number): void {
    if (!this.geo) return
    const g = this.background.getContext('2d')!
    g.drawImage(this.staticLayer, 0, 0)
    if (this.rain > 0) {
      g.fillStyle = withAlpha(this.pal.grip4, 0.08 * this.rain)
      g.fill(trackPath(this.geo, this.cam))
    }
    this.bgRain = this.rain
    this.bgBuiltAt = now
    this.skids.reset()
  }

  pushSnapshot(buf: Float32Array): void {
    if (this.prev && this.onRecycle) this.onRecycle(this.prev)
    this.prev = this.curr
    this.curr = buf
    this.currAt = performance.now()
  }

  start(): void {
    const loop = (now: number) => {
      this.frame(now)
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
  }

  stop(): void {
    cancelAnimationFrame(this.raf)
  }

  private interpolatedPoses(): number {
    const curr = this.curr
    if (!curr) return 0
    const prev = this.prev && this.prev.length === curr.length ? this.prev : curr
    const n = (curr.length - HEADER) / STRIDE
    const t = Math.min(1, Math.max(0, ((performance.now() - this.currAt) * this.speed) / STEP_MS))
    while (this.poses.length < n) this.poses.push({ x: 0, y: 0, yaw: 0, alpha: 1 })
    this.poses.length = n
    for (let i = 0; i < n; i++) {
      const o = HEADER + i * STRIDE
      const p = this.poses[i]!
      const jumped = Math.abs(curr[o + CarField.X]! - prev[o + CarField.X]!) > 20 // respawned
      const tt = jumped ? 1 : t
      p.x = prev[o + CarField.X]! + (curr[o + CarField.X]! - prev[o + CarField.X]!) * tt
      p.y = prev[o + CarField.Y]! + (curr[o + CarField.Y]! - prev[o + CarField.Y]!) * tt
      p.yaw = lerpAngle(prev[o + CarField.Yaw]!, curr[o + CarField.Yaw]!, tt)
      p.alpha = curr[o + CarField.Flags]! & CarFlag.Crashed ? 0.35 : 1
    }
    return n
  }

  private frame(now: number): void {
    const dt = this.lastFrame ? now - this.lastFrame : 16
    this.lastFrame = now
    this.trackFrameTime(dt)
    const ctx = this.ctx
    const geo = this.geo
    const w = this.canvas.width
    const h = this.canvas.height
    if (!geo) {
      ctx.fillStyle = this.pal.runoff
      ctx.fillRect(0, 0, w, h)
      return
    }
    const motion = !this.opts.reducedMotion

    // Static layers (with the intro: surface fades in while the edges draw themselves).
    const intro = this.introStart >= 0 ? Math.min(1, (now - this.introStart) / INTRO_MS) : 1
    if (intro < 1) {
      ctx.fillStyle = this.pal.runoff
      ctx.fillRect(0, 0, w, h)
      ctx.globalAlpha = intro * intro
      ctx.drawImage(this.staticLayer, 0, 0)
      ctx.globalAlpha = 1
      const eased = 1 - Math.pow(1 - intro, 3)
      ctx.strokeStyle = this.pal.paint
      ctx.lineWidth = Math.max(0.25 * this.cam.scale, 1.5)
      const total = geo.lengthM * this.cam.scale
      ctx.setLineDash([total * eased, total])
      ctx.stroke(edgePath(geo, this.cam, 1, geo.widthM / 2))
      ctx.stroke(edgePath(geo, this.cam, -1, geo.widthM / 2))
      ctx.setLineDash([])
      return
    }
    if (Math.abs(this.rain - this.bgRain) > 0.005 || now - this.bgBuiltAt > SKID_LIFETIME_MS) {
      this.rebuildBackground(now)
    }
    const curr = this.curr
    if (curr && curr[1] !== this.lastSkidStep) {
      this.lastSkidStep = curr[1]!
      this.skids.update(
        this.background.getContext('2d')!,
        (curr.length - HEADER) / STRIDE,
        (i) => {
          const o = HEADER + i * STRIDE
          return [
            curr[o]!,
            curr[o + 1]!,
            curr[o + 2]!,
            curr[o + CarField.RearSlip]!,
            !(curr[o + CarField.Flags]! & CarFlag.Crashed),
          ]
        },
        this.cam,
      )
    }
    ctx.drawImage(this.background, 0, 0)

    drawPuddles(ctx, geo, this.activePuddles, this.rain, this.cam, this.pal, now, motion)
    this.interpolatedPoses()

    const k = carScale(this.cam, w / Math.max(1, this.canvas.clientWidth))
    if (this.rain > 0.3) drawHeadlights(ctx, this.poses, this.cam, this.pal, k)
    drawCars(ctx, this.poses, this.opts.driver, this.cam, this.pal, k)

    this.rainLayer.draw(ctx, this.rain, this.windX, motion ? Math.min(dt, 50) : 0, this.pal)
  }

  private trackFrameTime(dt: number): void {
    if (this.slowReported || this.introPlaying || !this.curr) return
    this.frameTimes.push(dt)
    const total = this.frameTimes.reduce((a, b) => a + b, 0)
    if (total < 2000) return
    const avg = total / this.frameTimes.length
    this.slowReported = true
    if (avg > 20) this.opts.onSlow?.()
  }
}
