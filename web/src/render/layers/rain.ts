/**
 * Rain (SPEC 11.7, layer 8): up to 1,200 screen-space streaks, count ∝ rain level, angled by
 * the wind, drawn as one batched path. Reduced motion: a static streak texture.
 */
import { withAlpha, type Palette } from '../palette.ts'

const MAX_STREAKS = 1200

export class RainLayer {
  private readonly xs = new Float32Array(MAX_STREAKS)
  private readonly ys = new Float32Array(MAX_STREAKS)
  private readonly lens = new Float32Array(MAX_STREAKS)
  private readonly speeds = new Float32Array(MAX_STREAKS)
  private w = 1
  private h = 1
  private seed = 987654321
  maxStreaks = MAX_STREAKS

  private rand(): number {
    this.seed = (this.seed * 1664525 + 1013904223) >>> 0
    return this.seed / 4294967296
  }

  resize(w: number, h: number, dpr: number): void {
    this.w = w
    this.h = h
    for (let i = 0; i < MAX_STREAKS; i++) {
      this.xs[i] = this.rand() * w
      this.ys[i] = this.rand() * h
      this.lens[i] = (10 + this.rand() * 8) * dpr
      this.speeds[i] = (0.9 + this.rand() * 0.5) * dpr
    }
  }

  /**
   * Draw `rain` (0..1) worth of streaks. `windX` (m/s, screen-right positive) tilts them.
   * `dtMs` = time since the last frame (0 when motion is reduced).
   */
  draw(
    ctx: CanvasRenderingContext2D,
    rain: number,
    windX: number,
    dtMs: number,
    pal: Palette,
  ): void {
    const count = Math.round(rain * this.maxStreaks)
    if (count === 0) return
    const tilt = Math.max(-0.6, Math.min(0.6, windX / 25))
    const fall = 0.9 * dtMs // px per ms, scaled per streak
    ctx.strokeStyle = withAlpha(pal.rain, 0.25)
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let i = 0; i < count; i++) {
      let y = this.ys[i]! + fall * this.speeds[i]!
      let x = this.xs[i]! + fall * this.speeds[i]! * tilt
      if (y > this.h) {
        y -= this.h + this.lens[i]!
        x = this.rand() * this.w
      }
      if (x > this.w) x -= this.w
      if (x < 0) x += this.w
      this.ys[i] = y
      this.xs[i] = x
      const len = this.lens[i]!
      ctx.moveTo(x, y)
      ctx.lineTo(x - tilt * len, y - len)
    }
    ctx.stroke()
  }
}
