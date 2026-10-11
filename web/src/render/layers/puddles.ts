/** Puddles (SPEC 11.7, layer 4): radial pools in --grip-4, with expanding ripples in the rain. */
import type { TrackGeometry } from '../../worker/protocol.ts'
import { sx, sy, type Camera } from '../camera.ts'
import { withAlpha, type Palette } from '../palette.ts'

const RIPPLE_PERIOD_MS = 1500

export function drawPuddles(
  ctx: CanvasRenderingContext2D,
  geo: TrackGeometry,
  active: number,
  rain: number,
  cam: Camera,
  pal: Palette,
  nowMs: number,
  animate: boolean,
): void {
  for (let j = 0; j < active; j++) {
    const x = sx(cam, geo.puddles[4 * j]!)
    const y = sy(cam, geo.puddles[4 * j + 1]!)
    const r = geo.puddles[4 * j + 2]! * cam.scale
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, withAlpha(pal.grip4, 0.35))
    g.addColorStop(1, withAlpha(pal.grip4, 0))
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.ellipse(x, y, r, r * 0.82, (j * 0.7) % Math.PI, 0, 2 * Math.PI)
    ctx.fill()
    if (rain <= 0 || !animate) continue
    ctx.lineWidth = 1
    for (let k = 0; k < 3; k++) {
      const phase =
        ((nowMs + j * 397 + (k * RIPPLE_PERIOD_MS) / 3) % RIPPLE_PERIOD_MS) / RIPPLE_PERIOD_MS
      const ox = x + Math.sin(j * 3.1 + k * 1.7) * r * 0.35
      const oy = y + Math.cos(j * 2.3 + k * 2.9) * r * 0.3
      ctx.strokeStyle = withAlpha(pal.rain, 0.22 * (1 - phase) * rain)
      ctx.beginPath()
      ctx.arc(ox, oy, 1 + phase * r * 0.3, 0, 2 * Math.PI)
      ctx.stroke()
    }
  }
}
