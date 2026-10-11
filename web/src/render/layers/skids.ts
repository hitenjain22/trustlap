/**
 * Skid marks (SPEC 11.7, layer 5): drawn into the renderer's cached background whenever a car's
 * rear slip angle exceeds 6°, as two 0.3 m strokes for the rear wheels. They persist until the
 * background is rebuilt (track, size or rain change, or every 60 s in the endless hero).
 */
import { sx, sy, type Camera } from '../camera.ts'

const SLIP_THRESHOLD_RAD = (6 * Math.PI) / 180
const REAR_AXLE_M = 1.4
const HALF_TRACK_M = 0.8

export class SkidLayer {
  private prev = new Float32Array(0) // per car: left x, y, right x, y (screen px), NaN = no mark

  /** Forget the previous wheel positions (after the background is rebuilt or the track changes). */
  reset(): void {
    this.prev.fill(NaN)
  }

  /** Add marks for cars sliding this step. `get(i)` = [x, y, yaw, rear slip, visible] (world). */
  update(
    ctx: CanvasRenderingContext2D,
    n: number,
    get: (i: number) => [number, number, number, number, boolean],
    cam: Camera,
  ): void {
    if (this.prev.length !== 4 * n) this.prev = new Float32Array(4 * n).fill(NaN)
    ctx.strokeStyle = 'rgba(15, 18, 21, 0.35)'
    ctx.lineWidth = Math.max(0.3 * cam.scale, 1)
    ctx.lineCap = 'round'
    ctx.beginPath()
    for (let i = 0; i < n; i++) {
      const [x, y, yaw, slip, visible] = get(i)
      const o = 4 * i
      if (!visible || Math.abs(slip) < SLIP_THRESHOLD_RAD) {
        this.prev[o] = NaN
        continue
      }
      const c = Math.cos(yaw)
      const s = Math.sin(yaw)
      const rx = x - REAR_AXLE_M * c
      const ry = y - REAR_AXLE_M * s
      const lx = sx(cam, rx - HALF_TRACK_M * s)
      const ly = sy(cam, ry + HALF_TRACK_M * c)
      const rrx = sx(cam, rx + HALF_TRACK_M * s)
      const rry = sy(cam, ry - HALF_TRACK_M * c)
      if (!Number.isNaN(this.prev[o]!)) {
        ctx.moveTo(this.prev[o]!, this.prev[o + 1]!)
        ctx.lineTo(lx, ly)
        ctx.moveTo(this.prev[o + 2]!, this.prev[o + 3]!)
        ctx.lineTo(rrx, rry)
      }
      this.prev[o] = lx
      this.prev[o + 1] = ly
      this.prev[o + 2] = rrx
      this.prev[o + 3] = rry
    }
    ctx.stroke()
  }
}
