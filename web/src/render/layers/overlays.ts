/**
 * Overlays (SPEC 11.7 layer 9): tire-force arrows on the focused car, the confidence halo when
 * the student's crash risk passes τ, and driver labels (Chase camera only).
 */
import type { DriverKind } from '../../worker/protocol.ts'
import { sx, sy, type Camera } from '../camera.ts'
import { withAlpha, type Palette } from '../palette.ts'
import type { CarPose } from './cars.ts'

/** Axle positions along the body, matching the drawn 4.4 m car (front +1.3 m, rear −1.3 m). */
const AXLE_M = 1.3
/** Arrow length per newton (m/N): a full axle load of ~7 kN draws about 10 m. */
const M_PER_N = 1 / 700
const HALO_PERIOD_MS = 1200

function arrow(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: string,
  width: number,
): void {
  const len = Math.hypot(x1 - x0, y1 - y0)
  if (len < width * 2) return
  const ux = (x1 - x0) / len
  const uy = (y1 - y0) / len
  const head = Math.min(len * 0.4, width * 4)
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = width
  ctx.beginPath()
  ctx.moveTo(x0, y0)
  ctx.lineTo(x1 - ux * head * 0.8, y1 - uy * head * 0.8)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x1 - ux * head - uy * head * 0.5, y1 - uy * head + ux * head * 0.5)
  ctx.lineTo(x1 - ux * head + uy * head * 0.5, y1 - uy * head - ux * head * 0.5)
  ctx.closePath()
  ctx.fill()
}

/**
 * Tire forces at each axle, body frame (N): [fxFront, fyFront, fxRear, fyRear]. Longitudinal in
 * paint, lateral in sodium; `k` is the car-size enlargement so arrows stay in proportion.
 */
export function drawForces(
  ctx: CanvasRenderingContext2D,
  car: CarPose,
  forces: ArrayLike<number>,
  cam: Camera,
  pal: Palette,
  k: number,
  dpr: number,
): void {
  const c = Math.cos(car.yaw)
  const s = Math.sin(car.yaw)
  const width = 2 * dpr
  for (const [axle, fx, fy] of [
    [AXLE_M, forces[0]!, forces[1]!],
    [-AXLE_M, forces[2]!, forces[3]!],
  ] as const) {
    const ax = car.x + axle * k * c
    const ay = car.y + axle * k * s
    const x0 = sx(cam, ax)
    const y0 = sy(cam, ay)
    const lon = fx * M_PER_N * k
    const lat = fy * M_PER_N * k
    arrow(ctx, x0, y0, sx(cam, ax + lon * c), sy(cam, ay + lon * s), pal.paint, width)
    arrow(ctx, x0, y0, sx(cam, ax - lat * s), sy(cam, ay + lat * c), pal.sodium, width)
  }
}

/** A sodium ring around the car, pulsing every 1.2 s (static under reduced motion). */
export function drawHalo(
  ctx: CanvasRenderingContext2D,
  car: CarPose,
  cam: Camera,
  pal: Palette,
  k: number,
  dpr: number,
  now: number,
  motion: boolean,
): void {
  const phase = motion ? (now % HALO_PERIOD_MS) / HALO_PERIOD_MS : 0.5
  const pulse = 0.5 - 0.5 * Math.cos(2 * Math.PI * phase)
  const r = (3.2 * k + pulse * 1.2 * k) * cam.scale
  ctx.strokeStyle = withAlpha(pal.sodium, 0.9 - 0.5 * pulse)
  ctx.lineWidth = 2 * dpr
  ctx.beginPath()
  ctx.arc(sx(cam, car.x), sy(cam, car.y), r, 0, 2 * Math.PI)
  ctx.stroke()
}

/** Driver names above each car (Chase camera only, where cars are big enough to label). */
export function drawLabels(
  ctx: CanvasRenderingContext2D,
  cars: CarPose[],
  names: Record<DriverKind, string>,
  cam: Camera,
  pal: Palette,
  dpr: number,
): void {
  ctx.font = `600 ${12 * dpr}px Archivo, system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'bottom'
  for (const car of cars) {
    if (car.alpha < 1) continue
    const x = sx(cam, car.x)
    // Main cars labelled above, ghosts below, so a ghost racing alongside doesn't cover it.
    const y = car.ghost
      ? sy(cam, car.y) + 3.2 * cam.scale + 16 * dpr
      : sy(cam, car.y) - 3.2 * cam.scale
    if (x < -50 || y < -20 || x > ctx.canvas.width + 50 || y > ctx.canvas.height + 20) continue
    ctx.fillStyle = withAlpha(pal.runoff, 0.75)
    const label = car.ghost ? `${names[car.kind]} (ghost)` : names[car.kind]
    const w = ctx.measureText(label).width + 8 * dpr
    ctx.fillRect(x - w / 2, y - 16 * dpr, w, 16 * dpr)
    ctx.fillStyle = pal.paint
    ctx.fillText(label, x, y - 2 * dpr)
  }
}
