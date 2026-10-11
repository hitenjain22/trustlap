/**
 * Cars (SPEC 11.7, layer 7): 4.4 × 1.9 m bodies in the driver's identity color, a windshield
 * highlight, and soft headlight cones in the rain. Scripted cars are outline-only (SPEC 11.3).
 */
import { sx, sy, type Camera } from '../camera.ts'
import { withAlpha, type Palette } from '../palette.ts'

export type DriverKind = 'student' | 'teacher' | 'blind' | 'scripted'

const LENGTH_M = 4.4
const WIDTH_M = 1.9
/**
 * At whole-track zoom a 4.4 m car is only a few pixels long, so bodies are drawn at least this
 * long on screen (CSS px). Purely visual, like the dots on a broadcast track map.
 */
const MIN_CAR_PX = 13

/** How much to enlarge car bodies so they stay visible at this zoom (1 = true size). */
export function carScale(cam: Camera, dpr: number): number {
  return Math.max(1, (MIN_CAR_PX * dpr) / (LENGTH_M * cam.scale))
}

export interface CarPose {
  x: number
  y: number
  yaw: number
  alpha: number // 1 = normal, < 1 fading after a crash
}

let cone: HTMLCanvasElement | null = null
let coneColor = ''

/** Headlight cone sprite: a sodium radial gradient, pointing +x, origin at the left middle. */
function coneSprite(color: string): HTMLCanvasElement {
  if (cone && coneColor === color) return cone
  cone = document.createElement('canvas')
  cone.width = 128
  cone.height = 96
  const ctx = cone.getContext('2d')!
  const g = ctx.createRadialGradient(0, 48, 0, 0, 48, 128)
  g.addColorStop(0, withAlpha(color, 0.2))
  g.addColorStop(1, withAlpha(color, 0))
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.moveTo(0, 42)
  ctx.lineTo(128, 0)
  ctx.lineTo(128, 96)
  ctx.lineTo(0, 54)
  ctx.closePath()
  ctx.fill()
  coneColor = color
  return cone
}

function corner(cam: Camera, p: CarPose, along: number, across: number): [number, number] {
  const c = Math.cos(p.yaw)
  const s = Math.sin(p.yaw)
  return [sx(cam, p.x + along * c - across * s), sy(cam, p.y + along * s + across * c)]
}

function bodyPath(ctx: CanvasRenderingContext2D, cam: Camera, p: CarPose, scale = 1): void {
  const hl = (LENGTH_M / 2) * scale
  const hw = (WIDTH_M / 2) * scale
  const pts = [
    corner(cam, p, hl, hw),
    corner(cam, p, hl, -hw),
    corner(cam, p, -hl, -hw),
    corner(cam, p, -hl, hw),
  ]
  ctx.moveTo(pts[0]![0], pts[0]![1])
  for (let k = 1; k < 4; k++) ctx.lineTo(pts[k]![0], pts[k]![1])
  ctx.closePath()
}

export function drawHeadlights(
  ctx: CanvasRenderingContext2D,
  cars: CarPose[],
  cam: Camera,
  pal: Palette,
  k: number,
): void {
  const sprite = coneSprite(pal.sodium)
  const len = 14 * cam.scale * k
  const wid = 10 * cam.scale * k
  for (const p of cars) {
    if (p.alpha < 1) continue
    const [fx, fy] = corner(cam, p, (LENGTH_M / 2) * k, 0)
    ctx.setTransform(
      Math.cos(-p.yaw),
      Math.sin(-p.yaw),
      -Math.sin(-p.yaw),
      Math.cos(-p.yaw),
      fx,
      fy,
    )
    ctx.drawImage(sprite, 0, -wid / 2, len, wid)
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0)
}

export function drawCars(
  ctx: CanvasRenderingContext2D,
  cars: CarPose[],
  kind: DriverKind,
  cam: Camera,
  pal: Palette,
  k: number,
): void {
  const color = {
    student: pal.drvStudent,
    teacher: pal.drvTeacher,
    blind: pal.drvBlind,
    scripted: pal.drvScripted,
  }[kind]
  const outlineOnly = kind === 'scripted'
  for (const fading of [false, true]) {
    ctx.beginPath()
    let any = false
    for (const p of cars) {
      if (p.alpha < 1 !== fading) continue
      bodyPath(ctx, cam, p, k)
      any = true
    }
    if (!any) continue
    ctx.globalAlpha = fading ? 0.35 : 1
    if (outlineOnly) {
      ctx.fillStyle = withAlpha(color, 0.18)
      ctx.fill()
      ctx.strokeStyle = color
      ctx.lineWidth = Math.max(0.35 * cam.scale * k, 1.25)
      ctx.stroke()
    } else {
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)'
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }
  // Windshield highlight: a lighter band just ahead of the center.
  ctx.globalAlpha = 1
  ctx.beginPath()
  for (const p of cars) {
    if (p.alpha < 1) continue
    const a = corner(cam, p, 0.95 * k, 0.7 * k)
    const b = corner(cam, p, 0.95 * k, -0.7 * k)
    const c = corner(cam, p, 0.25 * k, -0.75 * k)
    const d = corner(cam, p, 0.25 * k, 0.75 * k)
    ctx.moveTo(a[0], a[1])
    ctx.lineTo(b[0], b[1])
    ctx.lineTo(c[0], c[1])
    ctx.lineTo(d[0], d[1])
    ctx.closePath()
  }
  ctx.fillStyle = outlineOnly ? withAlpha(color, 0.55) : 'rgba(233, 237, 240, 0.35)'
  ctx.fill()
}
