/**
 * Cars (SPEC 11.7 layer 7, 11.3): 4.4 × 1.9 m bodies in the driver's identity color, a
 * windshield highlight, and soft headlight cones in the rain. Scripted cars are outline-only.
 * Ghosts are drawn at 45% opacity with their driver's dash pattern; the focused car gets a
 * 2 px paint outline.
 */
import type { DriverKind } from '../../worker/protocol.ts'
import { sx, sy, type Camera } from '../camera.ts'
import { withAlpha, type Palette } from '../palette.ts'

export type { DriverKind }

const LENGTH_M = 4.4
const WIDTH_M = 1.9
/**
 * At whole-track zoom a 4.4 m car is only a few pixels long, so bodies are drawn at least this
 * long on screen (CSS px). Purely visual, like the dots on a broadcast track map.
 */
const MIN_CAR_PX = 13

/** Dash patterns per driver (SPEC 11.3), in CSS px. */
export const DRIVER_DASH: Record<DriverKind, number[]> = {
  human: [],
  student: [],
  teacher: [2, 4],
  blind: [6, 4],
  scripted: [10, 4],
}

/** How much to enlarge car bodies so they stay visible at this zoom (1 = true size). */
export function carScale(cam: Camera, dpr: number): number {
  return Math.max(1, (MIN_CAR_PX * dpr) / (LENGTH_M * cam.scale))
}

export interface CarPose {
  x: number
  y: number
  yaw: number
  alpha: number // 1 = normal, < 1 fading after a crash
  kind: DriverKind
  ghost: boolean
  focused: boolean
}

export function driverColor(kind: DriverKind, pal: Palette): string {
  return {
    student: pal.drvStudent,
    teacher: pal.drvTeacher,
    blind: pal.drvBlind,
    scripted: pal.drvScripted,
    human: pal.human,
  }[kind]
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

function bodyPath(ctx: CanvasRenderingContext2D, cam: Camera, p: CarPose, k: number): void {
  const hl = (LENGTH_M / 2) * k
  const hw = (WIDTH_M / 2) * k
  const a = corner(cam, p, hl, hw)
  const b = corner(cam, p, hl, -hw)
  const c = corner(cam, p, -hl, -hw)
  const d = corner(cam, p, -hl, hw)
  ctx.moveTo(a[0], a[1])
  ctx.lineTo(b[0], b[1])
  ctx.lineTo(c[0], c[1])
  ctx.lineTo(d[0], d[1])
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
    if (p.alpha < 1 || p.ghost) continue
    const [fx, fy] = corner(cam, p, (LENGTH_M / 2) * k, 0)
    const c = Math.cos(-p.yaw)
    const s = Math.sin(-p.yaw)
    ctx.setTransform(c, s, -s, c, fx, fy)
    ctx.drawImage(sprite, 0, -wid / 2, len, wid)
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0)
}

/** Draw one group of same-looking cars as one batched path. */
function drawGroup(
  ctx: CanvasRenderingContext2D,
  cars: CarPose[],
  kind: DriverKind,
  ghost: boolean,
  cam: Camera,
  pal: Palette,
  k: number,
  dpr: number,
): void {
  if (cars.length === 0) return
  const color = driverColor(kind, pal)
  const outlineOnly = kind === 'scripted'
  for (const fading of [false, true]) {
    const group = cars.filter((p) => p.alpha < 1 === fading)
    if (group.length === 0) continue
    ctx.beginPath()
    for (const p of group) bodyPath(ctx, cam, p, k)
    ctx.globalAlpha = (fading ? 0.35 : 1) * (ghost ? 0.45 : 1)
    if (outlineOnly || ghost) {
      ctx.fillStyle = withAlpha(color, outlineOnly ? 0.18 : 0.35)
      ctx.fill()
      ctx.setLineDash(ghost ? DRIVER_DASH[kind].map((v) => v * dpr * 0.6) : [])
      ctx.strokeStyle = color
      ctx.lineWidth = Math.max(0.35 * cam.scale * k, 1.25 * dpr)
      ctx.stroke()
      ctx.setLineDash([])
    } else {
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)'
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }
  ctx.globalAlpha = 1
  if (ghost) return
  // Windshield highlight: a lighter band just ahead of the center.
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

const KINDS: DriverKind[] = ['scripted', 'blind', 'teacher', 'student', 'human']

/** Ghosts first (underneath), then main cars by driver, then the focus outline on top. */
export function drawCars(
  ctx: CanvasRenderingContext2D,
  cars: CarPose[],
  cam: Camera,
  pal: Palette,
  k: number,
  dpr: number,
  showFocus: boolean,
): void {
  for (const ghost of [true, false]) {
    for (const kind of KINDS) {
      const group = cars.filter((p) => p.ghost === ghost && p.kind === kind)
      drawGroup(ctx, group, kind, ghost, cam, pal, k, dpr)
    }
  }
  if (!showFocus) return
  const focused = cars.find((p) => p.focused && p.alpha === 1)
  if (!focused) return
  ctx.beginPath()
  bodyPath(ctx, cam, focused, k * 1.25)
  ctx.strokeStyle = pal.paint
  ctx.lineWidth = 2 * dpr
  ctx.stroke()
}
