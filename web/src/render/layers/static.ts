/**
 * Static layers, drawn once per track and canvas size into an offscreen canvas (SPEC 11.7,
 * layers 1–3): runoff with a speckle texture, the track surface, edge paint, corner kerbs and
 * a checkered start/finish line.
 */
import type { TrackGeometry } from '../../worker/protocol.ts'
import { sx, sy, type Camera } from '../camera.ts'
import { withAlpha, type Palette } from '../palette.ts'

const KERB_RADIUS_M = 60
const KERB_WIDTH_M = 0.9
const EDGE_WIDTH_M = 0.25

let noiseTile: HTMLCanvasElement | null = null

/** 256 px tile of random 1 px speckles at 4% opacity (aggregate texture). */
function speckleTile(paint: string): HTMLCanvasElement {
  if (noiseTile) return noiseTile
  const tile = document.createElement('canvas')
  tile.width = tile.height = 256
  const ctx = tile.getContext('2d')!
  let seed = 1234567
  const rand = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296
  ctx.fillStyle = withAlpha(paint, 0.04)
  for (let i = 0; i < 2200; i++)
    ctx.fillRect(Math.floor(rand() * 256), Math.floor(rand() * 256), 1, 1)
  noiseTile = tile
  return tile
}

/** Closed path of the track surface: left edge forward, right edge backward. */
export function trackPath(geo: TrackGeometry, cam: Camera, offsetM = 0): Path2D {
  const path = new Path2D()
  const half = geo.widthM / 2 + offsetM
  const c = geo.centerline
  const nrm = geo.normal
  for (let i = 0; i <= geo.n; i++) {
    const k = 2 * (i % geo.n)
    const x = sx(cam, c[k]! + half * nrm[k]!)
    const y = sy(cam, c[k + 1]! + half * nrm[k + 1]!)
    if (i === 0) path.moveTo(x, y)
    else path.lineTo(x, y)
  }
  for (let i = geo.n; i >= 0; i--) {
    const k = 2 * (i % geo.n)
    path.lineTo(sx(cam, c[k]! - half * nrm[k]!), sy(cam, c[k + 1]! - half * nrm[k + 1]!))
  }
  path.closePath()
  return path
}

/** One edge polyline (side = +1 left, −1 right) at `offsetM` from the centerline. */
export function edgePath(geo: TrackGeometry, cam: Camera, side: 1 | -1, offsetM: number): Path2D {
  const path = new Path2D()
  const c = geo.centerline
  const nrm = geo.normal
  for (let i = 0; i <= geo.n; i++) {
    const k = 2 * (i % geo.n)
    const x = sx(cam, c[k]! + side * offsetM * nrm[k]!)
    const y = sy(cam, c[k + 1]! + side * offsetM * nrm[k + 1]!)
    if (i === 0) path.moveTo(x, y)
    else path.lineTo(x, y)
  }
  return path
}

export function drawStatic(
  ctx: CanvasRenderingContext2D,
  geo: TrackGeometry,
  cam: Camera,
  pal: Palette,
  w: number,
  h: number,
): void {
  // 1. Runoff.
  ctx.fillStyle = pal.runoff
  ctx.fillRect(0, 0, w, h)
  const pattern = ctx.createPattern(speckleTile(pal.paint), 'repeat')
  if (pattern) {
    ctx.fillStyle = pattern
    ctx.fillRect(0, 0, w, h)
  }

  // 2. Track surface.
  ctx.fillStyle = pal.track
  ctx.fill(trackPath(geo, cam))

  // 3a. Kerbs on tight corners, on the inside (apex) of the turn, alternating red / white.
  const half = geo.widthM / 2
  const c = geo.centerline
  const nrm = geo.normal
  // Batched into one path per color: far fewer fill calls when the layer is rebuilt.
  const kerbs = [new Path2D(), new Path2D()]
  const quad = (path: Path2D, i: number, side: number, inner: number, outer: number) => {
    const a = 2 * i
    const b = 2 * ((i + 1) % geo.n)
    path.moveTo(
      sx(cam, c[a]! + side * inner * nrm[a]!),
      sy(cam, c[a + 1]! + side * inner * nrm[a + 1]!),
    )
    path.lineTo(
      sx(cam, c[b]! + side * inner * nrm[b]!),
      sy(cam, c[b + 1]! + side * inner * nrm[b + 1]!),
    )
    path.lineTo(
      sx(cam, c[b]! + side * outer * nrm[b]!),
      sy(cam, c[b + 1]! + side * outer * nrm[b + 1]!),
    )
    path.lineTo(
      sx(cam, c[a]! + side * outer * nrm[a]!),
      sy(cam, c[a + 1]! + side * outer * nrm[a + 1]!),
    )
    path.closePath()
  }
  for (let i = 0; i < geo.n; i++) {
    const k = geo.curvature[i]!
    if (Math.abs(k) < 1 / KERB_RADIUS_M) continue
    quad(kerbs[i % 2]!, i, Math.sign(k), half, half + KERB_WIDTH_M)
  }
  ctx.fillStyle = pal.kerbRed
  ctx.fill(kerbs[0]!)
  ctx.fillStyle = pal.paint
  ctx.fill(kerbs[1]!)

  // 3b. Edge paint.
  ctx.strokeStyle = pal.paint
  ctx.lineWidth = Math.max(EDGE_WIDTH_M * cam.scale, 1)
  ctx.lineJoin = 'round'
  ctx.stroke(edgePath(geo, cam, 1, half))
  ctx.stroke(edgePath(geo, cam, -1, half))

  // 3c. Checkered start/finish line at index 0, two rows across the width.
  const tx = -nrm[1]!
  const ty = nrm[0]!
  const squares = 8
  const cell = geo.widthM / squares
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < squares; col++) {
      ctx.fillStyle = (row + col) % 2 === 0 ? pal.paint : pal.runoff
      const lat0 = -half + col * cell
      const lon0 = row * cell
      const pts = [
        [lat0, lon0],
        [lat0 + cell, lon0],
        [lat0 + cell, lon0 + cell],
        [lat0, lon0 + cell],
      ] as const
      ctx.beginPath()
      pts.forEach(([lat, lon], j) => {
        const x = sx(cam, c[0]! + lat * nrm[0]! + lon * tx)
        const y = sy(cam, c[1]! + lat * nrm[1]! + lon * ty)
        if (j === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.closePath()
      ctx.fill()
    }
  }
}
