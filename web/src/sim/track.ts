/**
 * Procedural tracks, ported line by line from `sim/trustlap/track.py` (SPEC 6.1).
 *
 * Generation is float64 in both languages and parity-tested to 1e-6 m, so the order of
 * floating-point operations below deliberately matches the Python source.
 */
import { PCG32, splitmix32 } from './rng.ts'
import type { SimConfig } from './config.ts'

type TrackConfig = SimConfig['track']

export type TrackFailure = 'too_short' | 'too_long' | 'too_tight' | 'self_intersection' | 'pinch'

/** A generated track in float64. Point arrays are interleaved [x0, y0, x1, y1, …]. */
export interface Track {
  seed: number
  attempt: number
  smoothIters: number
  controlPoints: Float64Array
  centerline: Float64Array
  tangent: Float64Array
  normal: Float64Array
  curvature: Float64Array
  widthM: number
  lengthM: number
  nPoints: number
}

/** A track as the simulator uses it: float32 arrays, exactly like one row of the Python bank. */
export interface TrackData {
  n: number
  lengthM: number
  widthM: number
  dsM: number
  centerline: Float32Array
  tangent: Float32Array
  normal: Float32Array
  curvature: Float32Array
}

/** Python-style modulo (result has the sign of the divisor). */
export function mod(a: number, n: number): number {
  return ((a % n) + n) % n
}

/** Round half to even, like Python's round() and numpy/jax round. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x)
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

// ---------------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------------

/** Random star-shaped control polygon and width for one seed (draw order matches Python). */
export function controlPointsFromSeed(
  seed: number,
  cfg: TrackConfig,
): { points: Float64Array; width: number } {
  const rng = new PCG32(seed)
  const n = rng.nextInt(cfg.n_ctrl_min, cfg.n_ctrl_max)
  const spacing = (2.0 * Math.PI) / n
  const angles: number[] = []
  for (let i = 0; i < n; i++) {
    angles.push(
      (2.0 * Math.PI * i) / n + rng.nextRange(-cfg.angle_jitter, cfg.angle_jitter) * spacing,
    )
  }
  angles.sort((a, b) => a - b)
  const radii: number[] = []
  for (let i = 0; i < n; i++) radii.push(rng.nextRange(cfg.radius_min_m, cfg.radius_max_m))
  for (let i = 0; i < n; i++) {
    if (rng.nextFloat01() < cfg.pull_prob) {
      const r = radii[i]!
      radii[i] = r - rng.nextRange(cfg.pull_min, cfg.pull_max) * r
    }
  }
  const width = rng.nextRange(cfg.width_min_m, cfg.width_max_m)
  const points = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    points[2 * i] = radii[i]! * Math.cos(angles[i]!)
    points[2 * i + 1] = radii[i]! * Math.sin(angles[i]!)
  }
  return { points, width }
}

function knot(ax: number, ay: number, bx: number, by: number, alpha: number): number {
  const dx = bx - ax
  const dy = by - ay
  return Math.sqrt(dx * dx + dy * dy) ** alpha
}

/** Dense samples of a closed centripetal Catmull–Rom spline (Barry–Goldman pyramid). */
export function catmullRomClosed(
  points: Float64Array,
  samplesPerSegment: number,
  alpha: number,
): Float64Array {
  const n = points.length / 2
  const out = new Float64Array(2 * n * samplesPerSegment)
  const px = (i: number) => points[2 * mod(i, n)]!
  const py = (i: number) => points[2 * mod(i, n) + 1]!
  for (let i = 0; i < n; i++) {
    const p0x = px(i - 1), p0y = py(i - 1) // prettier-ignore
    const p1x = px(i), p1y = py(i) // prettier-ignore
    const p2x = px(i + 1), p2y = py(i + 1) // prettier-ignore
    const p3x = px(i + 2), p3y = py(i + 2) // prettier-ignore
    const t0 = 0.0
    const t1 = t0 + knot(p0x, p0y, p1x, p1y, alpha)
    const t2 = t1 + knot(p1x, p1y, p2x, p2y, alpha)
    const t3 = t2 + knot(p2x, p2y, p3x, p3y, alpha)
    for (let k = 0; k < samplesPerSegment; k++) {
      const t = t1 + (t2 - t1) * (k / samplesPerSegment)
      const w0 = (t1 - t) / (t1 - t0), w1 = (t - t0) / (t1 - t0) // prettier-ignore
      const a1x = w0 * p0x + w1 * p1x
      const a1y = w0 * p0y + w1 * p1y
      const w2 = (t2 - t) / (t2 - t1), w3 = (t - t1) / (t2 - t1) // prettier-ignore
      const a2x = w2 * p1x + w3 * p2x
      const a2y = w2 * p1y + w3 * p2y
      const w4 = (t3 - t) / (t3 - t2), w5 = (t - t2) / (t3 - t2) // prettier-ignore
      const a3x = w4 * p2x + w5 * p3x
      const a3y = w4 * p2y + w5 * p3y
      const w6 = (t2 - t) / (t2 - t0), w7 = (t - t0) / (t2 - t0) // prettier-ignore
      const b1x = w6 * a1x + w7 * a2x
      const b1y = w6 * a1y + w7 * a2y
      const w8 = (t3 - t) / (t3 - t1), w9 = (t - t1) / (t3 - t1) // prettier-ignore
      const b2x = w8 * a2x + w9 * a3x
      const b2y = w8 * a2y + w9 * a3y
      const o = 2 * (i * samplesPerSegment + k)
      out[o] = w2 * b1x + w3 * b2x
      out[o + 1] = w2 * b1y + w3 * b2y
    }
  }
  return out
}

/** Resample a closed polyline at uniform arclength: N = round(L / ds), s_k = k · L / N. */
export function resampleClosed(
  dense: Float64Array,
  dsTarget: number,
): { points: Float64Array; length: number } {
  const m = dense.length / 2
  const cum = new Float64Array(m + 1)
  for (let j = 0; j < m; j++) {
    const b = 2 * ((j + 1) % m)
    const dx = dense[b]! - dense[2 * j]!
    const dy = dense[b + 1]! - dense[2 * j + 1]!
    cum[j + 1] = cum[j]! + Math.sqrt(dx * dx + dy * dy)
  }
  const length = cum[m]!
  const n = roundHalfEven(length / dsTarget)
  const ds = length / n
  const out = new Float64Array(2 * n)
  let j = 0
  for (let k = 0; k < n; k++) {
    const s = k * ds
    while (cum[j + 1]! < s) j++
    const a = 2 * j
    const b = 2 * ((j + 1) % m)
    const f = (s - cum[j]!) / (cum[j + 1]! - cum[j]!)
    out[2 * k] = dense[a]! + f * (dense[b]! - dense[a]!)
    out[2 * k + 1] = dense[a + 1]! + f * (dense[b + 1]! - dense[a + 1]!)
  }
  return { points: out, length }
}

/** Tangents (central differences), left normals, smoothed signed curvature. */
export function frames(
  centerline: Float64Array,
  ds: number,
  smoothWindow: number,
): { tangent: Float64Array; normal: Float64Array; curvature: Float64Array } {
  const n = centerline.length / 2
  const tangent = new Float64Array(2 * n)
  const normal = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    const a = 2 * mod(i - 1, n)
    const b = 2 * mod(i + 1, n)
    const dx = centerline[b]! - centerline[a]!
    const dy = centerline[b + 1]! - centerline[a + 1]!
    const norm = Math.sqrt(dx * dx + dy * dy)
    tangent[2 * i] = dx / norm
    tangent[2 * i + 1] = dy / norm
    normal[2 * i] = -tangent[2 * i + 1]!
    normal[2 * i + 1] = tangent[2 * i]!
  }
  const raw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const t0x = tangent[2 * i]!, t0y = tangent[2 * i + 1]! // prettier-ignore
    const j = 2 * ((i + 1) % n)
    const t1x = tangent[j]!, t1y = tangent[j + 1]! // prettier-ignore
    const cross = t0x * t1y - t0y * t1x
    const dot = t0x * t1x + t0y * t1y
    raw[i] = Math.atan2(cross, dot) / ds
  }
  const half = Math.floor(smoothWindow / 2)
  const curvature = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let acc = 0.0
    for (let k = -half; k <= half; k++) acc += raw[mod(i + k, n)]!
    curvature[i] = acc / smoothWindow
  }
  return { tangent, normal, curvature }
}

/** One Laplacian smoothing pass over a closed polygon: p_i += λ · (mean of neighbours − p_i). */
export function laplacianSmooth(points: Float64Array, lambda: number): Float64Array {
  const n = points.length / 2
  const out = new Float64Array(points.length)
  for (let i = 0; i < n; i++) {
    const a = 2 * mod(i - 1, n)
    const b = 2 * ((i + 1) % n)
    out[2 * i] = points[2 * i]! + lambda * ((points[a]! + points[b]!) / 2.0 - points[2 * i]!)
    out[2 * i + 1] =
      points[2 * i + 1]! + lambda * ((points[a + 1]! + points[b + 1]!) / 2.0 - points[2 * i + 1]!)
  }
  return out
}

/** Spline, resample and compute frames for a control polygon. No validation. */
export function buildTrack(
  controlPoints: Float64Array,
  widthM: number,
  seed: number,
  attempt: number,
  smoothIters: number,
  cfg: TrackConfig,
): Track {
  const dense = catmullRomClosed(controlPoints, cfg.samples_per_segment, cfg.catmull_rom_alpha)
  const { points, length } = resampleClosed(dense, cfg.ds_target_m)
  const nPoints = points.length / 2
  const { tangent, normal, curvature } = frames(
    points,
    length / nPoints,
    cfg.curvature_smooth_window,
  )
  return {
    seed,
    attempt,
    smoothIters,
    controlPoints,
    centerline: points,
    tangent,
    normal,
    curvature,
    widthM,
    lengthM: length,
    nPoints,
  }
}

// ---------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------

/** Null if valid, else the first failing reason (cheapest checks first, like Python). */
export function validateTrack(track: Track, cfg: TrackConfig): TrackFailure | null {
  if (track.lengthM < cfg.length_min_m) return 'too_short'
  if (track.lengthM > cfg.length_max_m || track.nPoints > cfg.n_max) return 'too_long'
  let maxK = 0
  for (const k of track.curvature) maxK = Math.max(maxK, Math.abs(k))
  if (maxK > 1.0 / cfg.min_radius_m) return 'too_tight'
  if (!clearanceOk(track, cfg)) return 'pinch'
  if (boundariesIntersect(track, cfg)) return 'self_intersection'
  return null
}

function clearanceOk(track: Track, cfg: TrackConfig): boolean {
  const c = track.centerline
  const n = track.nPoints
  const minD = track.widthM + cfg.clearance_extra_m
  const minD2 = minD * minD
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let gap = Math.abs(i - j)
      gap = Math.min(gap, n - gap)
      if (gap <= cfg.clearance_index_gap) continue
      const dx = c[2 * i]! - c[2 * j]!
      const dy = c[2 * i + 1]! - c[2 * j + 1]!
      if (dx * dx + dy * dy < minD2) return false
    }
  }
  return true
}

/** Left and right boundary polylines: centerline ± W/2 · normal (interleaved xy). */
export function boundaries(track: Track): { left: Float64Array; right: Float64Array } {
  const half = track.widthM / 2.0
  const left = new Float64Array(track.centerline.length)
  const right = new Float64Array(track.centerline.length)
  for (let i = 0; i < track.centerline.length; i++) {
    left[i] = track.centerline[i]! + half * track.normal[i]!
    right[i] = track.centerline[i]! - half * track.normal[i]!
  }
  return { left, right }
}

function stride(poly: Float64Array, step: number): Float64Array {
  const n = poly.length / 2
  const m = Math.ceil(n / step)
  const out = new Float64Array(2 * m)
  for (let k = 0; k < m; k++) {
    out[2 * k] = poly[2 * k * step]!
    out[2 * k + 1] = poly[2 * k * step + 1]!
  }
  return out
}

function orient(px: number, py: number, qx: number, qy: number, rx: number, ry: number): number {
  return (qx - px) * (ry - py) - (qy - py) * (rx - px)
}

function segmentsCross(a: Float64Array, i: number, b: Float64Array, j: number): boolean {
  const na = a.length / 2
  const nb = b.length / 2
  const a0x = a[2 * i]!, a0y = a[2 * i + 1]! // prettier-ignore
  const a1x = a[2 * ((i + 1) % na)]!, a1y = a[2 * ((i + 1) % na) + 1]! // prettier-ignore
  const b0x = b[2 * j]!, b0y = b[2 * j + 1]! // prettier-ignore
  const b1x = b[2 * ((j + 1) % nb)]!, b1y = b[2 * ((j + 1) % nb) + 1]! // prettier-ignore
  const d1 = orient(a0x, a0y, a1x, a1y, b0x, b0y)
  const d2 = orient(a0x, a0y, a1x, a1y, b1x, b1y)
  const d3 = orient(b0x, b0y, b1x, b1y, a0x, a0y)
  const d4 = orient(b0x, b0y, b1x, b1y, a1x, a1y)
  return d1 * d2 < 0 && d3 * d4 < 0
}

/** True if a closed polyline crosses itself (adjacent segments excluded). */
export function closedPolylineSelfIntersects(poly: Float64Array): boolean {
  const m = poly.length / 2
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let gap = Math.abs(i - j)
      gap = Math.min(gap, m - gap)
      if (gap > 1 && segmentsCross(poly, i, poly, j)) return true
    }
  }
  return false
}

function closedPolylinesCross(p: Float64Array, q: Float64Array): boolean {
  for (let i = 0; i < p.length / 2; i++) {
    for (let j = 0; j < q.length / 2; j++) if (segmentsCross(p, i, q, j)) return true
  }
  return false
}

function boundariesIntersect(track: Track, cfg: TrackConfig): boolean {
  const { left, right } = boundaries(track)
  const l = stride(left, cfg.validate_stride)
  const r = stride(right, cfg.validate_stride)
  return (
    closedPolylineSelfIntersects(l) || closedPolylineSelfIntersects(r) || closedPolylinesCross(l, r)
  )
}

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------

/** Build a track, Laplacian-smoothing the control points while the only problem is a tight corner. */
export function buildValidTrack(
  controlPoints: Float64Array,
  widthM: number,
  seed: number,
  attempt: number,
  cfg: TrackConfig,
): { track: Track; reason: TrackFailure | null } {
  let pts = controlPoints
  for (let it = 0; ; it++) {
    const track = buildTrack(pts, widthM, seed, attempt, it, cfg)
    const reason = validateTrack(track, cfg)
    if (reason !== 'too_tight' || it === cfg.autofix_max_iters) return { track, reason }
    pts = laplacianSmooth(pts, cfg.autofix_lambda)
  }
}

/** Generate the valid track for a seed (same result as Python's `generate_track`). */
export function generateTrack(seed: number, cfg: TrackConfig): Track {
  const s = seed >>> 0
  for (let attempt = 0; attempt <= cfg.max_attempts; attempt++) {
    const subSeed = attempt === 0 ? s : splitmix32((s ^ attempt) >>> 0)
    const { points, width } = controlPointsFromSeed(subSeed, cfg)
    const { track, reason } = buildValidTrack(points, width, s, attempt, cfg)
    if (reason === null) return track
  }
  throw new Error(`no valid track for seed ${s} in ${cfg.max_attempts} attempts`)
}

/** Convert to the float32 layout the simulator uses (same rounding as the Python bank). */
export function toTrackData(track: Track): TrackData {
  const lengthM = Math.fround(track.lengthM)
  return {
    n: track.nPoints,
    lengthM,
    widthM: Math.fround(track.widthM),
    dsM: Math.fround(lengthM / track.nPoints),
    centerline: Float32Array.from(track.centerline),
    tangent: Float32Array.from(track.tangent),
    normal: Float32Array.from(track.normal),
    curvature: Float32Array.from(track.curvature),
  }
}

// ---------------------------------------------------------------------------------------------
// Hand-drawn tracks (browser only, SPEC 6.1)
// ---------------------------------------------------------------------------------------------

/** Ramer–Douglas–Peucker simplification of an open polyline (interleaved xy), tolerance ε. */
export function simplifyRdp(points: Float64Array, epsilon: number): Float64Array {
  const n = points.length / 2
  if (n < 3) return points.slice()
  const keep = new Uint8Array(n)
  keep[0] = keep[n - 1] = 1
  const stack: [number, number][] = [[0, n - 1]]
  while (stack.length) {
    const [a, b] = stack.pop()!
    const ax = points[2 * a]!, ay = points[2 * a + 1]! // prettier-ignore
    const bx = points[2 * b]!, by = points[2 * b + 1]! // prettier-ignore
    const dx = bx - ax
    const dy = by - ay
    const len = Math.hypot(dx, dy)
    let worst = -1
    let worstD = 0
    for (let i = a + 1; i < b; i++) {
      const px = points[2 * i]!
      const py = points[2 * i + 1]!
      // A closed stroke starts and ends on the same point: measure from that point instead.
      const d =
        len < 1e-9
          ? Math.hypot(px - ax, py - ay)
          : Math.abs(dy * px - dx * py + bx * ay - by * ax) / len
      if (d > worstD) {
        worstD = d
        worst = i
      }
    }
    if (worst >= 0 && worstD > epsilon) {
      keep[worst] = 1
      stack.push([a, worst], [worst, b])
    }
  }
  const out: number[] = []
  for (let i = 0; i < n; i++) if (keep[i]) out.push(points[2 * i]!, points[2 * i + 1]!)
  return Float64Array.from(out)
}

/** Evenly resample a closed polygon to `count` points (by arclength). */
export function resamplePolygon(points: Float64Array, count: number): Float64Array {
  const n = points.length / 2
  const cum = [0]
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    cum.push(
      cum[i]! +
        Math.hypot(points[2 * j]! - points[2 * i]!, points[2 * j + 1]! - points[2 * i + 1]!),
    )
  }
  const total = cum[n]!
  const out = new Float64Array(2 * count)
  let k = 0
  for (let c = 0; c < count; c++) {
    const s = (c * total) / count
    while (cum[k + 1]! < s) k++
    const f = (s - cum[k]!) / Math.max(1e-9, cum[k + 1]! - cum[k]!)
    const j = (k + 1) % n
    out[2 * c] = points[2 * k]! + f * (points[2 * j]! - points[2 * k]!)
    out[2 * c + 1] = points[2 * k + 1]! + f * (points[2 * j + 1]! - points[2 * k + 1]!)
  }
  return out
}

export interface DrawnTrack {
  controlPoints: Float64Array
  track: Track
  reason: TrackFailure | null
}

/**
 * Turn a hand-drawn closed path (any units, y up) into a track: simplify (RDP, ε = 1.5 m, growing until
 * ≤ 30 control points; at least 8), scale so the spline length lands inside the legal range,
 * then spline + validate. `reason` is null when the track is valid.
 */
export function trackFromDrawing(
  path: Float64Array,
  cfg: TrackConfig,
  widthM = (cfg.width_min_m + cfg.width_max_m) / 2,
): DrawnTrack | null {
  if (path.length / 2 < 8) return null
  // The stroke arrives in screen units: bring it to a nominal size in meters first, so RDP's
  // ε means meters (the exact length is fitted after simplification).
  let strokeM = 0
  for (let i = 1; i < path.length / 2; i++) {
    strokeM += Math.hypot(path[2 * i]! - path[2 * i - 2]!, path[2 * i + 1]! - path[2 * i - 1]!)
  }
  const pre = (cfg.length_min_m + cfg.length_max_m) / 2 / Math.max(strokeM, 1e-9)
  path = path.map((v) => v * pre)
  let eps = 1.5
  let pts = simplifyRdp(path, eps)
  while (pts.length / 2 > 30) {
    eps *= 1.5
    pts = simplifyRdp(path, eps)
  }
  // Drop the closing duplicate (the path ends near its start) and enforce ≥ 8 points.
  const n = pts.length / 2
  if (n > 3 && Math.hypot(pts[0]! - pts[2 * (n - 1)]!, pts[1]! - pts[2 * (n - 1) + 1]!) < 1e-6) {
    pts = pts.slice(0, 2 * (n - 1))
  }
  if (pts.length / 2 < 8) pts = resamplePolygon(pts, 8)
  pts = fitLength(pts, cfg, widthM)
  const track = buildTrack(pts, widthM, -1, 0, 0, cfg)
  if (!Number.isFinite(track.lengthM)) return null // degenerate stroke (e.g. a single dot)
  return { controlPoints: pts, track, reason: validateTrack(track, cfg) }
}

/** Center control points and scale them so the spline length sits comfortably inside [min, max]. */
function fitLength(points: Float64Array, cfg: TrackConfig, widthM: number): Float64Array {
  const pts = points.slice()
  const m = pts.length / 2
  let cx = 0
  let cy = 0
  for (let i = 0; i < m; i++) {
    cx += pts[2 * i]! / m
    cy += pts[2 * i + 1]! / m
  }
  for (let i = 0; i < m; i++) {
    pts[2 * i] = pts[2 * i]! - cx
    pts[2 * i + 1] = pts[2 * i + 1]! - cy
  }
  const lengthOf = (p: Float64Array) => buildTrack(p, widthM, -1, 0, 0, cfg).lengthM
  const target = Math.min(cfg.length_max_m * 0.9, Math.max(cfg.length_min_m * 1.15, lengthOf(pts)))
  for (let iter = 0; iter < 3; iter++) {
    const scale = target / lengthOf(pts)
    if (Math.abs(scale - 1) < 0.01) break
    for (let i = 0; i < pts.length; i++) pts[i] = pts[i]! * scale
  }
  return pts
}

/**
 * SPEC 6.1 Auto-fix: Laplacian-smooth the control points (λ = 0.3) up to 10 times until valid.
 * Smoothing shrinks the loop, so each pass re-fits the length (which also opens corners up).
 */
export function autoFixDrawing(drawn: DrawnTrack, cfg: TrackConfig): DrawnTrack {
  let pts = drawn.controlPoints
  let result = drawn
  for (let it = 0; it < cfg.autofix_max_iters && result.reason !== null; it++) {
    pts = fitLength(laplacianSmooth(pts, cfg.autofix_lambda), cfg, drawn.track.widthM)
    const track = buildTrack(pts, drawn.track.widthM, -1, 0, it + 1, cfg)
    result = { controlPoints: pts, track, reason: validateTrack(track, cfg) }
  }
  return result
}

/** numpy.arange(start, stop, step) for a positive or negative step. */
function arange(start: number, stop: number, step: number): number[] {
  const n = Math.max(0, Math.ceil((stop - start) / step))
  return Array.from({ length: n }, (_, i) => start + i * step)
}

/**
 * The step-response test track, ported from `evaluate.stadium_bank` (sim/trustlap/evaluate.py):
 * two long straights joined by semicircles, sampled every `ds_target_m`. Arclength 0 is the
 * start of the bottom straight, heading +x.
 */
export function stadiumTrack(cfg: TrackConfig, straightM = 700, radiusM = 100, widthM = 12): Track {
  const ds = cfg.ds_target_m
  const pts: number[] = []
  for (const x of arange(0, straightM, ds)) pts.push(x, -radiusM)
  for (const a of arange(-Math.PI / 2, Math.PI / 2, ds / radiusM)) {
    pts.push(straightM + radiusM * Math.cos(a), radiusM * Math.sin(a))
  }
  for (const x of arange(straightM, 0, -ds)) pts.push(x, radiusM)
  for (const a of arange(Math.PI / 2, (3 * Math.PI) / 2, ds / radiusM)) {
    pts.push(radiusM * Math.cos(a), radiusM * Math.sin(a))
  }
  const centerline = Float64Array.from(pts)
  const n = centerline.length / 2
  let length = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    length += Math.hypot(
      centerline[2 * j]! - centerline[2 * i]!,
      centerline[2 * j + 1]! - centerline[2 * i + 1]!,
    )
  }
  const { tangent, normal, curvature } = frames(centerline, length / n, cfg.curvature_smooth_window)
  return {
    seed: 0,
    attempt: 0,
    smoothIters: 0,
    controlPoints: new Float64Array(0),
    centerline,
    tangent,
    normal,
    curvature,
    widthM,
    lengthM: length,
    nPoints: n,
  }
}
