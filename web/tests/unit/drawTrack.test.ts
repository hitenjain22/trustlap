import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { parseSimConfig } from '../../src/sim/config.ts'
import { autoFixDrawing, simplifyRdp, trackFromDrawing } from '../../src/sim/track.ts'

const cfg = parseSimConfig(
  JSON.parse(
    readFileSync(new URL('../../public/artifacts/v1/config.json', import.meta.url), 'utf8'),
  ),
).track

/** A hand-drawn-ish loop: an ellipse with wobble, many points like a pointer trace. */
function ellipse(rx: number, ry: number, n = 300, wobble = 0): Float64Array {
  const out: number[] = []
  for (let i = 0; i <= n; i++) {
    const a = (2 * Math.PI * i) / n
    out.push(rx * Math.cos(a) + wobble * Math.sin(7 * a), ry * Math.sin(a))
  }
  return Float64Array.from(out)
}

describe('drawn tracks', () => {
  it('RDP keeps the corners of a polyline', () => {
    const line = Float64Array.from([0, 0, 1, 0.01, 2, 0, 2, 1, 2, 2])
    expect(Array.from(simplifyRdp(line, 0.1))).toEqual([0, 0, 2, 0, 2, 2])
  })
  it('a smooth loop becomes a valid track scaled into the legal length', () => {
    const d = trackFromDrawing(ellipse(80, 50), cfg)!
    expect(d.reason).toBeNull()
    expect(d.track.lengthM).toBeGreaterThanOrEqual(cfg.length_min_m)
    expect(d.track.lengthM).toBeLessThanOrEqual(cfg.length_max_m)
    const n = d.controlPoints.length / 2
    expect(n).toBeGreaterThanOrEqual(8)
    expect(n).toBeLessThanOrEqual(30)
  })
  it('a stroke that ends exactly on its start point still simplifies', () => {
    const e = ellipse(80, 50)
    e[e.length - 2] = e[0]!
    e[e.length - 1] = e[1]!
    const d = trackFromDrawing(e, cfg)!
    expect(Number.isFinite(d.track.lengthM)).toBe(true)
    expect(d.reason).toBeNull()
  })
  it('a degenerate stroke is rejected rather than producing NaN', () => {
    expect(trackFromDrawing(new Float64Array(40).fill(5), cfg)).toBeNull()
  })
  it('a figure eight is rejected with a specific reason', () => {
    const out: number[] = []
    for (let i = 0; i < 300; i++) {
      const t = (2 * Math.PI * i) / 300
      out.push(300 * Math.sin(t), 150 * Math.sin(t) * Math.cos(t))
    }
    out.push(0, 0) // ends exactly where it started, like a real pointer
    const d = trackFromDrawing(Float64Array.from(out), cfg)!
    expect(['self_intersection', 'pinch']).toContain(d.reason)
  })
  it('auto-fix smooths a too-tight wiggly loop into a valid one', () => {
    const d = trackFromDrawing(ellipse(90, 60, 300, 12), cfg)!
    expect(d.reason).not.toBeNull()
    const fixed = autoFixDrawing(d, cfg)
    expect(fixed.reason).toBeNull()
  })
})
