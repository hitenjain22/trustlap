import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  benchHeadline,
  bestFitTrack,
  type Bench,
  type ShowcaseTrack,
} from '../../src/data/artifacts.ts'
import { lerpAngle } from '../../src/render/Renderer.ts'
import { parseSimConfig } from '../../src/sim/config.ts'
import { wrapAngle } from '../../src/sim/dynamics.ts'
import { roundHalfEven } from '../../src/sim/track.ts'

describe('angles', () => {
  it('interpolates on the shortest arc across ±π', () => {
    expect(lerpAngle(3.0, -3.0, 0.5)).toBeCloseTo(Math.PI, 6)
    expect(lerpAngle(0.1, 0.3, 0.5)).toBeCloseTo(0.2, 9)
  })
  it('wraps to (−π, π]', () => {
    expect(wrapAngle(Math.PI)).toBeCloseTo(Math.PI, 12)
    expect(wrapAngle(-Math.PI)).toBeCloseTo(Math.PI, 12)
    expect(wrapAngle(3 * Math.PI + 0.5)).toBeCloseTo(-Math.PI + 0.5, 12)
  })
})

describe('roundHalfEven matches Python round()', () => {
  it.each([
    [0.5, 0],
    [1.5, 2],
    [2.5, 2],
    [2.6, 3],
    [-1.5, -2],
  ])('%f → %i', (x, expected) => expect(roundHalfEven(x)).toBe(expected))
})

describe('benchmark headline', () => {
  const bench = (rows: Bench['rows']): Bench => ({
    created_utc: '2026-10-11T00:00:00+00:00',
    git_sha: 'abc',
    device: 'Tesla T4',
    rows,
  })
  it('compares GPU and CPU at the largest car count measured on both', () => {
    const h = benchHeadline(
      bench([
        { impl: 'jax-gpu', n_cars: 1024, steps_per_s: 4e6 },
        { impl: 'jax-gpu', n_cars: 65536, steps_per_s: 26e6 },
        { impl: 'jax-cpu', n_cars: 1024, steps_per_s: 29e3 },
        { impl: 'numpy-loop', n_cars: 1024, steps_per_s: 554 },
      ]),
    )
    expect(h).toEqual({ maxCars: 65536, speedup: 4e6 / 29e3, device: 'T4' })
  })
  it('returns null (not computed yet) without a CPU row', () => {
    expect(benchHeadline(bench([{ impl: 'jax-gpu', n_cars: 64, steps_per_s: 1 }]))).toBeNull()
  })
})

describe('showcase track fitting', () => {
  const t = (seed: number, aspect: number): ShowcaseTrack => ({
    seed,
    aspect,
    name: `T${seed}`,
    slug: `t${seed}`,
    length_m: 1000,
    width_m: 12,
    min_radius_m: 20,
  })
  it('picks the track whose shape matches the region', () => {
    const tracks = [t(1, 0.6), t(2, 1.0), t(3, 1.4)]
    expect(bestFitTrack(tracks, 1.5).seed).toBe(3)
    expect(bestFitTrack(tracks, 0.7).seed).toBe(1)
  })
})

describe('config schema', () => {
  const raw = JSON.parse(
    readFileSync(new URL('../../public/artifacts/v1/config.json', import.meta.url), 'utf8'),
  )
  it('accepts the exported config', () => {
    expect(parseSimConfig(raw).obs.policy_size).toBe(33)
  })
  it('rejects a broken one, loudly', () => {
    expect(() => parseSimConfig({ ...raw, car: { ...raw.car, mass_kg: 'heavy' } })).toThrow()
    expect(() => parseSimConfig({ ...raw, obs: { ...raw.obs, policy_size: 34 } })).toThrow(/layout/)
  })
})
