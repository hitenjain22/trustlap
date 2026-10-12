import { describe, expect, it } from 'vitest'
import { bestAiLaps, biggest, compareConditions } from '../../src/data/summary.ts'
import type { Metrics } from '../../src/data/schemas.ts'

const est = (value: number) => ({ value, ci: [value, value] as [number, number] })
const suite = (s: number, b: number, t = 0) => ({
  student: { completion: est(s) },
  blind: { completion: est(b) },
  teacher: { completion: est(t) },
})
const delta = (s: string, lo: number, hi: number) => ({
  suite: s,
  a: 'student',
  b: 'blind',
  metric: 'completion',
  value: (lo + hi) / 2,
  ci: [lo, hi] as [number, number],
})
const m = {
  schema_version: 1,
  generated_utc: '',
  suites: {
    dry: suite(0.64, 0.97),
    wet: suite(0.39, 0.04),
    heavy: suite(0.77, 0.82),
    ice: suite(0, 0, 0.01),
  },
  paired_deltas: [delta('dry', -0.39, -0.26), delta('wet', 0.31, 0.4), delta('heavy', -0.13, 0.01)],
} as unknown as Metrics

describe('plain-language summary', () => {
  const rows = compareConditions(m, ['dry', 'wet', 'heavy', 'missing'])
  it('uses the paired interval for the verdict and skips missing conditions', () => {
    expect(rows.map((r) => [r.suite, r.verdict])).toEqual([
      ['dry', 'worse'],
      ['wet', 'better'],
      ['heavy', 'same'],
    ])
    expect(rows[0]).toMatchObject({ adaptive: 64, basic: 97 })
  })
  it('picks the biggest clear gain and loss', () => {
    expect(biggest(rows, 'better')?.suite).toBe('wet')
    expect(biggest(rows, 'worse')?.suite).toBe('dry')
  })
  it('reports the best any AI did outside training', () => {
    expect(bestAiLaps(m, 'ice')).toBe(1)
    expect(bestAiLaps(m, 'nope')).toBeNull()
  })
})

describe('replication', () => {
  it('attaches how many training runs agreed with each verdict', () => {
    const rep = {
      ...m,
      replication: {
        seeds: [1, 2, 3],
        primary_seed: 1,
        agreement: {
          wet: { primary: 'better', same_in: 3, of: 3 },
          dry: { primary: 'worse', same_in: 2, of: 3 },
        },
      },
    } as unknown as Metrics
    const rows = compareConditions(rep, ['dry', 'wet', 'heavy'])
    expect(rows.find((r) => r.suite === 'wet')?.heldIn).toEqual({ same: 3, of: 3 })
    expect(rows.find((r) => r.suite === 'dry')?.heldIn).toEqual({ same: 2, of: 3 })
    expect(rows.find((r) => r.suite === 'heavy')?.heldIn).toBeUndefined()
  })
})
