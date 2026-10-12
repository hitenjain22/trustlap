import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { parseSimConfig } from '../../src/sim/config.ts'
import type { EnvState } from '../../src/sim/env.ts'
import { riskFeatures, riskProbability, type RiskModel } from '../../src/sim/risk.ts'
import { generateTrack, toTrackData } from '../../src/sim/track.ts'

const cfg = parseSimConfig(
  JSON.parse(
    readFileSync(new URL('../../public/artifacts/v1/config.json', import.meta.url), 'utf8'),
  ),
)
const golden = JSON.parse(readFileSync(new URL('./golden/risk.json', import.meta.url), 'utf8')) as {
  track_seeds: number[]
  model: RiskModel
  cases: { f: number[]; p: number }[]
  states: {
    track: number
    idx: number
    vx: number
    vy: number
    u: number
    mu: number
    f: number[]
  }[]
}

describe('crash-risk model parity with Python', () => {
  it('gives the same probability for the same features', () => {
    for (const c of golden.cases) {
      expect(riskProbability({ model: golden.model }, c.f)).toBeCloseTo(c.p, 5)
    }
  })
  it('computes the same features from car states on the golden tracks', () => {
    const tracks = golden.track_seeds.map((s) => toTrackData(generateTrack(s, cfg.track)))
    for (const s of golden.states) {
      const state = {
        car: { vxMps: Math.fround(s.vx), vyMps: Math.fround(s.vy) },
        idx: s.idx,
      } as unknown as EnvState
      const f = riskFeatures(state, tracks[s.track]!, Math.fround(s.u), Math.fround(s.mu), cfg)
      f.forEach((v, i) => expect(v).toBeCloseTo(s.f[i]!, 3))
    }
  })
})
