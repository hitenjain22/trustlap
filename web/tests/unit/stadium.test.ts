import { readFileSync } from 'node:fs'
import { expect, it } from 'vitest'
import { parseSimConfig } from '../../src/sim/config.ts'
import { stadiumTrack } from '../../src/sim/track.ts'

const cfg = parseSimConfig(
  JSON.parse(
    readFileSync(new URL('../../public/artifacts/v1/config.json', import.meta.url), 'utf8'),
  ),
).track

it('the puddle-test stadium matches evaluate.stadium_bank in Python', () => {
  // Reference values from `stadium_bank()` (sim/trustlap/evaluate.py), float32.
  const t = stadiumTrack(cfg)
  expect(t.lengthM).toBeCloseTo(2028.308, 2)
  expect(t.curvature[175]).toBeCloseTo(0, 5) // bottom straight
  expect(t.curvature[400]).toBeCloseTo(0.010018, 5) // right-hand bend, 1 / 100 m
  expect(t.curvature[600]).toBeCloseTo(0, 5) // top straight
})
