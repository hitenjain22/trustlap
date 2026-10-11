import { describe, expect, it } from 'vitest'
import { PCG32 } from '../../src/sim/rng.ts'
import { loadGolden } from './helpers.ts'

describe('rng parity (exact)', () => {
  const golden = loadGolden('rng')
  for (const c of golden.cases) {
    it(`seed ${c.seed}: first 1,000 outputs match`, () => {
      const rng = new PCG32(c.seed, golden.stream)
      const out = Array.from({ length: c.u32.length }, () => rng.nextU32())
      expect(out).toEqual(c.u32)
    })
  }
})
