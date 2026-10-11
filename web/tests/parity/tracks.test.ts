import { describe, expect, it } from 'vitest'
import { generateTrack } from '../../src/sim/track.ts'
import { loadConfig, loadGolden, Tolerance } from './helpers.ts'

describe('track parity (1e-6 m, float64 both sides)', () => {
  const cfg = loadConfig()
  const golden = loadGolden('tracks')
  for (const c of golden.cases) {
    it(`seed ${c.seed}`, () => {
      const t = generateTrack(c.seed, cfg.track)
      expect({ n: t.nPoints, attempt: t.attempt, smooth: t.smoothIters }).toEqual({
        n: c.n_points,
        attempt: c.attempt,
        smooth: c.smooth_iters,
      })
      const tol = new Tolerance(1e-6, 0)
      tol.check('length', c.length_m, t.lengthM)
      tol.check('width', c.width_m, t.widthM)
      c.every10_centerline.forEach(([x, y]: number[], k: number) => {
        tol.check(`centerline[${10 * k}].x`, x!, t.centerline[20 * k]!)
        tol.check(`centerline[${10 * k}].y`, y!, t.centerline[20 * k + 1]!)
      })
      c.every10_tangent.forEach(([x, y]: number[], k: number) => {
        tol.check(`tangent[${10 * k}].x`, x!, t.tangent[20 * k]!)
        tol.check(`tangent[${10 * k}].y`, y!, t.tangent[20 * k + 1]!)
      })
      c.every10_curvature.forEach((v: number, k: number) =>
        tol.check(`curvature[${10 * k}]`, v, t.curvature[10 * k]!),
      )
      expect(tol.failures).toEqual([])
    })
  }
})
