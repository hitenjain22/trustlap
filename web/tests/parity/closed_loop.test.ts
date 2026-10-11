import { expect, it } from 'vitest'
import { coreStep, createEnvState, createStepInfo } from '../../src/sim/env.ts'
import { scriptedAction } from '../../src/sim/scripted.ts'
import { condFrom, goldenTracks, loadConfig, loadGolden } from './helpers.ts'
import { resetEnvState, placePuddles } from '../../src/sim/env.ts'

it('scripted closed loop: 10 laps-in-progress × 400 steps (progress within 2%, crash flags agree)', () => {
  const cfg = loadConfig()
  const g = loadGolden('scripted_closed_loop')
  const tracks = goldenTracks(g.track_seeds, cfg)
  const action = new Float32Array(2)
  const info = createStepInfo()
  const report: string[] = []
  let flagAgree = 0
  for (const c of g.cases) {
    const track = tracks[c.track_index]!
    const cond = condFrom(
      Object.fromEntries(Object.entries(c.conditions).map(([k, v]) => [k, [v]])),
      0,
    )
    const state = createEnvState(cfg, cond)
    resetEnvState(state, track, 0, cfg.episode.eval_speed_mps)
    placePuddles(state.cond, track, 0)
    let crashed = false
    const progress: number[] = []
    for (let step = 1; step <= g.steps; step++) {
      if (!crashed) {
        scriptedAction(state, track, cfg, action)
        coreStep(state, action[0]!, action[1]!, 0, 0, track, cfg, info)
        crashed = info.terminated && !info.lapCompleted
      }
      if ((g.checkpoints as number[]).includes(step)) progress.push(state.progressM)
    }
    if (crashed === c.crashed) flagAgree++
    c.progress_m.forEach((p: number, k: number) => {
      const rel = Math.abs(p - progress[k]!) / Math.max(Math.abs(p), 1)
      if (rel > 0.02)
        report.push(
          `${c.suite} track ${c.track_index} step ${g.checkpoints[k]}: ${p} vs ${progress[k]}`,
        )
    })
  }
  expect(report).toEqual([])
  expect(flagAgree).toBeGreaterThanOrEqual(Math.ceil(0.9 * g.cases.length))
})
