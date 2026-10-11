import { describe, expect, it } from 'vitest'
import { coreStep, createObservation, createStepInfo, observe } from '../../src/sim/env.ts'
import { scriptedAction } from '../../src/sim/scripted.ts'
import { envStateFrom, goldenTracks, loadConfig, loadGolden, Tolerance } from './helpers.ts'

const cfg = loadConfig()

describe('env parity', () => {
  it('observations: 1,000 random states (no noise)', () => {
    const g = loadGolden('obs')
    const tracks = goldenTracks(g.track_seeds, cfg)
    const tol = new Tolerance()
    const out = createObservation(cfg)
    const ePsi = cfg.obs_layout.policy.indexOf('e_psi')
    for (let i = 0; i < g.n; i++) {
      const state = envStateFrom(g.state, i, cfg, g.history_cases)
      observe(state, tracks[g.state.track_id[i]]!, null, cfg, out)
      g.obs.policy[i].forEach((v: number, k: number) =>
        // e_ψ/π can sit on either side of ±1 when the heading is near-reversed.
        tol.check(
          `${i}.policy.${cfg.obs_layout.policy[k]}`,
          k === ePsi ? v * Math.PI : v,
          k === ePsi ? out.policy[k]! * Math.PI : out.policy[k]!,
          k === ePsi,
        ),
      )
      g.obs.privileged[i].forEach((v: number, k: number) =>
        tol.check(`${i}.privileged.${cfg.obs_layout.privileged[k]}`, v, out.privileged[k]!),
      )
      if (i < g.history_cases) {
        g.obs.history[i]
          .flat()
          .forEach((v: number, k: number) => tol.check(`${i}.history[${k}]`, v, state.history[k]!))
      }
    }
    expect(tol.failures).toEqual([])
  })

  it('core step: 1,000 random steps (progress, reward, termination)', () => {
    const g = loadGolden('env_step')
    const tracks = goldenTracks(g.track_seeds, cfg)
    const tol = new Tolerance()
    const info = createStepInfo()
    let flagMismatches = 0
    for (let i = 0; i < g.n; i++) {
      const state = envStateFrom(g.state, i, cfg, g.history_cases)
      const [aSteer, aLong] = g.action[i]
      const [nx, ny] = g.gust_noise[i]
      const reward = coreStep(state, aSteer, aLong, nx, ny, tracks[g.state.track_id[i]]!, cfg, info)
      const e = g.next_state
      tol.check(`${i}.s`, e.s_m[i], state.sM)
      tol.check(`${i}.progress`, e.progress_m[i], state.progressM)
      tol.check(`${i}.reward`, g.reward[i], reward)
      tol.check(`${i}.x`, e.car.x_m[i], state.car.xM)
      tol.check(`${i}.vx`, e.car.vx_mps[i], state.car.vxMps)
      tol.check(`${i}.mu_front`, g.info.mu_front[i], info.muFront)
      if (e.idx[i] !== state.idx) flagMismatches++
      for (const flag of [
        'lap_completed',
        'crashed',
        'stalled',
        'wrong_way',
        'terminated',
        'truncated',
      ] as const) {
        const ts = {
          lap_completed: info.lapCompleted,
          crashed: info.crashed,
          stalled: info.stalled,
          wrong_way: info.wrongWay,
          terminated: info.terminated,
          truncated: info.truncated,
        }[flag]
        if (g.info[flag][i] !== ts) flagMismatches++
      }
      if (i < g.history_cases) {
        e.history[i]
          .flat()
          .forEach((v: number, k: number) => tol.check(`${i}.history[${k}]`, v, state.history[k]!))
      }
    }
    expect(tol.failures).toEqual([])
    expect(flagMismatches).toBe(0)
  })

  it('scripted driver: 500 random states', () => {
    const g = loadGolden('scripted')
    const tracks = goldenTracks(g.track_seeds, cfg)
    const tol = new Tolerance()
    const action = new Float32Array(2)
    for (let i = 0; i < g.n; i++) {
      const state = envStateFrom(g.state, i, cfg, g.history_cases)
      scriptedAction(state, tracks[g.state.track_id[i]]!, cfg, action)
      tol.check(`${i}.steer`, g.action[i][0], action[0]!)
      tol.check(`${i}.long`, g.action[i][1], action[1]!)
    }
    expect(tol.failures).toEqual([])
  })
})
