import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { ActorPolicy } from '../../src/policy/actor.ts'
import { StudentEnsemble } from '../../src/policy/ensemble.ts'
import { parseNetwork } from '../../src/policy/loader.ts'
import {
  coreStep,
  createEnvState,
  createObservation,
  createStepInfo,
  observe,
  placePuddles,
  resetEnvState,
} from '../../src/sim/env.ts'
import { condFrom, goldenTracks, loadConfig, loadGolden, Tolerance } from './helpers.ts'

const artifact = (p: string) =>
  fileURLToPath(new URL(`../../public/artifacts/v1/${p}`, import.meta.url))

async function loadNet(name: string) {
  const json = JSON.parse(readFileSync(artifact(`policies/${name}.json`), 'utf8'))
  const buf = readFileSync(artifact(`policies/${json.bin}`))
  const bin = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
  return parseNetwork(json, bin)
}

async function loadPolicy(name: string): Promise<ActorPolicy> {
  return new ActorPolicy(await loadNet(name))
}

const hasArtifact = (name: string) => existsSync(artifact(`policies/${name}.json`))

const cfg = loadConfig()

describe('policy parity', () => {
  for (const name of ['blind', 'teacher'].filter(hasArtifact)) {
    it(`${name}: 500 random inputs → mean action (1e-4 abs)`, async () => {
      const g = loadGolden(`policy_${name}`)
      const policy = await loadPolicy(name)
      const obs = Float32Array.from(g.obs.flat())
      const priv = Float32Array.from(g.privileged.flat())
      const out = new Float32Array(2 * g.n)
      policy.act(obs, g.n, out, priv)
      const tol = new Tolerance(1e-4, 0)
      g.action.forEach(([s, l]: number[], i: number) => {
        tol.check(`${i}.steer`, s!, out[2 * i]!)
        tol.check(`${i}.long`, l!, out[2 * i + 1]!)
      })
      expect(tol.failures).toEqual([])
    })
  }

  it('student ensemble: 300 random (obs, history) → action, uncertainty, ẑ, grip (1e-4 abs)', async () => {
    const goldenDir = fileURLToPath(new URL('./golden/', import.meta.url))
    if (!existsSync(`${goldenDir}students.json`)) return
    const g = loadGolden('students')
    const studentFiles = readdirSync(artifact('policies'))
      .filter((f) => /^student_\d+\.json$/.test(f))
      .sort()
    const ens = new StudentEnsemble(
      await loadNet('teacher'),
      await Promise.all(studentFiles.map((f) => loadNet(f.replace('.json', '')))),
    )
    const out = ens.act(Float32Array.from(g.obs.flat()), Float32Array.from(g.history.flat()), g.n)
    const tol = new Tolerance(1e-4, 0)
    for (let i = 0; i < g.n; i++) {
      tol.check(`${i}.steer`, g.action[i][0], out.actions[2 * i]!)
      tol.check(`${i}.long`, g.action[i][1], out.actions[2 * i + 1]!)
      tol.check(`${i}.u`, g.uncertainty[i], out.uncertainty[i]!)
      tol.check(`${i}.grip_f`, g.grip_mean[i][0], out.gripMean[2 * i]!)
      tol.check(`${i}.grip_r`, g.grip_mean[i][1], out.gripMean[2 * i + 1]!)
      g.z_mean[i].forEach((v: number, j: number) =>
        tol.check(`${i}.z${j}`, v, out.zMean[i * ens.latent + j]!),
      )
    }
    expect(tol.failures).toEqual([])
  })

  it('closed loop: the policy drives 20 cases × 400 steps like Python (progress within 2%, ≥ 18/20 crash flags agree)', async () => {
    const g = loadGolden('closed_loop')
    const policy = await loadPolicy(g.policy)
    const tracks = goldenTracks(g.track_seeds, cfg)
    const obs = createObservation(cfg)
    const action = new Float32Array(2)
    const info = createStepInfo()
    const mismatches: string[] = []
    let agree = 0
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
          observe(state, track, null, cfg, obs)
          policy.act(obs.policy, 1, action, obs.privileged)
          coreStep(state, action[0]!, action[1]!, 0, 0, track, cfg, info)
          crashed = info.terminated && !info.lapCompleted
        }
        if ((g.checkpoints as number[]).includes(step)) progress.push(state.progressM)
      }
      if (crashed === c.crashed) agree++
      c.progress_m.forEach((p: number, k: number) => {
        if (Math.abs(p - progress[k]!) / Math.max(Math.abs(p), 1) > 0.02) {
          mismatches.push(
            `${c.suite} track ${c.track_index} @${g.checkpoints[k]}: ${p} vs ${progress[k]}`,
          )
        }
      })
    }
    expect(agree).toBeGreaterThanOrEqual(18)
    expect(mismatches.length).toBeLessThanOrEqual(2) // chaotic dynamics: allow rare drift (SPEC 9.4)
  })
})
