/**
 * Student ensemble (SPEC 7.6–7.7): K adaptation modules estimate ẑ_k (and grip) from the last
 * second of motion; the teacher actor drives on the mean ẑ. Uncertainty
 * u = mean over action dims of std_k(actor(obs, ẑ_k)) + 0.5 · mean over latent dims of std_k(ẑ_k).
 * Mirrors `confidence.ensemble_act` in Python (parity-tested).
 */
import type { LoadedNetwork } from './loader.ts'
import { Mlp } from './mlp.ts'

interface Student {
  trunk: Mlp
  zHead: Mlp
  gripHead: Mlp
  gripScale: number
}

export interface EnsembleOutputs {
  actions: Float32Array // [n × 2]
  uncertainty: Float32Array // [n]
  zMean: Float32Array // [n × latent]
  gripMean: Float32Array // [n × 2] μ̂_front, μ̂_rear
}

/** Population std (divide by K), like numpy/jax `std`. */
function std(values: Float64Array, k: number): number {
  let mean = 0
  for (let i = 0; i < k; i++) mean += values[i]!
  mean /= k
  let v = 0
  for (let i = 0; i < k; i++) v += (values[i]! - mean) ** 2
  return Math.sqrt(v / k)
}

export class StudentEnsemble {
  private readonly students: Student[]
  private readonly actor: Mlp
  private readonly obsMean: Float32Array
  private readonly obsStd: Float32Array
  private readonly clipValue: number
  readonly latent: number
  readonly size: number
  private input = new Float32Array(0)
  private zAll = new Float32Array(0)
  private gripAll = new Float32Array(0)
  private aAll = new Float32Array(0)
  private readonly scratch: Float64Array
  readonly out: EnsembleOutputs = {
    actions: new Float32Array(0),
    uncertainty: new Float32Array(0),
    zMean: new Float32Array(0),
    gripMean: new Float32Array(0),
  }

  /** `teacher` = the teacher network (actor + encoder + obs_norm); `students` = student_k. */
  constructor(teacher: LoadedNetwork, students: LoadedNetwork[]) {
    const m = teacher.meta
    if (m.kind !== 'teacher' || !m.obs_norm) throw new Error('ensemble needs the teacher network')
    if (students.length === 0) throw new Error('ensemble needs at least one student')
    this.actor = new Mlp(teacher.layers.actor!)
    this.obsMean = Float32Array.from(m.obs_norm.mean)
    this.obsStd = Float32Array.from(m.obs_norm.std)
    this.clipValue = m.obs_clip
    this.students = students.map((s) => ({
      trunk: new Mlp(s.layers.trunk!),
      zHead: new Mlp(s.layers.z_head!),
      gripHead: new Mlp(s.layers.grip_head!),
      gripScale: Number((s.meta as { grip_scale?: number }).grip_scale ?? 1.3),
    }))
    this.latent = this.students[0]!.zHead.outputSize
    this.size = this.students.length
    this.scratch = new Float64Array(this.size)
  }

  private grow(n: number, k: number): void {
    const d = this.obsMean.length
    const width = d + this.latent
    if (this.input.length < n * width) this.input = new Float32Array(n * width)
    if (this.zAll.length < k * n * this.latent) this.zAll = new Float32Array(k * n * this.latent)
    if (this.gripAll.length < k * n * 2) this.gripAll = new Float32Array(k * n * 2)
    if (this.aAll.length < k * n * 2) this.aAll = new Float32Array(k * n * 2)
    if (this.out.actions.length < n * 2) {
      this.out.actions = new Float32Array(n * 2)
      this.out.uncertainty = new Float32Array(n)
      this.out.zMean = new Float32Array(n * this.latent)
      this.out.gripMean = new Float32Array(n * 2)
    }
  }

  /** Fill rows of `input` with [normalized obs, z] for the actor. */
  private actorInput(obs: Float32Array, n: number, z: Float32Array, zOffset: number): void {
    const d = this.obsMean.length
    const width = d + this.latent
    const c = this.clipValue
    for (let b = 0; b < n; b++) {
      for (let j = 0; j < d; j++) {
        const v = (obs[b * d + j]! - this.obsMean[j]!) / this.obsStd[j]!
        this.input[b * width + j] = Math.min(c, Math.max(-c, v))
      }
      for (let j = 0; j < this.latent; j++)
        this.input[b * width + d + j] = z[zOffset + b * this.latent + j]!
    }
  }

  /**
   * Decide for `n` cars. `obs` [n × 33] raw, `history` [n × 200]. `members` (≤ K) limits how
   * many students run (the hero uses 1 to stay light on phones; uncertainty is then 0).
   * Results land in `this.out` (valid until the next call).
   */
  act(
    obs: Float32Array,
    history: Float32Array,
    n: number,
    members = this.size,
    withUncertainty = true,
  ): EnsembleOutputs {
    const k = Math.max(1, Math.min(members, this.size))
    const L = this.latent
    this.grow(n, k)
    for (let s = 0; s < k; s++) {
      const st = this.students[s]!
      const hidden = st.trunk.forward(history, n)
      const z = st.zHead.forward(hidden, n)
      this.zAll.set(z, s * n * L)
      const g = st.gripHead.forward(hidden, n)
      for (let i = 0; i < 2 * n; i++) this.gripAll[s * n * 2 + i] = g[i]! * st.gripScale
    }
    const o = this.out
    for (let b = 0; b < n; b++) {
      for (let j = 0; j < L; j++) {
        let acc = 0
        for (let s = 0; s < k; s++) acc += this.zAll[s * n * L + b * L + j]!
        o.zMean[b * L + j] = acc / k
      }
      for (let j = 0; j < 2; j++) {
        let acc = 0
        for (let s = 0; s < k; s++) acc += this.gripAll[s * n * 2 + b * 2 + j]!
        o.gripMean[b * 2 + j] = acc / k
      }
    }
    this.actorInput(obs, n, o.zMean, 0)
    const y = this.actor.forward(this.input, n)
    for (let i = 0; i < 2 * n; i++) o.actions[i] = Math.min(1, Math.max(-1, y[i]!))
    if (!withUncertainty || k === 1) {
      o.uncertainty.fill(0, 0, n)
      return o
    }
    for (let s = 0; s < k; s++) {
      this.actorInput(obs, n, this.zAll, s * n * L)
      const a = this.actor.forward(this.input, n)
      for (let i = 0; i < 2 * n; i++) this.aAll[s * n * 2 + i] = Math.min(1, Math.max(-1, a[i]!))
    }
    for (let b = 0; b < n; b++) {
      let actionStd = 0
      for (let j = 0; j < 2; j++) {
        for (let s = 0; s < k; s++) this.scratch[s] = this.aAll[s * n * 2 + b * 2 + j]!
        actionStd += std(this.scratch, k)
      }
      let zStd = 0
      for (let j = 0; j < L; j++) {
        for (let s = 0; s < k; s++) this.scratch[s] = this.zAll[s * n * L + b * L + j]!
        zStd += std(this.scratch, k)
      }
      o.uncertainty[b] = actionStd / 2 + (0.5 * zStd) / L
    }
    return o
  }
}
