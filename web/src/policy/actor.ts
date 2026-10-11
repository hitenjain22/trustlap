/**
 * Deployable driving policy (SPEC 7.1): normalize observations with the frozen training
 * statistics, run the actor MLP, take the mean action, clip to [−1, 1].
 * Blind: actor(obs). Teacher: actor(obs, z = encoder(privileged)).
 */
import type { LoadedNetwork } from './loader.ts'
import { Mlp } from './mlp.ts'

export class ActorPolicy {
  readonly kind: 'blind' | 'teacher'
  private readonly actor: Mlp
  private readonly encoder: Mlp | null
  private readonly obsMean: Float32Array
  private readonly obsStd: Float32Array
  private readonly privMean: Float32Array | null
  private readonly privStd: Float32Array | null
  private readonly clipValue: number
  private input = new Float32Array(0)
  private privInput = new Float32Array(0)
  readonly obsSize: number
  readonly privSize: number

  constructor(net: LoadedNetwork) {
    const m = net.meta
    if (m.kind === 'student') throw new Error('ActorPolicy needs a blind or teacher network')
    if (!m.obs_norm) throw new Error(`${m.name}: missing obs_norm`)
    this.kind = m.kind
    this.actor = new Mlp(net.layers.actor!)
    this.encoder = m.kind === 'teacher' ? new Mlp(net.layers.encoder!) : null
    this.obsMean = Float32Array.from(m.obs_norm.mean)
    this.obsStd = Float32Array.from(m.obs_norm.std)
    this.obsSize = this.obsMean.length
    this.privMean = m.privileged_norm ? Float32Array.from(m.privileged_norm.mean) : null
    this.privStd = m.privileged_norm ? Float32Array.from(m.privileged_norm.std) : null
    this.privSize = this.privMean?.length ?? 0
    this.clipValue = m.obs_clip
  }

  /**
   * Mean actions for `n` cars. `obs` [n × 33] raw observations; `privileged` [n × 14] (teacher
   * only). Writes [n × 2] into `out`.
   */
  act(obs: Float32Array, n: number, out: Float32Array, privileged?: Float32Array): void {
    const d = this.obsSize
    const latent = this.encoder?.outputSize ?? 0
    const width = d + latent
    if (this.input.length < n * width) this.input = new Float32Array(n * width)
    const c = this.clipValue
    for (let b = 0; b < n; b++) {
      for (let k = 0; k < d; k++) {
        const z = (obs[b * d + k]! - this.obsMean[k]!) / this.obsStd[k]!
        this.input[b * width + k] = Math.min(c, Math.max(-c, z))
      }
    }
    if (this.encoder) {
      if (!privileged || !this.privMean || !this.privStd)
        throw new Error('teacher needs privileged input')
      const p = this.privSize
      if (this.privInput.length < n * p) this.privInput = new Float32Array(n * p)
      for (let b = 0; b < n; b++) {
        for (let k = 0; k < p; k++) {
          const z = (privileged[b * p + k]! - this.privMean[k]!) / this.privStd[k]!
          this.privInput[b * p + k] = Math.min(c, Math.max(-c, z))
        }
      }
      const zs = this.encoder.forward(this.privInput, n)
      for (let b = 0; b < n; b++) {
        // The exported encoder ends in tanh already (z = tanh(encoder(privileged))).
        for (let k = 0; k < latent; k++) this.input[b * width + d + k] = zs[b * latent + k]!
      }
    }
    const y = this.actor.forward(this.input, n)
    for (let i = 0; i < 2 * n; i++) out[i] = Math.min(1, Math.max(-1, y[i]!))
  }
}
