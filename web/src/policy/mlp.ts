/**
 * Batched MLP inference over Float32Arrays (SPEC 3.2, 10.6): no dependencies, no per-call
 * allocation. Kernels arrive as [in, out] row-major (Flax layout, y = x·W + b) and are
 * transposed once at load so the inner loop reads weights sequentially.
 */

export type Activation = 'elu' | 'tanh' | 'linear' | 'sigmoid'

export interface DenseLayer {
  name: string
  in: number
  out: number
  activation: Activation
  kernel: Float32Array // [in × out] row-major, as exported
  bias: Float32Array // [out]
}

interface PreparedLayer {
  in: number
  out: number
  activation: Activation
  weightsT: Float32Array // [out × in]
  bias: Float32Array
}

export class Mlp {
  readonly inputSize: number
  readonly outputSize: number
  private readonly layers: PreparedLayer[]
  private bufA: Float32Array
  private bufB: Float32Array
  private capacity = 0

  constructor(layers: DenseLayer[]) {
    if (layers.length === 0) throw new Error('MLP needs at least one layer')
    for (let i = 1; i < layers.length; i++) {
      if (layers[i]!.in !== layers[i - 1]!.out)
        throw new Error(`layer ${layers[i]!.name}: shape mismatch`)
    }
    this.layers = layers.map((l) => {
      if (l.kernel.length !== l.in * l.out || l.bias.length !== l.out) {
        throw new Error(`layer ${l.name}: weights don't match ${l.in}×${l.out}`)
      }
      const weightsT = new Float32Array(l.in * l.out)
      for (let i = 0; i < l.in; i++) {
        for (let o = 0; o < l.out; o++) weightsT[o * l.in + i] = l.kernel[i * l.out + o]!
      }
      return { in: l.in, out: l.out, activation: l.activation, weightsT, bias: l.bias }
    })
    this.inputSize = layers[0]!.in
    this.outputSize = layers[layers.length - 1]!.out
    this.bufA = new Float32Array(0)
    this.bufB = new Float32Array(0)
  }

  private ensure(n: number): void {
    if (n <= this.capacity) return
    const width = Math.max(...this.layers.map((l) => Math.max(l.in, l.out)))
    this.bufA = new Float32Array(n * width)
    this.bufB = new Float32Array(n * width)
    this.capacity = n
  }

  /**
   * Run `n` inputs ([n × inputSize], row-major). Returns a view [n × outputSize] that is
   * valid until the next call.
   */
  forward(input: Float32Array, n: number): Float32Array {
    this.ensure(n)
    let x = input
    let out = this.bufA
    for (const layer of this.layers) {
      const { in: nin, out: nout, weightsT: w, bias } = layer
      for (let b = 0; b < n; b++) {
        const xo = b * nin
        const yo = b * nout
        for (let o = 0; o < nout; o++) {
          let acc = bias[o]!
          const wo = o * nin
          for (let i = 0; i < nin; i++) acc += x[xo + i]! * w[wo + i]!
          out[yo + o] = activate(acc, layer.activation)
        }
      }
      x = out
      out = out === this.bufA ? this.bufB : this.bufA
    }
    return x.subarray(0, n * this.outputSize)
  }
}

function activate(v: number, a: Activation): number {
  switch (a) {
    case 'elu':
      return v > 0 ? v : Math.expm1(v)
    case 'tanh':
      return Math.tanh(v)
    case 'sigmoid':
      return 1 / (1 + Math.exp(-v))
    default:
      return v
  }
}
