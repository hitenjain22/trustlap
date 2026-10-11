import { describe, expect, it } from 'vitest'
import { Mlp } from '../../src/policy/mlp.ts'

describe('MLP', () => {
  it('matches a hand-computed 2-layer example', () => {
    // x = [1, 2]; layer 1: W = [[1, -1], [0.5, 1]] ([in, out]), b = [0, -4], ELU; layer 2: W = [[1], [1]], b = [0.5]
    const mlp = new Mlp([
      {
        name: 'l1',
        in: 2,
        out: 2,
        activation: 'elu',
        kernel: Float32Array.from([1, -1, 0.5, 1]),
        bias: Float32Array.from([0, -4]),
      },
      {
        name: 'l2',
        in: 2,
        out: 1,
        activation: 'linear',
        kernel: Float32Array.from([1, 1]),
        bias: Float32Array.from([0.5]),
      },
    ])
    // h1 = elu(1·1 + 2·0.5 + 0) = 2; h2 = elu(1·(−1) + 2·1 − 4) = elu(−3) = e^−3 − 1
    const y = mlp.forward(Float32Array.from([1, 2]), 1)
    expect(y[0]).toBeCloseTo(2 + Math.expm1(-3) + 0.5, 6)
  })
  it('processes a batch row by row', () => {
    const mlp = new Mlp([
      {
        name: 'l',
        in: 1,
        out: 1,
        activation: 'tanh',
        kernel: Float32Array.from([2]),
        bias: Float32Array.from([0]),
      },
    ])
    const y = mlp.forward(Float32Array.from([0, 0.5, -1]), 3)
    expect(Array.from(y)).toEqual([0, Math.fround(Math.tanh(1)), Math.fround(Math.tanh(-2))])
  })
  it('rejects mismatched shapes', () => {
    expect(
      () =>
        new Mlp([
          {
            name: 'l',
            in: 2,
            out: 2,
            activation: 'linear',
            kernel: new Float32Array(3),
            bias: new Float32Array(2),
          },
        ]),
    ).toThrow()
  })
})
