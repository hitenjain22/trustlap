/**
 * Portable PCG32, bit-identical with `sim/trustlap/rng.py` (PCG-XSH-RR 64/32, O'Neill 2014,
 * reference seeding with stream 54). 64-bit state arithmetic uses BigInt; outputs are uint32.
 */

const MASK64 = (1n << 64n) - 1n
const MULTIPLIER = 6364136223846793005n
const DEFAULT_STREAM = 54

export class PCG32 {
  private state = 0n
  private readonly inc: bigint

  constructor(seed: number, stream: number = DEFAULT_STREAM) {
    this.inc = ((BigInt(stream) << 1n) | 1n) & MASK64
    this.nextU32()
    this.state = (this.state + (BigInt(seed) & MASK64)) & MASK64
    this.nextU32()
  }

  /** Next uniformly distributed 32-bit unsigned integer. */
  nextU32(): number {
    const old = this.state
    this.state = (old * MULTIPLIER + this.inc) & MASK64
    const xorshifted = Number((((old >> 18n) ^ old) >> 27n) & 0xffffffffn)
    const rot = Number(old >> 59n)
    return ((xorshifted >>> rot) | (xorshifted << (-rot & 31))) >>> 0
  }

  /** Uniform float in [0, 1): u32 / 2^32 (exact in float64). */
  nextFloat01(): number {
    return this.nextU32() / 4294967296
  }

  /** Uniform float in [lo, hi). */
  nextRange(lo: number, hi: number): number {
    return lo + (hi - lo) * this.nextFloat01()
  }

  /** Uniform integer in [lo, hi], inclusive (modulo method, like the Python version). */
  nextInt(lo: number, hi: number): number {
    return lo + (this.nextU32() % (hi - lo + 1))
  }

  /** Standard normal draw (Box–Muller). Browser-only: training noise comes from jax.random. */
  nextNormal(): number {
    const u1 = 1 - this.nextFloat01() // (0, 1]
    const u2 = this.nextFloat01()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
}

/** 32-bit integer hash (32-bit SplitMix variant), used to derive track retry seeds. */
export function splitmix32(x: number): number {
  const seeded = (x + 0x9e3779b9) >>> 0
  let z = seeded
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0
  return (z ^ (z >>> 16)) >>> 0
}
