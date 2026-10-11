"""Portable PCG32 random number generator, bit-identical with `web/src/sim/rng.ts`.

Used for anything the browser must reproduce exactly (track generation, browser-side
condition sampling). JAX training randomness uses `jax.random` instead.

Algorithm: PCG-XSH-RR 64/32 (M. E. O'Neill, 2014), seeded exactly like the reference
`pcg32_srandom_r(rng, initstate, initseq)`. All arithmetic is on Python ints with explicit
64-/32-bit masks so it matches the BigInt / `>>> 0` arithmetic in TypeScript.
"""

from __future__ import annotations

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
PCG_MULTIPLIER = 6364136223846793005  # LCG multiplier from the PCG reference
DEFAULT_STREAM = 54  # stream id used by the PCG reference demo; fixed for TrustLap


class PCG32:
    """PCG32 generator: 64-bit state, 32-bit output."""

    def __init__(self, seed: int, stream: int = DEFAULT_STREAM) -> None:
        self.state = 0
        self.inc = ((stream << 1) | 1) & MASK64
        self.next_u32()
        self.state = (self.state + (seed & MASK64)) & MASK64
        self.next_u32()

    def next_u32(self) -> int:
        """Next uniformly distributed 32-bit unsigned integer."""
        old = self.state
        self.state = (old * PCG_MULTIPLIER + self.inc) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & MASK32
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK32

    def next_float01(self) -> float:
        """Uniform float in [0, 1): u32 / 2^32 (exact in float64)."""
        return self.next_u32() / 4294967296.0

    def next_range(self, lo: float, hi: float) -> float:
        """Uniform float in [lo, hi)."""
        return lo + (hi - lo) * self.next_float01()

    def next_int(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi], inclusive (modulo method; bias < 1e-8 for small ranges)."""
        return lo + self.next_u32() % (hi - lo + 1)


def splitmix32(x: int) -> int:
    """32-bit integer hash (the 32-bit SplitMix variant), used to derive retry seeds."""
    x = (x + 0x9E3779B9) & MASK32
    z = x
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & MASK32
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & MASK32
    return (z ^ (z >> 16)) & MASK32
