/**
 * Live puddle-test scoring, the same rule as `evaluate.notice_latencies` (sim/trustlap/evaluate.py):
 * time from the true rear grip first dropping by 0.1 until the estimate has fallen by half the
 * true drop, counted only while grip is low or within 0.5 s after it recovers.
 */
export const NOTICE_MARGIN_MS = 500 // evaluate.NOTICE_MARGIN_MS

export type PuddleResult =
  | { state: 'waiting' } // not in the puddle yet, or still inside the scoring window
  | { state: 'noticed'; latencyMs: number }
  | { state: 'missed' }
  | { state: 'noEstimate' } // this driver doesn't estimate grip

/** Score a run from samples: sim time (s), true rear grip, estimated rear grip (NaN = none). */
export function scorePuddleTest(t: number[], mu: number[], muHat: number[]): PuddleResult {
  if (!mu.length) return { state: 'waiting' }
  if (muHat.every((v) => !Number.isFinite(v))) return { state: 'noEstimate' }
  const base = mu[0]!
  const t0 = mu.findIndex((v) => v < base - 0.1)
  if (t0 < 0) return { state: 'waiting' }
  let lowEnd = t0
  for (let i = t0; i < mu.length; i++) if (mu[i]! < base - 0.1) lowEnd = i
  const windowEndS = t[lowEnd]! + NOTICE_MARGIN_MS / 1000
  let minMu = Infinity
  for (let i = t0; i < mu.length; i++) minMu = Math.min(minMu, mu[i]!)
  const threshold = 0.5 * (base - minMu)
  for (let i = t0; i < mu.length && t[i]! <= windowEndS; i++) {
    if (muHat[i]! <= muHat[t0]! - threshold) {
      return { state: 'noticed', latencyMs: Math.round((t[i]! - t[t0]!) * 1000) }
    }
  }
  // Decide only once the window has fully passed.
  return t[t.length - 1]! > windowEndS ? { state: 'missed' } : { state: 'waiting' }
}
