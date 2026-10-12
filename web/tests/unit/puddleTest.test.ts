import { expect, it } from 'vitest'
import { scorePuddleTest } from '../../src/state/puddleTest.ts'

/** 50 ms samples; the same cases as test_notice_latencies_window_and_control (Python). */
const series = (n: number, f: (i: number) => number) => Array.from({ length: n }, (_, i) => f(i))
const t = series(100, (i) => i * 0.05)
const mu = series(100, (i) => (i >= 10 && i < 16 ? 0.6 : 1.0)) // low for 300 ms

it('a prompt response counts', () => {
  const hat = series(100, (i) => (i >= 12 ? 0.75 : 1.0))
  expect(scorePuddleTest(t, mu, hat)).toEqual({ state: 'noticed', latencyMs: 100 })
})
it('a dip seconds later does not', () => {
  const hat = series(100, (i) => (i >= 60 ? 0.7 : 1.0))
  expect(scorePuddleTest(t, mu, hat)).toEqual({ state: 'missed' })
})
it('waits until the window has passed', () => {
  const hat = series(100, () => 1.0)
  expect(scorePuddleTest(t.slice(0, 18), mu.slice(0, 18), hat.slice(0, 18))).toEqual({
    state: 'waiting',
  })
  expect(scorePuddleTest(t.slice(0, 12), mu.slice(0, 12), hat.slice(0, 12))).toEqual({
    state: 'waiting',
  })
})
it('drivers without a grip estimate are reported as such', () => {
  expect(
    scorePuddleTest(
      t,
      mu,
      series(100, () => NaN),
    ),
  ).toEqual({ state: 'noEstimate' })
})
