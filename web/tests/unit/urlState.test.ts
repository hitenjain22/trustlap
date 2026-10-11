import { describe, expect, it } from 'vitest'
import { parsePlayground, serializePlayground } from '../../src/state/urlState.ts'

describe('playground URL state', () => {
  it('parses the spec example', () => {
    const s = parsePlayground(
      'track=s1234&rain=60&puddles=3&wind=8&windDir=135&gust=1.5&mass=1.1&driver=student&ghosts=blind,teacher&gov=1&tau=20&speed=1',
    )
    expect(s.track).toEqual({ seed: 1234 })
    expect(s.values).toMatchObject({
      rain: 0.6,
      puddles: 3,
      windMps: 8,
      windDirDeg: 135,
      gustMps: 1.5,
      massMult: 1.1,
    })
    expect(s.driver).toBe('student')
    expect(s.ghosts).toEqual(['blind', 'teacher'])
    expect(s.governor).toBe(true)
    expect(s.tau).toBeCloseTo(0.2)
    expect(s.speed).toBe(1)
  })
  it('clamps out-of-range values and drops junk', () => {
    const s = parsePlayground(
      'rain=500&puddles=99&driver=robot&ghosts=blind,ufo&speed=7&track=../evil&lag=abc&unknown=1',
    )
    expect(s.values.rain).toBe(1)
    expect(s.values.puddles).toBe(8)
    expect(s.values.steerTauS).toBeUndefined()
    expect(s.driver).toBeNull()
    expect(s.ghosts).toEqual(['blind'])
    expect(s.speed).toBeNull()
    expect(s.track).toBeNull()
  })
  it('accepts a showcase slug and round-trips', () => {
    const q = serializePlayground({
      trackSeed: 1,
      trackSlug: 'hairpin-valley',
      values: { rain: 0.35, delaySteps: 2 },
      driver: 'blind',
      ghosts: ['scripted'],
      governor: false,
      tau: 0.15,
      speed: 2,
    })
    const s = parsePlayground(q)
    expect(s.track).toEqual({ slug: 'hairpin-valley' })
    expect(s.values).toMatchObject({ rain: 0.35, delaySteps: 2 })
    expect(s).toMatchObject({ driver: 'blind', ghosts: ['scripted'], governor: false, speed: 2 })
  })
})
