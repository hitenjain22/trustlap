/**
 * Shareable Playground state in the URL (SPEC 10.5). Parsed with zod, clamped to valid ranges,
 * unknown keys dropped. Units in the URL are friendly: percentages for rain and τ, ms-free
 * seconds for lag, m/s for wind.
 *
 *   /playground?track=s1234&rain=60&puddles=3&wind=8&windDir=135&gust=1.5&mass=1.1&driver=student&ghosts=blind,teacher&gov=1&tau=20&speed=1
 */
import { z } from 'zod'
import { DRIVER_KINDS, type ConditionValues, type DriverKind } from '../worker/protocol.ts'

export interface PlaygroundState {
  track: { seed: number } | { slug: string } | null
  values: Partial<ConditionValues>
  driver: DriverKind | null
  ghosts: DriverKind[] | null
  governor: boolean | null
  tau: number | null
  speed: number | null
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const numIn = (lo: number, hi: number) =>
  z.coerce
    .number()
    .refine(Number.isFinite)
    .transform((v) => clamp(v, lo, hi))
    .optional()
    .catch(undefined)

// Generous bounds: every slider's full range (training ∪ test extremes) fits inside.
const schema = z.object({
  track: z
    .string()
    .regex(/^(s\d{1,10}|[a-z0-9-]{1,40})$/)
    .optional()
    .catch(undefined),
  rain: numIn(0, 100),
  grip: numIn(0.1, 1.3),
  puddles: numIn(0, 8),
  wind: numIn(0, 20),
  windDir: numIn(0, 360),
  gust: numIn(0, 6),
  mass: numIn(0.5, 2),
  power: numIn(0.3, 1.5),
  brake: numIn(0.3, 1.5),
  drag: numIn(0.3, 2.5),
  wear: numIn(0, 1),
  lag: numIn(0.01, 0.5),
  delay: numIn(0, 4),
  noise: numIn(0, 0.3),
  driver: z.enum(DRIVER_KINDS).optional().catch(undefined),
  ghosts: z.string().max(60).optional().catch(undefined),
  gov: z.enum(['0', '1']).optional().catch(undefined),
  tau: numIn(5, 50),
  speed: z.enum(['0.5', '1', '2']).optional().catch(undefined),
})

const KEYS: [keyof z.infer<typeof schema>, keyof ConditionValues, number][] = [
  ['rain', 'rain', 100],
  ['grip', 'muDry', 1],
  ['puddles', 'puddles', 1],
  ['wind', 'windMps', 1],
  ['windDir', 'windDirDeg', 1],
  ['gust', 'gustMps', 1],
  ['mass', 'massMult', 1],
  ['power', 'powerMult', 1],
  ['brake', 'brakeMult', 1],
  ['drag', 'dragAreaM2', 1],
  ['wear', 'wearPerKm', 1],
  ['lag', 'steerTauS', 1],
  ['delay', 'delaySteps', 1],
  ['noise', 'obsNoise', 1],
]

/** Parse a query string; anything invalid is dropped, anything out of range clamped. */
export function parsePlayground(search: string): PlaygroundState {
  const raw = Object.fromEntries(new URLSearchParams(search))
  const q = schema.parse(raw)
  const values: Partial<ConditionValues> = {}
  for (const [k, field, scale] of KEYS) {
    const v = q[k] as number | undefined
    if (v !== undefined)
      values[field] = k === 'puddles' || k === 'delay' ? Math.round(v) : v / scale
  }
  const ghosts = q.ghosts
    ?.split(',')
    .filter((g): g is DriverKind => (DRIVER_KINDS as readonly string[]).includes(g))
  return {
    track: q.track
      ? q.track.startsWith('s') && /^s\d+$/.test(q.track)
        ? { seed: Number(q.track.slice(1)) }
        : { slug: q.track }
      : null,
    values,
    driver: q.driver ?? null,
    ghosts: ghosts ?? null,
    governor: q.gov === undefined ? null : q.gov === '1',
    tau: q.tau === undefined ? null : q.tau / 100,
    speed: q.speed === undefined ? null : Number(q.speed),
  }
}

/** Serialize the current Playground state (only what's set). */
export function serializePlayground(s: {
  trackSeed: number | null
  trackSlug: string | null
  values: Partial<ConditionValues>
  driver: DriverKind
  ghosts: DriverKind[]
  governor: boolean
  tau: number
  speed: number
}): string {
  const p = new URLSearchParams()
  if (s.trackSlug) p.set('track', s.trackSlug)
  else if (s.trackSeed !== null) p.set('track', `s${s.trackSeed}`)
  for (const [k, field, scale] of KEYS) {
    const v = s.values[field]
    if (v !== undefined) p.set(k, String(Math.round(v * scale * 1000) / 1000))
  }
  p.set('driver', s.driver)
  p.set('ghosts', s.ghosts.join(','))
  p.set('gov', s.governor ? '1' : '0')
  p.set('tau', String(Math.round(s.tau * 100)))
  p.set('speed', String(s.speed))
  return p.toString()
}
