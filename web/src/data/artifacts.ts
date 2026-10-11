/**
 * Fetch + validate artifacts exported by `sim/trustlap/export.py` (SPEC 9). Every file is
 * checked with zod at load; a bad file throws (and the UI shows the error state).
 */
import { z } from 'zod'
import { parseSimConfig, type SimConfig } from '../sim/config.ts'

export const ARTIFACT_VERSION = 'v1'
const base = `/artifacts/${ARTIFACT_VERSION}`

export const showcaseSchema = z.object({
  tracks: z
    .array(
      z.object({
        seed: z.number().int(),
        name: z.string(),
        slug: z.string(),
        length_m: z.number(),
        width_m: z.number(),
        min_radius_m: z.number(),
        aspect: z.number().positive(), // bounding-box width / height
      }),
    )
    .min(1),
})
export type ShowcaseTrack = z.infer<typeof showcaseSchema>['tracks'][number]

export const benchSchema = z.object({
  created_utc: z.string(),
  git_sha: z.string(),
  device: z.string(),
  rows: z.array(z.object({ impl: z.string(), n_cars: z.number().int(), steps_per_s: z.number() })),
})
export type Bench = z.infer<typeof benchSchema>

/** Headline benchmark numbers, derived only from measured rows (nothing extrapolated). */
export function benchHeadline(
  bench: Bench,
): { maxCars: number; speedup: number; device: string } | null {
  const gpu = bench.rows.filter((r) => r.impl.startsWith('jax-') && r.impl !== 'jax-cpu')
  const cpu = bench.rows.filter((r) => r.impl === 'jax-cpu')
  if (gpu.length === 0 || cpu.length === 0) return null
  const maxCars = Math.max(...gpu.map((r) => r.n_cars))
  // Compare at the largest car count measured on both.
  const shared = gpu.filter((g) => cpu.some((c) => c.n_cars === g.n_cars)).map((r) => r.n_cars)
  if (shared.length === 0) return null
  const n = Math.max(...shared)
  const g = gpu.find((r) => r.n_cars === n)!
  const c = cpu.find((r) => r.n_cars === n)!
  return {
    maxCars,
    speedup: g.steps_per_s / c.steps_per_s,
    device: bench.device.replace(/^Tesla /, ''),
  }
}

const cache = new Map<string, Promise<unknown>>()

async function fetchJson<T>(path: string, parse: (raw: unknown) => T): Promise<T> {
  const key = `${base}/${path}`
  if (!cache.has(key)) {
    cache.set(
      key,
      fetch(key).then(async (res) => {
        if (!res.ok) throw new Error(`${key}: HTTP ${res.status}`)
        try {
          return parse(await res.json())
        } catch (error) {
          console.error(`Artifact ${key} failed validation`, error)
          throw error
        }
      }),
    )
  }
  return cache.get(key) as Promise<T>
}

export const loadSimConfig = (): Promise<SimConfig> => fetchJson('config.json', parseSimConfig)
export const loadShowcase = (): Promise<ShowcaseTrack[]> =>
  fetchJson('tracks/showcase.json', (raw) => showcaseSchema.parse(raw).tracks)

export const loadBench = (): Promise<Bench> =>
  fetchJson('bench.json', (raw) => benchSchema.parse(raw))

/** The showcase track whose shape best fits a screen region of this width / height. */
export function bestFitTrack(tracks: ShowcaseTrack[], regionAspect: number): ShowcaseTrack {
  const score = (t: ShowcaseTrack) => Math.abs(Math.log(t.aspect / regionAspect))
  return tracks.reduce((best, t) => (score(t) < score(best) ? t : best))
}

/** Forget cached failures so a "Reload" button can retry. */
export function clearArtifactCache(): void {
  cache.clear()
}
