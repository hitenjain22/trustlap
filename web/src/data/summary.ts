/**
 * Plain-language summary of the evaluation (Results "Does it work?"), computed from metrics.json.
 * "Clearly better/worse" means the paired 95% interval for Adaptive − Basic completion excludes
 * zero; anything else is "no clear difference". Nothing here is typed by hand.
 */
import type { Metrics } from './schemas.ts'

/** Conditions inside the training range, in the order a newcomer reads them. */
export const TRAINED_CONDITIONS = [
  'dry',
  'wet',
  'puddles',
  'worn',
  'wind',
  'heavy',
  'laggy',
  'combined',
] as const
/** Conditions never trained on. */
export const UNTRAINED_CONDITIONS = ['ice', 'extreme_lag', 'noisy_sensors', 'very_heavy'] as const

export type Verdict = 'better' | 'worse' | 'same'

export interface ConditionRow {
  suite: string
  /** Laps finished per 100 (rounded) by the Adaptive AI and the Basic AI. */
  adaptive: number
  basic: number
  verdict: Verdict
  /** In how many independent training runs the same verdict held (when replicated). */
  heldIn?: { same: number; of: number }
}

const per100 = (v: number) => Math.round(v * 100)

/** One row per condition with both drivers' completion and the paired verdict. */
export function compareConditions(m: Metrics, suites: readonly string[]): ConditionRow[] {
  const rows: ConditionRow[] = []
  for (const suite of suites) {
    const a = m.suites?.[suite]?.student?.completion?.value
    const b = m.suites?.[suite]?.blind?.completion?.value
    if (a == null || b == null) continue
    const d = (m.paired_deltas ?? []).find(
      (x) => x.suite === suite && x.a === 'student' && x.b === 'blind' && x.metric === 'completion',
    )
    const [lo, hi] = d?.ci ?? [null, null]
    const verdict: Verdict =
      lo != null && lo > 0 ? 'better' : hi != null && hi < 0 ? 'worse' : 'same'
    const rep = m.replication?.agreement[suite]
    rows.push({
      suite,
      adaptive: per100(a),
      basic: per100(b),
      verdict,
      ...(rep ? { heldIn: { same: rep.same_in, of: rep.of } } : {}),
    })
  }
  return rows
}

/** Best laps per 100 any AI driver managed in a condition (for the "outside training" line). */
export function bestAiLaps(m: Metrics, suite: string): number | null {
  const vals = (['student', 'blind', 'teacher'] as const)
    .map((d) => m.suites?.[suite]?.[d]?.completion?.value)
    .filter((v): v is number => v != null)
  return vals.length ? per100(Math.max(...vals)) : null
}

/** The condition where the Adaptive AI gained (or lost) the most, among clear verdicts. */
export function biggest(rows: ConditionRow[], verdict: 'better' | 'worse'): ConditionRow | null {
  const pick = rows.filter((r) => r.verdict === verdict)
  if (!pick.length) return null
  return pick.reduce((best, r) =>
    verdict === 'better'
      ? r.adaptive - r.basic > best.adaptive - best.basic
        ? r
        : best
      : r.adaptive - r.basic < best.adaptive - best.basic
        ? r
        : best,
  )
}
