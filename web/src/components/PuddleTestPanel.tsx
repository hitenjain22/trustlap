/**
 * Puddle test panel (SPEC 10.2): a larger chart of true vs estimated rear grip since the test
 * started, and a one-line result scored with the evaluation's rule (state/puddleTest.ts).
 */
import { strings } from '../copy/strings.ts'
import type { Telemetry } from '../hooks/useLiveSim.ts'
import { scorePuddleTest, type PuddleResult } from '../state/puddleTest.ts'
import { Button } from './Button.tsx'

const P = strings.puddleTest
const W = 640
const H = 160
const MU_MAX = 1.3

function line(t: number[], v: number[], t0: number, span: number): string {
  let d = ''
  v.forEach((y, i) => {
    if (!Number.isFinite(y)) return
    const x = ((t[i]! - t0) / span) * W
    d += `${d ? 'L' : 'M'}${x.toFixed(1)} ${(H - (Math.min(y, MU_MAX) / MU_MAX) * (H - 4) - 2).toFixed(1)}`
  })
  return d
}

function resultText(r: PuddleResult): string {
  switch (r.state) {
    case 'waiting':
      return P.waiting
    case 'noticed':
      return P.noticed(r.latencyMs)
    case 'missed':
      return P.missed
    case 'noEstimate':
      return P.noEstimate
  }
}

export function PuddleTestPanel({
  t,
  onRunAgain,
  onEnd,
}: {
  t: Telemetry | null
  onRunAgain: () => void
  onEnd: () => void
}) {
  const h = t?.history
  const result = h ? scorePuddleTest(h.t, h.muRear, h.muHatRear) : ({ state: 'waiting' } as const)
  const t0 = h?.t[0] ?? 0
  const span = Math.max(1, (h?.t[h.t.length - 1] ?? 0) - t0)
  return (
    <section className="border-t border-line px-4 py-4 md:px-6" aria-labelledby="puddle-test-title">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="puddle-test-title" className="text-[16px] font-[620] text-paint">
          {P.title}
        </h2>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={onRunAgain}>
            {P.runAgain}
          </Button>
          <Button variant="ghost" onClick={onEnd}>
            {P.end}
          </Button>
        </div>
      </div>
      <p className="mt-1 max-w-[72ch] text-[14px] text-text-2">{P.body}</p>
      <figure className="mt-3">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="block h-40 w-full"
          role="img"
          aria-label={P.chartLabel}
        >
          <line x1={0} x2={W} y1={H - 0.5} y2={H - 0.5} stroke="var(--line)" />
          {h && (
            <>
              <path
                d={line(h.t, h.muRear, t0, span)}
                fill="none"
                stroke="var(--grip-3)"
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
              <path
                d={line(h.t, h.muHatRear, t0, span)}
                fill="none"
                stroke="var(--sodium)"
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
            </>
          )}
        </svg>
        <figcaption className="mt-2 flex flex-wrap gap-4 text-[13px] text-text-2">
          <span className="flex items-center gap-2">
            <span className="h-0.5 w-5 bg-grip-3" aria-hidden="true" />
            {P.truth}
          </span>
          <span className="flex items-center gap-2">
            <span className="h-0.5 w-5 bg-sodium" aria-hidden="true" />
            {P.estimate}
          </span>
        </figcaption>
      </figure>
      <p role="status" className="mt-3 text-[15px] text-paint">
        {resultText(result)}
      </p>
    </section>
  )
}
