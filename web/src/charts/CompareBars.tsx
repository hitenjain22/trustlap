/**
 * The one chart a newcomer needs: for each condition, laps finished out of 100 by the Adaptive AI
 * and the Basic AI, as two thin bars with the number at the end, and a plain verdict. Plain HTML,
 * so it reflows on any screen and reads naturally with a screen reader.
 */
import { useEffect, useRef, useState } from 'react'
import { strings } from '../copy/strings.ts'
import type { ConditionRow } from '../data/summary.ts'

const C = strings.results.simple

function Bar({
  value,
  color,
  label,
  shown,
  delay,
}: {
  value: number
  color: string
  label: string
  shown: boolean
  delay: number
}) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-3 flex-1 rounded-full bg-asphalt-700" aria-hidden="true">
        <div
          className="h-3 rounded-full transition-[width] duration-700 ease-[cubic-bezier(0.2,0.8,0.2,1)]"
          style={{
            width: shown ? `${Math.max(value, 0.5)}%` : '0%',
            background: `linear-gradient(90deg, color-mix(in srgb, ${color} 70%, transparent), ${color})`,
            transitionDelay: `${delay}ms`,
          }}
        />
      </div>
      <span className="numeral w-9 text-right text-[14px] text-paint">{value}</span>
      <span className="sr-only">{label}</span>
    </div>
  )
}

export function CompareBars({ rows }: { rows: ConditionRow[] }) {
  // Bars grow in when the chart scrolls into view (instant under reduced motion).
  const ref = useRef<HTMLElement | null>(null)
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => e?.isIntersecting && setShown(true), {
      threshold: 0.2,
    })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return (
    <figure ref={ref}>
      <figcaption className="flex flex-wrap gap-x-5 gap-y-1 text-[13px] text-text-2">
        <span className="flex items-center gap-2">
          <span className="h-3 w-5 rounded-full bg-[var(--drv-student)]" aria-hidden="true" />
          {strings.drivers.student.name}
        </span>
        <span className="flex items-center gap-2">
          <span className="h-3 w-5 rounded-full bg-[var(--drv-blind)]" aria-hidden="true" />
          {strings.drivers.blind.name}
        </span>
        <span className="text-text-3">{C.unit}</span>
      </figcaption>
      <ul className="mt-4 flex flex-col divide-y divide-line">
        {rows.map((r, i) => {
          const cond = strings.conditions[r.suite as keyof typeof strings.conditions]
          return (
            <li
              key={r.suite}
              className="grid gap-x-6 gap-y-2 py-3 md:grid-cols-[minmax(180px,240px)_1fr_150px] md:items-center"
            >
              <div>
                <p className="text-[15px] font-[620] text-paint">{cond?.name ?? r.suite}</p>
                <p className="text-[13px] text-text-3">{cond?.what}</p>
              </div>
              <div className="flex flex-col gap-1.5">
                <Bar
                  value={r.adaptive}
                  color="var(--drv-student)"
                  label={C.barLabel(strings.drivers.student.name, r.adaptive)}
                  shown={shown}
                  delay={i * 60}
                />
                <Bar
                  value={r.basic}
                  color="var(--drv-blind)"
                  label={C.barLabel(strings.drivers.blind.name, r.basic)}
                  shown={shown}
                  delay={i * 60 + 30}
                />
              </div>
              <p className="text-[13px] text-text-2">
                <span aria-hidden="true">
                  {r.verdict === 'better' ? '▲ ' : r.verdict === 'worse' ? '▼ ' : '= '}
                </span>
                {C.verdict[r.verdict]}
              </p>
            </li>
          )
        })}
      </ul>
    </figure>
  )
}
