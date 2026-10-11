/**
 * How it works (SPEC 10.2): pipeline diagram, one section per component (what / why / measured /
 * borrowed from), model card, limitations and how to reproduce. Credits per SPEC 17.
 */
import { Link } from 'react-router'
import { REPO_URL, strings } from '../copy/strings.ts'

const H = strings.how
const STAGES = H.pipeline

function Pipeline() {
  const w = 1040
  const box = { w: 150, h: 64 }
  const gap = (w - STAGES.length * box.w) / (STAGES.length - 1)
  return (
    <svg viewBox={`0 0 ${w} 96`} className="w-full" role="img" aria-label={H.pipelineLabel}>
      <defs>
        <marker
          id="arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto"
        >
          <path d="M0 0L10 5L0 10z" fill="var(--text-3)" />
        </marker>
      </defs>
      {STAGES.map((stage, i) => {
        const x = i * (box.w + gap)
        return (
          <g key={stage}>
            <rect
              x={x}
              y={16}
              width={box.w}
              height={box.h}
              rx={10}
              fill="var(--asphalt-800)"
              stroke={i === STAGES.length - 1 ? 'var(--sodium)' : 'var(--line)'}
            />
            <text
              x={x + box.w / 2}
              y={53}
              textAnchor="middle"
              className="fill-[var(--paint)] text-[15px] font-[560]"
            >
              {stage}
            </text>
            {i < STAGES.length - 1 && (
              <line
                x1={x + box.w + 4}
                x2={x + box.w + gap - 6}
                y1={48}
                y2={48}
                stroke="var(--text-3)"
                strokeWidth={1.5}
                markerEnd="url(#arrow)"
              />
            )}
          </g>
        )
      })}
    </svg>
  )
}

export default function How() {
  return (
    <div className="mx-auto max-w-[1280px] px-4 py-12 md:px-6 md:py-16">
      <h1 className="text-[40px] leading-[1.08] font-bold [font-stretch:118%] md:text-[49px]">
        {H.title}
      </h1>
      <p className="mt-3 max-w-[72ch] text-text-2">{H.intro}</p>
      <div className="mt-10 overflow-x-auto">
        <div className="min-w-[640px]">
          <Pipeline />
        </div>
      </div>

      <div className="mt-12 flex flex-col gap-12">
        {H.components.map((c) => (
          <section
            key={c.title}
            className="grid gap-4 border-t border-line pt-6 md:grid-cols-[260px_1fr]"
            aria-labelledby={c.id}
          >
            <h2 id={c.id} className="h3">
              {c.title}
            </h2>
            <dl className="grid max-w-[72ch] gap-4">
              {(
                [
                  [H.labels.what, c.what],
                  [H.labels.why, c.why],
                  [H.labels.measured, c.measured],
                  [H.labels.borrowed, c.borrowed],
                ] as const
              ).map(([k, v]) => (
                <div key={k}>
                  <dt className="label">{k}</dt>
                  <dd className="mt-1 text-text-2">{v}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>

      <section className="mt-16 border-t border-line pt-6" aria-labelledby="card">
        <h2 id="card" className="h2">
          {H.card.title}
        </h2>
        <dl className="mt-6 grid max-w-[900px] gap-x-8 gap-y-4 md:grid-cols-[200px_1fr]">
          {H.card.rows.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="label pt-0.5">{k}</dt>
              <dd className="text-text-2">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-6 text-text-2">
          {H.card.resultsLead} <Link to="/results">{H.card.resultsLink}</Link>
        </p>
      </section>

      <section className="mt-16 border-t border-line pt-6" aria-labelledby="repro">
        <h2 id="repro" className="h2">
          {H.repro.title}
        </h2>
        <p className="mt-3 max-w-[72ch] text-text-2">{H.repro.body}</p>
        <pre className="mt-4 overflow-x-auto rounded-panel border border-line bg-asphalt-800 p-4 text-[13px] leading-6 text-paint">
          <code>{H.repro.commands}</code>
        </pre>
        <p className="mt-4 text-text-2">
          <a href={REPO_URL} className="text-sodium underline">
            {H.repro.repo}
          </a>
        </p>
      </section>
    </div>
  )
}
