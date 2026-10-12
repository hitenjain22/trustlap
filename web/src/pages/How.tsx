/**
 * How it works (SPEC 10.2), as a numbered walk-through: five plain steps, each with one number
 * read from the artifacts and the technical write-up folded underneath. The model card and the
 * reproduction commands sit at the end. Deep link: /how-it-works#step-3.
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import { REPO_URL, strings } from '../copy/strings.ts'
import {
  benchHeadline,
  loadBench,
  loadMetrics,
  loadSimConfig,
  type Bench,
} from '../data/artifacts.ts'
import type { Metrics } from '../data/schemas.ts'
import type { SimConfig } from '../sim/config.ts'
import { accent, STEP_ACCENTS } from '../styles/accents.ts'

const H = strings.how
const STEPS = H.steps

function stepFromHash(): number {
  const n = Number(/step-(\d)/.exec(window.location.hash)?.[1] ?? 1)
  return Math.min(STEPS.length, Math.max(1, n)) - 1
}

export default function How() {
  const [step, setStep] = useState(stepFromHash)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [bench, setBench] = useState<Bench | null>(null)
  const [cfg, setCfg] = useState<SimConfig | null>(null)
  useEffect(() => {
    const onHash = () => setStep(stepFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => {
    loadMetrics().then(setMetrics, () => setMetrics(null))
    loadBench().then(setBench, () => setBench(null))
    loadSimConfig().then(setCfg, () => setCfg(null))
  }, [])
  const go = (i: number) => {
    setStep(i)
    window.history.replaceState(null, '', `#step-${i + 1}`)
  }

  // The one number per step, from metrics.json / bench.json / config.json (null = not available).
  const h = metrics?.headline
  const bh = bench ? benchHeadline(bench) : null
  const stat = (() => {
    switch (step) {
      case 0:
        return bh
          ? STEPS[0].stat(
              bh.maxCars.toLocaleString('en-US'),
              Math.round(bh.speedup).toLocaleString('en-US'),
            )
          : null
      case 1:
        return h?.train_env_steps_total != null && h.train_wall_minutes != null
          ? STEPS[1].stat(
              h.train_env_steps_total >= 1e9
                ? `${(h.train_env_steps_total / 1e9).toFixed(1)} billion`
                : `${Math.round(h.train_env_steps_total / 1e6)} million`,
              Math.round(h.train_wall_minutes).toString(),
            )
          : null
      case 2:
        return h?.grip_mae != null && cfg
          ? STEPS[2].stat(
              h.grip_mae.toFixed(2),
              cfg.conditions.mu_dry.nominal.toFixed(1),
              cfg.conditions.mu_dry.ood.toFixed(2),
            )
          : null
      case 3:
        return metrics?.calibration?.ece != null
          ? STEPS[3].stat(metrics.calibration.ece.toFixed(3))
          : null
      case 4:
        return metrics?.protocol?.test_tracks && metrics.suites
          ? STEPS[4].stat(
              String(metrics.protocol.test_tracks),
              String(Object.keys(metrics.suites).length),
            )
          : null
      default:
        return null
    }
  })()
  const s = STEPS[step]!
  const details = H.components.filter((c) => (s.components as readonly string[]).includes(c.id))

  return (
    <div className="mx-auto max-w-[1080px] px-4 py-12 md:px-6 md:py-16">
      <h1 className="text-[40px] leading-[1.08] font-bold [font-stretch:118%] md:text-[49px]">
        {H.title}
      </h1>
      <p className="mt-3 max-w-[68ch] text-text-2">{H.lead}</p>

      <ol className="mt-10 grid grid-cols-2 gap-2 sm:grid-cols-5" aria-label={H.title}>
        {STEPS.map((x, i) => (
          <li key={x.tab}>
            <button
              type="button"
              aria-current={i === step ? 'step' : undefined}
              onClick={() => go(i)}
              style={accent(STEP_ACCENTS[i]!)}
              className={`card flex min-h-12 w-full items-center gap-2 px-3 text-left text-[14px] font-[560] transition-[border-color,color,transform] duration-150 hover:-translate-y-px ${
                i === step ? 'card--selected text-paint' : 'text-text-2 hover:text-paint'
              }`}
            >
              <span
                className="grid size-6 shrink-0 place-items-center rounded-full text-[12px] font-[650]"
                style={{
                  background: i === step ? STEP_ACCENTS[i] : 'var(--asphalt-700)',
                  color: i === step ? 'var(--sodium-ink)' : 'var(--text-2)',
                }}
              >
                {i + 1}
              </span>
              {x.tab}
            </button>
          </li>
        ))}
      </ol>

      <section
        className="card card--accent mt-8 p-6 md:p-8"
        style={accent(STEP_ACCENTS[step]!)}
        aria-live="polite"
      >
        <p className="eyebrow">{`Step ${step + 1} of ${STEPS.length}`}</p>
        <h2 className="mt-2 text-[28px] leading-tight font-bold text-paint">{s.title}</h2>
        <div className="mt-4 grid gap-8 md:grid-cols-[1fr_260px]">
          <div className="flex max-w-[62ch] flex-col gap-3 text-[16px] text-text-2">
            {s.body.map((p) => (
              <p key={p}>{p}</p>
            ))}
            {'link' in s && s.link && (
              <p>
                <Link to={s.link.to} className="text-sodium underline">
                  {s.link.label}
                </Link>
              </p>
            )}
          </div>
          {stat && (
            <div className="border-line md:border-l md:pl-6">
              <p className="numeral text-[34px] leading-tight text-paint">{stat.value}</p>
              <p className="mt-2 text-[14px] text-text-2">{stat.meaning}</p>
            </div>
          )}
        </div>

        {details.length > 0 && (
          <details className="group mt-8 border-t border-line pt-4">
            <summary className="cursor-pointer list-none text-[15px] font-[620] text-paint">
              {H.detailsLabel}
            </summary>
            <div className="mt-4 flex flex-col gap-8">
              {details.map((c) => (
                <div key={c.id}>
                  <h3 className="h3">{c.title}</h3>
                  <dl className="mt-3 grid max-w-[72ch] gap-3">
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
                        <dd className="mt-1 text-[15px] text-text-2">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ))}
            </div>
          </details>
        )}

        {step < STEPS.length - 1 && (
          <div className="mt-8">
            <button type="button" className="btn-primary" onClick={() => go(step + 1)}>
              {H.next}: {STEPS[step + 1]!.tab}
            </button>
          </div>
        )}
      </section>

      <details className="card group mt-12">
        <summary className="cursor-pointer list-none px-5 py-4 text-[18px] font-[620] text-paint">
          {H.moreTitle}
        </summary>
        <div className="border-t border-line px-5 pb-8">
          <h2 id="card" className="h2 mt-6">
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
          <h2 id="repro" className="h2 mt-10">
            {H.repro.title}
          </h2>
          <p className="mt-3 max-w-[72ch] text-text-2">{H.repro.body}</p>
          <pre className="mt-4 overflow-x-auto rounded-panel border border-line bg-asphalt p-4 text-[13px] leading-6 text-paint">
            <code>{H.repro.commands}</code>
          </pre>
          <p className="mt-4 text-text-2">
            <a href={REPO_URL} className="text-sodium underline">
              {H.repro.repo}
            </a>
          </p>
        </div>
      </details>
    </div>
  )
}
