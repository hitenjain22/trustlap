/**
 * Playground telemetry for the focused car (SPEC 10.2): speed, lap times, grip gauges (true
 * tick vs estimate bar), crash-risk meter with the governor threshold, 10 s sparklines and
 * input bars. Text uses text tokens; color marks carry meaning (grip blue, AI sodium).
 */
import { useState } from 'react'
import { strings } from '../copy/strings.ts'
import type { Telemetry } from '../hooks/useLiveSim.ts'
import type { DriverKind } from '../worker/protocol.ts'

const T = strings.telemetry
const MU_MAX = 1.3
const TOP_SPEED_MPS = 70
const reduced = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

const fmtLap = (s: number) => {
  if (!Number.isFinite(s)) return '–'
  const m = Math.floor(s / 60)
  const sec = s - m * 60
  return `${m}:${sec.toFixed(2).padStart(5, '0')}`
}

function Sparkline({
  series,
  max,
  label,
}: {
  series: { values: number[]; color: string; dash?: string }[]
  max: number
  label: string
}) {
  const w = 160
  const h = 36
  return (
    <figure className="min-w-0">
      <svg
        width="100%"
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="none"
        className="block h-9"
        aria-hidden="true"
      >
        <line x1={0} x2={w} y1={h - 0.5} y2={h - 0.5} stroke="var(--asphalt-700)" />
        {series.map((s, i) => {
          const pts = s.values
            .map((v, k) =>
              Number.isFinite(v)
                ? `${(k / 99) * w},${h - (Math.min(v, max) / max) * (h - 2) - 1}`
                : null,
            )
            .filter(Boolean)
            .join(' ')
          return pts ? (
            <polyline
              key={i}
              points={pts}
              fill="none"
              stroke={s.color}
              strokeWidth={1.5}
              strokeDasharray={s.dash}
              vectorEffect="non-scaling-stroke"
            />
          ) : null
        })}
      </svg>
      <figcaption className="mt-1 text-[12px] text-text-3">{label}</figcaption>
    </figure>
  )
}

function GripGauge({ label, truth, estimate }: { label: string; truth: number; estimate: number }) {
  const pct = (v: number) => `${(Math.max(0, Math.min(v, MU_MAX)) / MU_MAX) * 100}%`
  const hasEst = Number.isFinite(estimate)
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 text-[13px]">
        <span className="text-text-2">{label}</span>
        <span className="numeral text-paint">
          {hasEst
            ? T.gripReading(estimate.toFixed(2), truth.toFixed(2))
            : T.gripTrueOnly(truth.toFixed(2))}
        </span>
      </div>
      <div
        className="relative mt-1 h-3 rounded-full bg-asphalt-700"
        role="img"
        aria-label={
          hasEst
            ? T.gripAria(label, estimate.toFixed(2), truth.toFixed(2))
            : T.gripTrueOnly(truth.toFixed(2))
        }
      >
        {hasEst && (
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-sodium"
            style={{ width: pct(estimate) }}
          />
        )}
        <div
          className="absolute -inset-y-1 w-[3px] rounded bg-grip-2"
          style={{ left: `calc(${pct(truth)} - 1.5px)` }}
        />
      </div>
    </div>
  )
}

/** Phone strip (SPEC 10.2 mobile): speed, rear grip estimate vs truth, crash risk. */
export function TelemetryStrip({ t }: { t: Telemetry | null }) {
  if (!t) return <div className="h-12 px-4 py-3 text-[13px] text-text-3">{T.waiting}</div>
  const est = Number.isFinite(t.muHatRear) ? t.muHatRear.toFixed(2) : '–'
  return (
    <dl
      className="grid grid-cols-3 gap-3 border-t border-line px-4 py-2 text-[13px]"
      aria-label={T.title}
    >
      <div>
        <dt className="text-text-3">{T.speedShort}</dt>
        <dd className="numeral text-[20px] leading-tight text-paint">
          {Math.round(t.speedMps * 3.6)} <span className="text-[12px] text-text-2">km/h</span>
        </dd>
      </div>
      <div>
        <dt className="text-text-3">{T.rear}</dt>
        <dd className="numeral leading-tight text-paint">
          {T.gripShort(est, t.muRear.toFixed(2))}
        </dd>
      </div>
      <div>
        <dt className="text-text-3">{T.risk}</dt>
        <dd className="numeral text-[20px] leading-tight text-paint">
          {Number.isFinite(t.pCrash) ? `${Math.round(t.pCrash * 100)}%` : '–'}
        </dd>
      </div>
    </dl>
  )
}

export function TelemetryPanel({
  t,
  tau,
  governorOn,
}: {
  t: Telemetry | null
  tau: number | null
  governorOn: boolean
}) {
  const [mph, setMph] = useState(false)
  const [more, setMore] = useState(false)
  if (!t) return <div className="h-[168px] px-4 py-4 text-text-3 md:px-6">{T.waiting}</div>
  const speed = mph ? t.speedMps * 2.23694 : t.speedMps * 3.6
  const stretch = reduced() ? 100 : 75 + 50 * Math.min(1, t.speedMps / TOP_SPEED_MPS)
  const delta = t.lastLap - t.ghostLastLap
  const risk = t.pCrash
  return (
    <section
      className={`grid gap-x-8 gap-y-5 border-t border-line px-4 py-4 md:px-6 ${more ? 'md:grid-cols-[minmax(170px,auto)_1fr_1fr]' : 'md:grid-cols-[minmax(170px,auto)_1fr]'}`}
      aria-label={T.title}
    >
      <div>
        <div className="flex items-baseline gap-2">
          <span
            className="numeral text-[44px] leading-none text-paint"
            style={{ fontStretch: `${stretch}%` }}
          >
            {Math.round(speed)}
          </span>
          <button
            type="button"
            className="text-[13px] text-text-2 underline"
            onClick={() => setMph((m) => !m)}
            aria-label={T.unitToggle}
          >
            {mph ? 'mph' : 'km/h'}
          </button>
        </div>
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[13px]">
          <dt className="text-text-3">{T.lap}</dt>
          <dd className="numeral text-paint">{t.crashed ? T.crashed : fmtLap(t.lapTime)}</dd>
          <dt className="text-text-3">{T.best}</dt>
          <dd className="numeral text-paint">{fmtLap(t.bestLap)}</dd>
          <dt className="text-text-3">{strings.playground.crashes}</dt>
          <dd className="text-paint">
            {Object.entries(t.crashesByDriver)
              .map(([k, v]) => `${strings.playground.drivers[k as DriverKind]} ${v}`)
              .join(' · ')}
          </dd>
          {Number.isFinite(delta) && (
            <>
              <dt className="text-text-3">{T.delta}</dt>
              <dd className="numeral text-paint">
                {delta > 0 ? '+' : '−'}
                {Math.abs(delta).toFixed(2)} s
              </dd>
            </>
          )}
        </dl>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className="text-[13px] text-text-3">{T.gripKey}</p>
          <button
            type="button"
            className="text-[13px] text-sodium underline"
            aria-expanded={more}
            onClick={() => setMore((m) => !m)}
          >
            {more ? T.lessTelemetry : T.moreTelemetry}
          </button>
        </div>
        <GripGauge label={T.front} truth={t.muFront} estimate={t.muHatFront} />
        <GripGauge label={T.rear} truth={t.muRear} estimate={t.muHatRear} />
        <div>
          <div className="flex items-baseline justify-between gap-2 text-[13px]">
            <span className="text-text-2">{T.risk}</span>
            <span className="numeral text-paint">
              {Number.isFinite(risk) ? `${Math.round(risk * 100)}%` : '–'}
              {tau !== null && Number.isFinite(risk) && (
                <span className="ml-2 text-text-3">
                  {governorOn ? T.slowsAbove(Math.round(tau * 100)) : T.governorOff}
                </span>
              )}
            </span>
          </div>
          <div className="relative mt-1 h-3 rounded-full bg-asphalt-700" aria-hidden="true">
            {Number.isFinite(risk) && (
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-sodium/80"
                style={{ width: `${Math.min(1, risk) * 100}%` }}
              />
            )}
            {tau !== null && governorOn && (
              <div
                className="absolute -inset-y-1 w-[2px] bg-paint"
                style={{ left: `${tau * 100}%` }}
              />
            )}
          </div>
          {!Number.isFinite(risk) && (
            <p className="mt-1 text-[12px] text-text-3">{T.riskStudentOnly}</p>
          )}
        </div>
      </div>

      {more && (
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-3 gap-3">
            <Sparkline
              series={[{ values: t.history.speed, color: 'var(--paint)' }]}
              max={TOP_SPEED_MPS}
              label={T.sparkSpeed}
            />
            <Sparkline
              series={[
                { values: t.history.muRear, color: 'var(--grip-3)' },
                { values: t.history.muHatRear, color: 'var(--sodium)' },
              ]}
              max={MU_MAX}
              label={T.sparkGrip}
            />
            <Sparkline
              series={[{ values: t.history.pCrash, color: 'var(--sodium)' }]}
              max={1}
              label={T.sparkRisk}
            />
          </div>
          <div className="grid grid-cols-2 gap-4 text-[12px] text-text-3">
            <div>
              <div
                className="relative h-2 rounded-full bg-asphalt-700"
                role="img"
                aria-label={T.steerAria(Math.round(t.steer * 100))}
              >
                <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <div
                  className="absolute inset-y-0 rounded-full bg-paint/80"
                  style={{
                    left: `${50 + Math.min(0, -t.steer) * 50}%`,
                    width: `${Math.abs(t.steer) * 50}%`,
                  }}
                />
              </div>
              <p className="mt-1">{T.steering}</p>
            </div>
            <div>
              <div
                className="relative h-2 rounded-full bg-asphalt-700"
                role="img"
                aria-label={T.pedalAria(Math.round(t.long * 100))}
              >
                <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <div
                  className={`absolute inset-y-0 rounded-full ${t.long >= 0 ? 'bg-paint/80' : 'bg-text-3'}`}
                  style={{
                    left: `${t.long >= 0 ? 50 : 50 + t.long * 50}%`,
                    width: `${Math.abs(t.long) * 50}%`,
                  }}
                />
              </div>
              <p className="mt-1">{T.pedals}</p>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
