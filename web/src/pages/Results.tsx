/**
 * Results (SPEC 10.2): headline stats and every figure, all read from metrics.json / bench.json.
 * A missing metric shows "Not computed yet", never a number we didn't measure.
 */
import { useEffect, useState } from 'react'
import { ChartFrame, type TableData } from '../charts/ChartFrame.tsx'
import { DotPlot, type DotDatum } from '../charts/DotPlot.tsx'
import { DRIVER_STYLE, SUITE_LABEL, type ChartDriver } from '../charts/drivers.ts'
import { LineChart, type Series } from '../charts/LineChart.tsx'
import { strings } from '../copy/strings.ts'
import { benchHeadline, loadBench, loadMetrics, type Bench } from '../data/artifacts.ts'
import type { DriverSuite, Estimate, Metrics } from '../data/schemas.ts'
import { CompareBars } from '../charts/CompareBars.tsx'
import { accent } from '../styles/accents.ts'
import {
  bestAiLaps,
  biggest,
  compareConditions,
  TRAINED_CONDITIONS,
  UNTRAINED_CONDITIONS,
} from '../data/summary.ts'

const IN_DIST = ['dry', 'wet', 'puddles', 'wind', 'worn', 'laggy', 'heavy', 'combined']
const OOD = ['ice', 'extreme_lag', 'noisy_sensors', 'very_heavy']
const MAIN_DRIVERS: ChartDriver[] = ['student', 'blind', 'teacher', 'scripted']
const S = strings.results
const SS = strings.results.simple
const condName = (s: string) => strings.conditions[s as keyof typeof strings.conditions]?.name ?? s
/** "a, b and c" */
const list = (xs: string[]) =>
  xs.length <= 1 ? (xs[0] ?? '') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`

function SummaryCard({
  label,
  value,
  note,
  tint,
}: {
  label: string
  value: string
  note: string
  tint: string
}) {
  return (
    <div className="card card--accent p-5" style={accent(tint)}>
      <p className="eyebrow">{label}</p>
      <p className="numeral mt-2 text-[30px] leading-tight text-paint">{value}</p>
      <p className="mt-2 text-[14px] text-text-2">{note}</p>
    </div>
  )
}

const pct = (v: number) => `${Math.round(v * 100)}%`
const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`
const pp = (v: number) => `${v > 0 ? '+' : ''}${Math.round(v * 100)}`
const secs = (v: number) => `${v.toFixed(1)} s`
const fmtCi = (e: Estimate, f: (v: number) => string) =>
  e.value === null
    ? '–'
    : `${f(e.value)} (${e.ci[0] == null ? '–' : f(e.ci[0])} to ${e.ci[1] == null ? '–' : f(e.ci[1])})`

function finite(
  e: Estimate | null | undefined,
): e is Estimate & { value: number; ci: [number, number] } {
  return !!e && e.value !== null && e.ci[0] !== null && e.ci[1] !== null && Number.isFinite(e.value)
}

function dots(
  m: Metrics,
  suites: string[],
  drivers: ChartDriver[],
  key: keyof DriverSuite,
): DotDatum[] {
  const out: DotDatum[] = []
  for (const s of suites) {
    for (const d of drivers) {
      const e = m.suites?.[s]?.[d]?.[key] as Estimate | null | undefined
      if (finite(e))
        out.push({ row: SUITE_LABEL[s] ?? s, driver: d, value: e.value, ci: [e.ci[0], e.ci[1]] })
    }
  }
  return out
}

function table(
  m: Metrics,
  suites: string[],
  drivers: ChartDriver[],
  key: keyof DriverSuite,
  f: (v: number) => string,
): TableData {
  return {
    columns: ['Condition', ...drivers.map((d) => DRIVER_STYLE[d].label)],
    rows: suites.map((s) => [
      SUITE_LABEL[s] ?? s,
      ...drivers.map((d) => {
        const e = m.suites?.[s]?.[d]?.[key] as Estimate | null | undefined
        return e ? fmtCi(e, f) : '–'
      }),
    ]),
  }
}

function StatTile({ label, value, note }: { label: string; value: string | null; note: string }) {
  return (
    <div className="card p-5">
      <p className="label">{label}</p>
      <p className="numeral mt-2 text-[36px] leading-none text-paint">{value ?? '–'}</p>
      <p className="mt-2 text-[13px] text-text-2">{value === null ? S.notComputed : note}</p>
    </div>
  )
}

export default function Results() {
  const [metrics, setMetrics] = useState<Metrics | null | undefined>(undefined)
  const [bench, setBench] = useState<Bench | null>(null)
  useEffect(() => {
    loadMetrics().then(setMetrics, (e) => (console.error(e), setMetrics(null)))
    loadBench().then(setBench, () => setBench(null))
  }, [])

  if (metrics === undefined)
    return <section className="mx-auto max-w-[1280px] px-4 py-16 md:px-6" aria-busy="true" />
  const m: Metrics = metrics ?? { schema_version: 1, generated_utc: '' }
  const h = m.headline ?? {}
  const p = m.protocol ?? {}
  const source = S.source(m.generated_utc.slice(0, 10), m.git_sha ?? '–')
  const bh = bench ? benchHeadline(bench) : null

  // Figure 1: completion, sorted by student − blind.
  const delta = (s: string) => {
    const a = m.suites?.[s]?.student?.completion?.value
    const b = m.suites?.[s]?.blind?.completion?.value
    return a != null && b != null ? a - b : -Infinity
  }
  const sorted = [...IN_DIST].filter((s) => m.suites?.[s]).sort((a, b) => delta(b) - delta(a))
  const wins = sorted.filter((s) => delta(s) > 0).length
  const comp = dots(m, sorted, MAIN_DRIVERS, 'completion')

  // Figure 2: paired deltas student − blind.
  const pd = (m.paired_deltas ?? []).filter(
    (d) => d.a === 'student' && d.b === 'blind' && d.metric === 'completion',
  )
  const pdDots: DotDatum[] = pd
    .filter(
      (d) => d.value !== null && d.ci[0] != null && d.ci[1] != null && IN_DIST.includes(d.suite),
    )
    .map((d) => ({
      row: SUITE_LABEL[d.suite] ?? d.suite,
      driver: 'student',
      value: d.value!,
      ci: [d.ci[0]!, d.ci[1]!],
    }))
  const clearlyBetter = pdDots.filter((d) => d.ci[0] > 0).length

  // Figure 3–4.
  const lap = dots(
    m,
    sorted,
    ['student', 'student_governed', 'blind', 'teacher', 'scripted'],
    'paired_lap_time_s',
  )
  const lapVals = lap.flatMap((d) => d.ci)
  const ood = dots(
    m,
    OOD.filter((s) => m.suites?.[s]),
    MAIN_DRIVERS,
    'completion',
  )

  // Figure 5: puddle step.
  const ps = m.puddle_step
  const lat = h.puddle_latency_ms
  const puddleSeries: Series[] = ps
    ? [
        {
          id: 'true',
          label: 'True grip',
          color: 'var(--grip-3)',
          step: true,
          points: ps.t_ms.map((t, i) => ({ x: t, y: ps.mu_true[i]! })),
        },
        {
          id: 'hat',
          label: 'Student estimate',
          color: 'var(--drv-student)',
          points: ps.t_ms.map((t, i) => ({ x: t, y: ps.mu_hat[i]! })),
        },
      ]
    : []

  // Figure 6: reliability.
  const cal = m.calibration
  const relSeries: Series[] = cal?.bins
    ? [
        {
          id: 'rel',
          label: 'Student ensemble',
          color: 'var(--drv-student)',
          markers: true,
          points: cal.bins.map((b) => ({
            x: b.p_pred,
            y: b.p_obs,
            note: `${b.n.toLocaleString('en-US')} steps`,
          })),
        },
      ]
    : []
  const relMax = cal?.bins ? Math.max(0.05, ...cal.bins.flatMap((b) => [b.p_pred, b.p_obs])) : 1

  // Figure 7: governor sweep.
  const sweep = (m.governor_sweep ?? []).filter(
    (r) => finite(r.crash) && finite(r.paired_lap_time_s),
  )
  // Claim fewer crashes only if some threshold's crash interval sits wholly below "governor off".
  const govOff = sweep.find((r) => r.tau === null)
  const govHelps =
    govOff != null && sweep.some((r) => r.tau !== null && r.crash.ci[1]! < govOff.crash.value!)
  const govSeries: Series[] = sweep.length
    ? [
        {
          id: 'gov',
          label: 'Student + governor',
          color: 'var(--drv-student)',
          markers: true,
          points: [...sweep]
            .sort((a, b) => a.paired_lap_time_s.value! - b.paired_lap_time_s.value!)
            .map((r) => ({
              x: r.paired_lap_time_s.value!,
              y: r.crash.value!,
              xCi: [r.paired_lap_time_s.ci[0]!, r.paired_lap_time_s.ci[1]!] as [number, number],
              yCi: [r.crash.ci[0]!, r.crash.ci[1]!] as [number, number],
              note:
                r.tau === null
                  ? 'governor off'
                  : `τ = ${pct(r.tau)}${r.tau === cal?.tau_default ? ' (chosen)' : ''}`,
            })),
        },
      ]
    : []

  // Figure 8: benchmark.
  const impls = bench ? [...new Set(bench.rows.map((r) => r.impl))] : []
  const benchColor = (impl: string) =>
    impl === 'jax-cpu' ? 'var(--text-2)' : impl === 'numpy-loop' ? 'var(--text-3)' : 'var(--sodium)'
  const benchLabel = (impl: string) =>
    impl === 'jax-cpu'
      ? 'JAX on CPU'
      : impl === 'numpy-loop'
        ? 'Python loop (NumPy)'
        : `JAX on ${bench?.device.replace(/^Tesla /, '')} GPU`
  const benchSeries: Series[] = bench
    ? impls.map((impl) => ({
        id: impl,
        label: benchLabel(impl),
        color: benchColor(impl),
        dash: impl === 'numpy-loop' ? '6 4' : impl === 'jax-cpu' ? '2 4' : undefined,
        markers: true,
        points: bench.rows
          .filter((r) => r.impl === impl)
          .map((r) => ({ x: r.n_cars, y: r.steps_per_s })),
      }))
    : []

  // Figure 9: generalization.
  const gen = m.generalization
  const genDots: DotDatum[] = gen
    ? (['student', 'blind', 'teacher'] as ChartDriver[]).flatMap((d) =>
        (['train_bank', 'test_bank'] as const)
          .map((bankKey) => ({ bankKey, e: gen[bankKey][d]?.completion }))
          .filter(
            (
              x,
            ): x is {
              bankKey: 'train_bank' | 'test_bank'
              e: Estimate & { value: number; ci: [number, number] }
            } => finite(x.e),
          )
          .map(({ bankKey, e }) => ({
            row: bankKey === 'train_bank' ? 'Training tracks' : 'Unseen test tracks',
            driver: d,
            value: e.value,
            ci: e.ci,
          })),
      )
    : []

  const red = h.student_crash_reduction_vs_blind_combined
  // The teacher gap (t − s) / (t − b) only means something when the teacher beats the blind driver.
  const tComb = m.suites?.combined?.teacher?.completion?.value
  const bComb = m.suites?.combined?.blind?.completion?.value
  const teacherBeatsBlind = tComb != null && bComb != null && tComb > bComb
  const gap = h.teacher_gap_combined

  // The newcomer's view: one comparison, a few cards and plain takeaways (all from metrics.json).
  const rows = compareConditions(m, TRAINED_CONDITIONS)
  const gain = biggest(rows, 'better')
  const loss = biggest(rows, 'worse')
  const names = (v: 'better' | 'worse' | 'same') =>
    rows.filter((r) => r.verdict === v).map((r) => condName(r.suite).toLowerCase())
  const failing = UNTRAINED_CONDITIONS.filter((c) => {
    const best = bestAiLaps(m, c)
    return best !== null && best <= 15
  })
  const takeaways = [
    names('better').length ? SS.better(list(names('better'))) : null,
    names('worse').length ? SS.worse(list(names('worse'))) : null,
    names('same').length ? SS.same(list(names('same'))) : null,
    failing.length
      ? SS.untrained(
          list(failing.map((c) => condName(c).toLowerCase())),
          Math.max(...failing.map((c) => bestAiLaps(m, c)!)),
        )
      : null,
    lat?.noticed === 0 && lat.trials ? SS.puddle(lat.trials) : null,
    m.calibration?.ece != null &&
    m.calibration.ece < 0.05 &&
    m.calibration.auroc != null &&
    m.calibration.auroc < 0.7
      ? SS.risk
      : null,
  ].filter((t): t is string => t !== null)
  return (
    <div className="mx-auto max-w-[1280px] px-4 py-12 md:px-6 md:py-16">
      <p className="eyebrow">{S.title}</p>
      <h1 className="mt-2 text-[40px] leading-[1.08] font-bold [font-stretch:118%] md:text-[49px]">
        {SS.question}
      </h1>
      <p className="mt-3 max-w-[72ch] text-text-2">
        {p.test_tracks ? SS.intro(p.test_tracks) : S.introMissing}
      </p>

      {rows.length > 0 && (
        <>
          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {finite(red) && (
              <SummaryCard
                tint="var(--violet)"
                label={SS.cards.combinedLabel}
                value={
                  red.value >= 0
                    ? SS.cards.fewer(Math.round(red.value * 100))
                    : SS.cards.more(Math.round(-red.value * 100))
                }
                note={
                  red.value >= 0
                    ? SS.cards.combinedNote(
                        Math.round(red.ci[0] * 100),
                        Math.round(red.ci[1] * 100),
                      )
                    : SS.cards.combinedNote(
                        Math.round(-red.ci[1] * 100),
                        Math.round(-red.ci[0] * 100),
                      )
                }
              />
            )}
            {gain && (
              <SummaryCard
                tint="var(--teal)"
                label={SS.cards.gainLabel(condName(gain.suite))}
                value={`+${gain.adaptive - gain.basic} ${SS.cards.perHundred}`}
                note={SS.cards.lapsNote(gain.adaptive, gain.basic)}
              />
            )}
            {loss && (
              <SummaryCard
                tint="var(--coral)"
                label={SS.cards.lossLabel(condName(loss.suite))}
                value={`−${loss.basic - loss.adaptive} ${SS.cards.perHundred}`}
                note={SS.cards.lapsNote(loss.adaptive, loss.basic)}
              />
            )}
          </div>

          <section className="mt-14" aria-labelledby="simple-chart">
            <h2 id="simple-chart" className="h2">
              {SS.chartTitle}
            </h2>
            <div className="mt-6">
              <CompareBars rows={rows} />
            </div>
            {p.laps_per_driver_per_suite && (
              <p className="mt-3 text-[13px] text-text-3">
                {SS.chartNote(p.laps_per_driver_per_suite)}
              </p>
            )}
          </section>

          <section className="mt-14" aria-labelledby="takeaways">
            <h2 id="takeaways" className="h2">
              {SS.takeawaysTitle}
            </h2>
            <ul className="mt-4 flex max-w-[72ch] list-disc flex-col gap-2 pl-5 text-text-2">
              {takeaways.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </section>
        </>
      )}

      <details className="card group mt-16">
        <summary className="cursor-pointer list-none px-5 py-4">
          <span className="text-[18px] font-[620] text-paint">{SS.fullTitle}</span>
          <span className="mt-1 block text-[14px] text-text-2">{SS.fullSummary}</span>
        </summary>
        <div className="border-t border-line px-5 pb-8">
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <StatTile
              label={finite(red) && red.value < 0 ? S.tiles.crashIncrease : S.tiles.crashReduction}
              value={finite(red) ? pct(Math.abs(red.value)) : null}
              note={
                finite(red)
                  ? red.value < 0
                    ? `${S.tiles.crashReductionNote}. 95% CI ${pct(-red.ci[1])} to ${pct(-red.ci[0])} more.`
                    : `${S.tiles.crashReductionNote}. 95% CI ${pct(red.ci[0])} to ${pct(red.ci[1])}.`
                  : ''
              }
            />
            <StatTile
              label={S.tiles.teacherGap}
              value={finite(gap) && teacherBeatsBlind ? pct(gap.value) : finite(gap) ? 'n/a' : null}
              note={
                finite(gap) && !teacherBeatsBlind
                  ? S.tiles.teacherGapUndefined
                  : finite(gap)
                    ? `${S.tiles.teacherGapNote}. 95% CI ${pct(gap.ci[0])} to ${pct(gap.ci[1])}.`
                    : ''
              }
            />
            <StatTile
              label={S.tiles.gripMae}
              value={h.grip_mae != null ? `μ ± ${h.grip_mae.toFixed(3)}` : null}
              note={S.tiles.gripMaeNote}
            />
            <StatTile
              label={S.tiles.latency}
              value={
                lat?.median != null
                  ? `${Math.round(lat.median)} ms`
                  : lat?.noticed === 0 && lat.trials
                    ? S.tiles.notNoticed
                    : null
              }
              note={
                lat?.iqr && lat.noticed != null && lat.trials && lat.noticed < lat.trials
                  ? S.tiles.noticedSomeNote(
                      lat.noticed,
                      lat.trials,
                      Math.round(lat.iqr[0]!),
                      Math.round(lat.iqr[1]!),
                    )
                  : lat?.iqr
                    ? S.tiles.latencyNote(Math.round(lat.iqr[0]!), Math.round(lat.iqr[1]!))
                    : lat?.noticed === 0 && lat.trials
                      ? S.tiles.notNoticedNote(0, lat.trials)
                      : ''
              }
            />
            <StatTile
              label={S.tiles.cars}
              value={bh ? bh.maxCars.toLocaleString('en-US') : null}
              note={bh ? S.tiles.carsNote(Math.round(bh.speedup).toLocaleString('en-US')) : ''}
            />
            <StatTile
              label={S.tiles.compute}
              value={
                h.train_env_steps_total != null
                  ? h.train_env_steps_total >= 1e9
                    ? `${(h.train_env_steps_total / 1e9).toFixed(1)}B steps`
                    : `${Math.round(h.train_env_steps_total / 1e6)}M steps`
                  : null
              }
              note={
                h.train_wall_minutes != null
                  ? S.tiles.computeNote(
                      Math.round(h.train_wall_minutes).toString(),
                      h.train_hardware ?? 'CPU',
                    )
                  : ''
              }
            />
          </div>

          <div className="mt-16 flex flex-col gap-16">
            <h2 className="sr-only">{S.figuresHeading}</h2>
            <ChartFrame
              title={
                comp.length
                  ? S.figures.completion.title(wins, sorted.length)
                  : S.figures.completion.neutral
              }
              subtitle={S.figures.completion.subtitle}
              summary={`Completion rates by condition for ${MAIN_DRIVERS.length} drivers. The student beats the blind driver in ${wins} of ${sorted.length} conditions.`}
              drivers={MAIN_DRIVERS}
              table={m.suites ? table(m, sorted, MAIN_DRIVERS, 'completion', pct1) : null}
              source={source}
              missing={comp.length === 0}
            >
              <DotPlot
                rows={sorted.map((s) => SUITE_LABEL[s] ?? s)}
                data={comp}
                domain={[0, 1]}
                format={pct}
                axisLabel={S.figures.completion.axis}
              />
            </ChartFrame>

            <ChartFrame
              title={
                pdDots.length
                  ? S.figures.delta.title(clearlyBetter, pdDots.length)
                  : S.figures.delta.neutral
              }
              subtitle={S.figures.delta.subtitle}
              summary={`Paired completion difference, student minus blind; clearly positive in ${clearlyBetter} of ${pdDots.length} conditions.`}
              table={{
                columns: ['Condition', 'Student − blind (pp)'],
                rows: pd
                  .filter((d) => IN_DIST.includes(d.suite))
                  .map((d) => [
                    SUITE_LABEL[d.suite] ?? d.suite,
                    `${d.value === null ? '–' : pp(d.value)} (${d.ci[0] == null ? '–' : pp(d.ci[0])} to ${d.ci[1] == null ? '–' : pp(d.ci[1])})`,
                  ]),
              }}
              source={source}
              missing={pdDots.length === 0}
            >
              <DotPlot
                rows={pdDots.map((d) => d.row)}
                data={pdDots}
                domain={[
                  Math.min(-0.1, ...pdDots.map((d) => d.ci[0])),
                  Math.max(0.1, ...pdDots.map((d) => d.ci[1])),
                ]}
                format={pp}
                axisLabel={S.figures.delta.axis}
                zero
              />
            </ChartFrame>

            <ChartFrame
              title={lap.length ? S.figures.lapTime.title : S.figures.lapTime.neutral}
              subtitle={S.figures.lapTime.subtitle}
              summary="Paired lap times by condition for each driver."
              drivers={['student', 'student_governed', 'blind', 'teacher', 'scripted']}
              table={
                m.suites
                  ? table(
                      m,
                      sorted,
                      ['student', 'student_governed', 'blind', 'teacher', 'scripted'],
                      'paired_lap_time_s',
                      secs,
                    )
                  : null
              }
              source={source}
              missing={lap.length === 0}
            >
              <DotPlot
                rows={sorted.map((s) => SUITE_LABEL[s] ?? s)}
                data={lap}
                domain={[Math.min(...lapVals) * 0.95, Math.max(...lapVals) * 1.02]}
                format={(v) => `${Math.round(v)} s`}
                axisLabel={S.figures.lapTime.axis}
              />
            </ChartFrame>

            <ChartFrame
              title={ood.length ? S.figures.ood.title : S.figures.ood.neutral}
              subtitle={S.figures.ood.subtitle}
              summary="Completion rates under conditions the drivers never trained on."
              drivers={MAIN_DRIVERS}
              table={
                m.suites
                  ? table(
                      m,
                      OOD.filter((s) => m.suites?.[s]),
                      MAIN_DRIVERS,
                      'completion',
                      pct1,
                    )
                  : null
              }
              source={source}
              missing={ood.length === 0}
            >
              <DotPlot
                rows={OOD.filter((s) => m.suites?.[s]).map((s) => SUITE_LABEL[s] ?? s)}
                data={ood}
                domain={[0, 1]}
                format={pct}
                axisLabel={S.figures.ood.axis}
              />
            </ChartFrame>

            <ChartFrame
              title={
                lat?.median != null
                  ? S.figures.puddle.title(Math.round(lat.median), lat.noticed, lat.trials)
                  : lat?.noticed === 0 && lat.trials
                    ? S.figures.puddle.titleNotNoticed(0, lat.trials)
                    : S.figures.puddle.titleMissing
              }
              subtitle={S.figures.puddle.subtitle}
              summary="Line chart: true rear grip drops as the car enters a puddle; the student's estimate follows it with a short lag."
              table={
                ps
                  ? {
                      columns: ['Time (ms)', 'True μ', 'Estimated μ'],
                      rows: ps.t_ms.map((t, i) => [
                        t,
                        ps.mu_true[i]!.toFixed(3),
                        ps.mu_hat[i]!.toFixed(3),
                      ]),
                    }
                  : null
              }
              source={source}
              missing={!ps}
            >
              <LineChart
                series={puddleSeries}
                xLabel={S.figures.puddle.x}
                yLabel={S.figures.puddle.y}
                formatX={(v) => `${Math.round(v)}`}
                formatY={(v) => v.toFixed(2)}
                vline={{ x: 0, label: 'grip drops' }}
              />
            </ChartFrame>

            <ChartFrame
              title={
                cal?.ece != null
                  ? S.figures.reliability.title(cal.ece.toFixed(3))
                  : S.figures.reliability.neutral
              }
              subtitle={`${S.figures.reliability.subtitle}${cal?.brier != null ? ` Brier ${cal.brier.toFixed(3)}, AUROC ${cal.auroc == null ? '–' : cal.auroc.toFixed(2)}.` : ''}`}
              summary="Reliability diagram: predicted crash risk against observed crash rate, with the diagonal for perfect calibration."
              table={
                cal?.bins
                  ? {
                      columns: ['Bin', 'Predicted', 'Observed', 'Steps'],
                      rows: cal.bins.map((b, i) => [i + 1, pct1(b.p_pred), pct1(b.p_obs), b.n]),
                    }
                  : null
              }
              source={source}
              missing={!cal?.bins}
            >
              <LineChart
                series={relSeries}
                xLabel={S.figures.reliability.x}
                yLabel={S.figures.reliability.y}
                formatX={pct}
                formatY={pct}
                xDomain={[0, relMax]}
                yDomain={[0, relMax]}
                diagonal
                directLabels={false}
              />
            </ChartFrame>

            <ChartFrame
              title={
                !sweep.length
                  ? S.figures.governor.neutral
                  : govHelps
                    ? S.figures.governor.title
                    : S.figures.governor.titleNoEffect
              }
              subtitle={S.figures.governor.subtitle}
              summary="Crash rate against lap time for each governor threshold."
              table={
                sweep.length
                  ? {
                      columns: ['Threshold τ', 'Crash rate', 'Paired lap time'],
                      rows: sweep.map((r) => [
                        r.tau === null ? 'off' : pct(r.tau),
                        fmtCi(r.crash, pct1),
                        fmtCi(r.paired_lap_time_s, secs),
                      ]),
                    }
                  : null
              }
              source={source}
              missing={sweep.length === 0}
            >
              <LineChart
                series={govSeries}
                xLabel={S.figures.governor.x}
                yLabel={S.figures.governor.y}
                formatX={(v) => `${v.toFixed(1)} s`}
                formatY={pct}
                directLabels={false}
              />
            </ChartFrame>

            <ChartFrame
              title={
                bh
                  ? S.figures.bench.title(
                      Math.round(bh.speedup).toLocaleString('en-US'),
                      bh.maxCars.toLocaleString('en-US'),
                    )
                  : S.figures.bench.neutral
              }
              subtitle={S.figures.bench.subtitle}
              summary="Log-log chart of simulation throughput against number of cars for GPU, CPU and a plain Python loop."
              table={
                bench
                  ? {
                      columns: ['Implementation', 'Cars', 'Car-steps / s'],
                      rows: bench.rows.map((r) => [
                        benchLabel(r.impl),
                        r.n_cars.toLocaleString('en-US'),
                        Math.round(r.steps_per_s).toLocaleString('en-US'),
                      ]),
                    }
                  : null
              }
              source={bench ? S.benchSource(bench.created_utc.slice(0, 10), bench.git_sha) : ''}
              missing={!bench}
            >
              <LineChart
                series={benchSeries}
                xLabel={S.figures.bench.x}
                yLabel={S.figures.bench.y}
                formatX={(v) => (v >= 1000 ? `${v / 1000}k` : `${v}`)}
                formatY={(v) => (v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : `${v}`)}
                logX
                logY
              />
            </ChartFrame>

            <ChartFrame
              title={
                genDots.length ? S.figures.generalization.title : S.figures.generalization.neutral
              }
              subtitle={S.figures.generalization.subtitle}
              summary="Completion on training tracks against unseen test tracks."
              drivers={['student', 'blind', 'teacher']}
              table={
                gen
                  ? {
                      columns: ['Driver', 'Training tracks', 'Unseen tracks'],
                      rows: (['student', 'blind', 'teacher'] as ChartDriver[]).map((d) => [
                        DRIVER_STYLE[d].label,
                        gen.train_bank[d] ? fmtCi(gen.train_bank[d].completion, pct1) : '–',
                        gen.test_bank[d] ? fmtCi(gen.test_bank[d].completion, pct1) : '–',
                      ]),
                    }
                  : null
              }
              source={source}
              missing={genDots.length === 0}
            >
              <DotPlot
                rows={['Training tracks', 'Unseen test tracks']}
                data={genDots}
                domain={[0, 1]}
                format={pct}
                axisLabel={S.figures.generalization.axis}
                rowHeight={44}
              />
            </ChartFrame>
          </div>
        </div>
      </details>

      <section className="mt-16 border-t border-line pt-6" aria-labelledby="limits">
        <h2 id="limits" className="h2">
          {S.limitsTitle}
        </h2>
        <ul className="mt-4 flex max-w-[72ch] list-disc flex-col gap-2 pl-5 text-text-2">
          {S.limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>
    </div>
  )
}
