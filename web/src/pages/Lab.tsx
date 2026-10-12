/**
 * Training Lab (SPEC 10.2): watch real checkpoints drive, the population's fitness, the PBT
 * family tree, the curriculum, and how the winning lineage tuned itself. All from artifacts.
 */
import { scaleLinear } from 'd3-scale'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ChartFrame } from '../charts/ChartFrame.tsx'
import { LineChart, type Series } from '../charts/LineChart.tsx'
import { useWidth } from '../charts/useWidth.ts'
import { Button } from '../components/Button.tsx'
import { PlayIcon } from '../components/icons.tsx'
import { LiveCanvas } from '../components/LiveCanvas.tsx'
import { strings } from '../copy/strings.ts'
import {
  loadCheckpoints,
  loadCurriculum,
  loadLineage,
  loadPopulation,
  loadShowcase,
} from '../data/artifacts.ts'
import type { Checkpoints, Curriculum, Lineage, Population } from '../data/schemas.ts'
import { useLiveSim } from '../hooks/useLiveSim.ts'

const L = strings.lab
const LAB_CARS = 16
const STOP_MS = 6000
const millions = (v: number) =>
  v === 0 ? '0' : v >= 1e6 ? `${Math.round(v / 1e6)}M` : `${Math.round(v / 1e3)}k`

/** Ids on the winner's ancestry: the winner, the member it copied, that member's source, … */
function ancestry(lineage: Lineage | null): Set<number> {
  const out = new Set<number>()
  if (!lineage || lineage.winner_id == null) return out
  let id: number | undefined = lineage.winner_id
  while (id !== undefined && !out.has(id)) {
    out.add(id)
    id = lineage.events.find((e) => e.child_id === id)?.parent_id
  }
  return out
}

function WatchItLearn({ checkpoints }: { checkpoints: Checkpoints['checkpoints'] }) {
  const [stop, setStop] = useState(0)
  const [playing, setPlaying] = useState(false)
  const seed = useMemo(() => loadShowcase().then((t) => t[Math.min(5, t.length - 1)]!.seed), [])
  const first = checkpoints[0]!
  const sim = useLiveSim({
    nCars: LAB_CARS,
    trackSeed: seed,
    conditions: { rain: first.difficulty },
    autoPuddles: true,
    layout: 'spread',
    driver: 'teacher',
    checkpoint: first.name,
    intro: false,
    seed: 11,
  })
  const go = (i: number) => {
    const c = checkpoints[i]!
    setStop(i)
    void sim.setCheckpoint(c.name)
    void sim.setConditions({ rain: c.difficulty })
  }
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)
  useEffect(() => {
    if (!playing) return
    timer.current = setInterval(() => {
      setStop((s) => {
        const next = (s + 1) % checkpoints.length
        const c = checkpoints[next]!
        void sim.setCheckpoint(c.name)
        void sim.setConditions({ rain: c.difficulty })
        return next
      })
    }, STOP_MS)
    return () => clearInterval(timer.current!)
  }, [playing, checkpoints, sim])
  const c = checkpoints[stop]!
  return (
    <section aria-labelledby="watch" className="grid gap-6 lg:grid-cols-[1fr_360px]">
      <LiveCanvas
        canvasRef={sim.canvasRef}
        phase={sim.status.phase}
        label={L.watch.canvasLabel(millions(c.env_steps))}
        className="aspect-[4/3] w-full overflow-hidden rounded-panel border border-[var(--edge)] lg:aspect-[16/10]"
      />
      <div>
        <h2 id="watch" className="h2">
          {L.watch.title}
        </h2>
        <p className="mt-3 text-text-2">{L.watch.body}</p>
        <p className="numeral mt-6 text-[40px] leading-none text-paint">{millions(c.env_steps)}</p>
        <p className="label mt-1">{L.watch.stepsLabel}</p>
        <p className="mt-4 text-paint" aria-live="polite">
          {L.watch.caption(Math.round(c.completion * 100), c.difficulty)}
        </p>
        <div className="mt-6 flex items-center gap-3">
          <Button
            onClick={() => setPlaying((p) => !p)}
            aria-label={playing ? L.watch.pause : L.watch.play}
          >
            <PlayIcon playing={playing} />
          </Button>
          <input
            type="range"
            min={0}
            max={checkpoints.length - 1}
            step={1}
            value={stop}
            onChange={(e) => (setPlaying(false), go(Number(e.currentTarget.value)))}
            aria-label={L.watch.scrubber}
            aria-valuetext={`${millions(c.env_steps)} training steps`}
            className="flex-1 accent-[var(--sodium)]"
          />
        </div>
        <div className="mt-2 flex justify-between text-[12px] text-text-3">
          <span>{millions(checkpoints[0]!.env_steps)}</span>
          <span>{millions(checkpoints[checkpoints.length - 1]!.env_steps)}</span>
        </div>
      </div>
    </section>
  )
}

function FamilyTree({
  lineage,
  population,
  winners,
}: {
  lineage: Lineage
  population: Population
  winners: Set<number>
}) {
  const { ref, width } = useWidth<HTMLDivElement>()
  const slots = population.members
  const lane = 28
  const margin = { top: 10, right: 16, bottom: 36, left: 64 }
  const height = margin.top + slots * lane + margin.bottom
  const pts = population.points
  const maxSteps = pts[pts.length - 1]?.env_steps ?? 1
  const x = scaleLinear()
    .domain([0, maxSteps])
    .range([margin.left, width - margin.right])
  const y = (slot: number) => margin.top + slot * lane + lane / 2
  // Tenures: contiguous runs of the same id in each slot.
  const tenures: { slot: number; id: number; from: number; to: number }[] = []
  for (let s = 0; s < slots; s++) {
    let start = 0
    for (let i = 1; i <= pts.length; i++) {
      if (i === pts.length || pts[i]!.ids[s] !== pts[start]!.ids[s]) {
        tenures.push({
          slot: s,
          id: pts[start]!.ids[s]!,
          from: start === 0 ? 0 : pts[start]!.env_steps,
          to: pts[i - 1]!.env_steps,
        })
        start = i
      }
    }
  }
  const slotOf = (id: number, steps: number) =>
    tenures.find((t) => t.id === id && t.from <= steps + 1 && t.to >= steps - 1)?.slot
  return (
    <div ref={ref}>
      <svg width={width} height={height} className="block">
        {Array.from({ length: slots }, (_, s) => (
          <text
            key={s}
            x={margin.left - 10}
            y={y(s) + 4}
            textAnchor="end"
            className="fill-[var(--text-3)] text-[12px]"
          >
            {L.tree.slot(s + 1)}
          </text>
        ))}
        {tenures.map((t, i) => (
          <line
            key={i}
            x1={x(t.from)}
            x2={x(t.to)}
            y1={y(t.slot)}
            y2={y(t.slot)}
            stroke={winners.has(t.id) ? 'var(--sodium)' : 'var(--text-3)'}
            strokeOpacity={winners.has(t.id) ? 1 : 0.45}
            strokeWidth={winners.has(t.id) ? 3 : 2}
          >
            <title>{L.tree.member(t.id)}</title>
          </line>
        ))}
        {lineage.events.map((e, i) => {
          const from = slotOf(e.parent_id, e.env_steps)
          if (from === undefined) return null
          const x0 = x(e.env_steps)
          const y0 = y(from)
          const y1 = y(e.slot)
          const win = winners.has(e.child_id)
          return (
            <path
              key={i}
              d={`M${x0 - 14} ${y0} C ${x0} ${y0}, ${x0 - 8} ${y1}, ${x0 + 6} ${y1}`}
              fill="none"
              stroke={win ? 'var(--sodium)' : 'var(--text-3)'}
              strokeOpacity={win ? 1 : 0.5}
              strokeWidth={win ? 2 : 1.25}
            >
              <title>
                {L.tree.event(
                  e.parent_id,
                  e.child_id,
                  millions(e.env_steps),
                  e.hp_before.lr,
                  e.hp_after.lr,
                  e.hp_before.ent,
                  e.hp_after.ent,
                )}
              </title>
            </path>
          )
        })}
        {x.ticks(width < 520 ? 3 : 6).map((t) => (
          <text
            key={t}
            x={x(t)}
            y={height - 14}
            textAnchor="middle"
            className="fill-[var(--text-3)] text-[12px]"
          >
            {millions(t)}
          </text>
        ))}
      </svg>
    </div>
  )
}

export default function Lab() {
  const [checkpoints, setCheckpoints] = useState<Checkpoints['checkpoints'] | null>(null)
  const [population, setPopulation] = useState<Population | null>(null)
  const [lineage, setLineage] = useState<Lineage | null>(null)
  const [curriculum, setCurriculum] = useState<Curriculum | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [hover, setHover] = useState<string | null>(null)
  useEffect(() => {
    Promise.all([loadCheckpoints(), loadPopulation(), loadLineage(), loadCurriculum()])
      .then(([c, p, l, cu]) => {
        setCheckpoints(c?.checkpoints ?? null)
        setPopulation(p)
        setLineage(l)
        setCurriculum(cu)
      })
      .catch((e) => console.error(e))
      .finally(() => setLoaded(true))
  }, [])
  const winners = useMemo(() => ancestry(lineage), [lineage])

  const fitnessSeries: Series[] = useMemo(() => {
    if (!population) return []
    const slots: Series[] = Array.from({ length: population.members }, (_, s) => ({
      id: `slot${s}`,
      label: L.tree.slot(s + 1),
      color: 'var(--text-3)',
      width: hover === `slot${s}` ? 2.5 : 1,
      opacity: hover && hover !== `slot${s}` && hover !== 'winner' ? 0.35 : 0.8,
      points: population.points.map((p) => ({ x: p.env_steps, y: p.fitness[s]! })),
    }))
    const best = population.points.map((p) => {
      const s = p.ids.findIndex((id) => winners.has(id))
      return { x: p.env_steps, y: s >= 0 ? p.fitness[s]! : NaN }
    })
    return [
      ...slots,
      {
        id: 'winner',
        label: L.population.winner,
        color: 'var(--sodium)',
        width: 2.5,
        points: best,
      },
    ]
  }, [population, winners, hover])

  // The newcomer's view: the winning driver's progress over training, one line.
  const winnerSeries: Series[] = fitnessSeries.filter((x) => x.id === 'winner')
  const first = checkpoints?.[0]
  const last = checkpoints?.[checkpoints.length - 1]
  const maxD = curriculum ? Math.max(...curriculum.points.map((p) => p.d)) : null
  const progressLine =
    first && last && maxD !== null
      ? L.simple.progress(
          Math.round(first.completion * 100),
          Math.round(last.completion * 100),
          Math.round(maxD * 100),
        )
      : null

  // The logged learning rate includes the linear decay to 0 (SPEC 7.2); PBT tunes the base rate,
  // so undo the known decay (and skip the last 5%, where the decayed value rounds to nothing).
  const totalUpdates = population ? Math.max(...population.points.map((p) => p.update)) : 1
  const hpSeries = (key: 'lr' | 'ent'): Series[] =>
    population
      ? [
          {
            id: key,
            label: L.population.winner,
            color: 'var(--sodium)',
            points: population.points.flatMap((p) => {
              const s = p.ids.findIndex((id) => winners.has(id))
              const decay = 1 - (p.update - 1) / totalUpdates
              if (key === 'lr' && decay < 0.05) return []
              const v = s >= 0 ? p[key][s]! : NaN
              return [{ x: p.env_steps, y: key === 'lr' ? v / decay : v }]
            }),
          },
        ]
      : []

  const source = L.source
  return (
    <div className="mx-auto max-w-[1280px] px-4 py-12 md:px-6 md:py-16">
      <h1 className="text-[40px] leading-[1.08] font-bold [font-stretch:118%] md:text-[49px]">
        {L.title}
      </h1>
      <p className="mt-3 max-w-[72ch] text-text-2">{L.intro}</p>
      <div className="mt-12 flex flex-col gap-16">
        {checkpoints && checkpoints.length > 0 ? (
          <WatchItLearn checkpoints={checkpoints} />
        ) : loaded ? (
          <p className="text-text-3">{strings.results.notComputed}</p>
        ) : (
          // Same footprint as the section below, so nothing jumps when the checkpoints arrive.
          <div className="grid gap-6 lg:grid-cols-[1fr_360px]" aria-hidden="true">
            <div className="aspect-[4/3] w-full rounded-panel border border-line lg:aspect-[16/10]" />
            <div className="min-h-[360px]" />
          </div>
        )}

        {loaded && winnerSeries.length > 0 && (
          <section aria-labelledby="simple-learn">
            <h2 id="simple-learn" className="h2">
              {L.simple.title}
            </h2>
            {progressLine && <p className="mt-3 max-w-[72ch] text-text-2">{progressLine}</p>}
            <div className="mt-6">
              <LineChart
                series={winnerSeries}
                xLabel={L.population.x}
                yLabel={L.simple.y}
                formatX={millions}
                formatY={(v) => v.toFixed(0)}
                directLabels={false}
                height={260}
              />
            </div>
            <p className="mt-2 text-[13px] text-text-3">{L.simple.note}</p>
          </section>
        )}

        <details className="card group">
          <summary className="cursor-pointer list-none px-5 py-4">
            <span className="text-[18px] font-[620] text-paint">{L.simple.fullTitle}</span>
            <span className="mt-1 block text-[14px] text-text-2">
              {population ? L.simple.fullSummary(population.members) : ''}
            </span>
          </summary>
          <div className="flex flex-col gap-16 border-t border-line px-5 pb-8 pt-6">
            <ChartFrame
              title={L.population.title(population?.members ?? 0)}
              subtitle={L.population.subtitle}
              summary={L.population.summary}
              table={
                population
                  ? {
                      columns: [
                        'Steps',
                        ...Array.from({ length: population.members }, (_, s) => L.tree.slot(s + 1)),
                      ],
                      rows: population.points
                        .filter((_, i) => i % Math.ceil(population.points.length / 20) === 0)
                        .map((p) => [millions(p.env_steps), ...p.fitness.map((f) => f.toFixed(2))]),
                    }
                  : null
              }
              source={source}
              missing={!population}
              loading={!loaded}
            >
              <LineChart
                series={fitnessSeries}
                xLabel={L.population.x}
                yLabel={L.population.y}
                formatX={millions}
                formatY={(v) => v.toFixed(0)}
                directLabels={false}
                onHoverSeries={setHover}
              />
            </ChartFrame>

            <ChartFrame
              title={L.tree.title(lineage?.events.length ?? 0)}
              subtitle={L.tree.subtitle}
              summary={L.tree.summary}
              table={
                lineage
                  ? {
                      columns: ['Steps', 'Copy', 'Learning rate', 'Entropy coef.'],
                      rows: lineage.events.map((e) => [
                        millions(e.env_steps),
                        `#${e.parent_id} → #${e.child_id}`,
                        `${e.hp_before.lr.toExponential(1)} → ${e.hp_after.lr.toExponential(1)}`,
                        `${e.hp_before.ent.toExponential(1)} → ${e.hp_after.ent.toExponential(1)}`,
                      ]),
                    }
                  : null
              }
              source={source}
              missing={!lineage || !population}
              loading={!loaded}
            >
              {lineage && population && (
                <FamilyTree lineage={lineage} population={population} winners={winners} />
              )}
            </ChartFrame>

            <ChartFrame
              title={L.curriculum.title(
                curriculum ? Math.max(...curriculum.points.map((p) => p.d)) : 0,
              )}
              subtitle={L.curriculum.subtitle}
              summary={L.curriculum.summary}
              table={
                curriculum
                  ? {
                      columns: ['Steps', 'Difficulty'],
                      rows: curriculum.points
                        .filter((_, i) => i % 10 === 0)
                        .map((p) => [millions(p.env_steps), p.d.toFixed(2)]),
                    }
                  : null
              }
              source={source}
              missing={!curriculum}
              loading={!loaded}
            >
              <LineChart
                series={
                  curriculum
                    ? [
                        {
                          id: 'd',
                          label: 'Difficulty',
                          color: 'var(--sodium)',
                          step: true,
                          points: curriculum.points.map((p) => ({ x: p.env_steps, y: p.d })),
                        },
                      ]
                    : []
                }
                xLabel={L.population.x}
                yLabel={L.curriculum.y}
                formatX={millions}
                formatY={(v) => v.toFixed(1)}
                yDomain={[0, 1]}
                directLabels={false}
                height={240}
              />
            </ChartFrame>

            <section aria-labelledby="tuning" className="border-t border-line pt-6">
              <h2 id="tuning" className="h3">
                {L.tuning.title}
              </h2>
              <p className="mt-1 max-w-[72ch] text-[14px] text-text-2">{L.tuning.subtitle}</p>
              {!loaded ? (
                <div className="mt-4 h-[240px]" aria-hidden="true" />
              ) : population ? (
                <div className="mt-4 grid grid-cols-1 gap-8 md:grid-cols-2">
                  <LineChart
                    series={hpSeries('lr')}
                    xLabel={L.population.x}
                    yLabel={L.tuning.lr}
                    formatX={millions}
                    formatY={(v) => v.toExponential(1)}
                    logY
                    directLabels={false}
                    height={220}
                  />
                  <LineChart
                    series={hpSeries('ent')}
                    xLabel={L.population.x}
                    yLabel={L.tuning.ent}
                    formatX={millions}
                    formatY={(v) => v.toExponential(1)}
                    logY
                    directLabels={false}
                    height={220}
                  />
                </div>
              ) : (
                <p className="mt-4 text-text-3">{strings.results.notComputed}</p>
              )}
            </section>
          </div>
        </details>
      </div>
    </div>
  )
}
