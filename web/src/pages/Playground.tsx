import { useEffect, useMemo, useState } from 'react'
import { Button } from '../components/Button.tsx'
import { DiceIcon, PlayIcon } from '../components/icons.tsx'
import { LiveCanvas } from '../components/LiveCanvas.tsx'
import { Slider } from '../components/Slider.tsx'
import { strings } from '../copy/strings.ts'
import { loadShowcase, loadSimConfig, type ShowcaseTrack } from '../data/artifacts.ts'
import { useLiveSim } from '../hooks/useLiveSim.ts'
import type { SimConfig } from '../sim/config.ts'
import type { ConditionValues } from '../worker/protocol.ts'

const PLAYGROUND_CARS = 3
const SPEEDS = [0.5, 1, 2] as const

type Key = keyof ConditionValues
interface Control {
  key: Key
  label: string
  min: number
  max: number
  step: number
  trained?: [number, number]
  format: (v: number) => string
  spoken: (v: number) => string
}

const pct = (v: number) => `${Math.round(v * 100)}%`
const times = (v: number) => `×${v.toFixed(2)}`

/** Slider definitions, all ranges from config.json: full range = training range ∪ test extremes. */
function controlsFrom(cfg: SimConfig): Record<'weather' | 'car' | 'electronics', Control[]> {
  const c = cfg.conditions
  const span = (r: { lo: number; hi: number; ood: number; nominal: number }) => ({
    min: Math.min(r.lo, r.ood, r.nominal),
    max: Math.max(r.hi, r.ood),
    trained: [r.lo, r.hi] as [number, number],
  })
  const dtMs = cfg.time.control_dt_s * 1000
  return {
    weather: [
      {
        key: 'rain',
        label: 'Rain',
        step: 0.01,
        ...span(c.rain),
        format: pct,
        spoken: (v) => `${Math.round(v * 100)} percent rain`,
      },
      {
        key: 'puddles',
        label: 'Puddles',
        min: 0,
        max: c.puddle_slots,
        step: 1,
        trained: [0, c.puddle_slots],
        format: (v) => `${v}`,
        spoken: (v) => `${v} puddles`,
      },
      {
        key: 'windMps',
        label: 'Wind',
        step: 0.5,
        ...span(c.wind_speed_mps),
        format: (v) => `${v.toFixed(1)} m/s`,
        spoken: (v) => `${v} meters per second of wind`,
      },
      {
        key: 'windDirDeg',
        label: 'Wind direction',
        min: 0,
        max: 355,
        step: 5,
        format: (v) => `${v}°`,
        spoken: (v) => `wind toward ${v} degrees`,
      },
      {
        key: 'gustMps',
        label: 'Gusts',
        step: 0.1,
        ...span(c.gust_sigma_mps),
        format: (v) => `${v.toFixed(1)} m/s`,
        spoken: (v) => `gusts of ${v} meters per second`,
      },
    ],
    car: [
      {
        key: 'massMult',
        label: 'Mass',
        step: 0.01,
        ...span(c.mass_mult),
        format: times,
        spoken: (v) => `mass times ${v.toFixed(2)}`,
      },
      {
        key: 'powerMult',
        label: 'Power',
        step: 0.01,
        ...span(c.power_mult),
        format: times,
        spoken: (v) => `power times ${v.toFixed(2)}`,
      },
      {
        key: 'brakeMult',
        label: 'Brakes',
        step: 0.01,
        ...span(c.brake_mult),
        format: times,
        spoken: (v) => `brakes times ${v.toFixed(2)}`,
      },
      {
        key: 'dragAreaM2',
        label: 'Drag',
        step: 0.01,
        ...span(c.drag_area_m2),
        format: (v) => `${v.toFixed(2)} m²`,
        spoken: (v) => `drag area ${v.toFixed(2)} square meters`,
      },
      {
        key: 'wearPerKm',
        label: 'Tire wear',
        step: 0.01,
        ...span(c.tire_wear_per_km),
        format: (v) => `${pct(v)} / km`,
        spoken: (v) => `${Math.round(v * 100)} percent grip lost per kilometer`,
      },
    ],
    electronics: [
      {
        key: 'steerTauS',
        label: 'Steering lag',
        step: 0.005,
        ...span(c.steer_tau_s),
        format: (v) => `${Math.round(v * 1000)} ms`,
        spoken: (v) => `${Math.round(v * 1000)} milliseconds of steering lag`,
      },
      {
        key: 'delaySteps',
        label: 'Input delay',
        min: 0,
        max: c.delay_ood_steps,
        step: 1,
        trained: [0, c.delay_max_steps],
        format: (v) => `${v * dtMs} ms`,
        spoken: (v) => `${v * dtMs} milliseconds of input delay`,
      },
    ],
  }
}

/** Playground skeleton (SPEC 10.2): live canvas + condition sliders. More arrives with the AI drivers. */
export default function Playground() {
  const [cfg, setCfg] = useState<SimConfig | null>(null)
  const [tracks, setTracks] = useState<ShowcaseTrack[]>([])
  const [seed, setSeed] = useState<number | null>(null)
  const firstSeed = useMemo(() => loadShowcase().then((t) => t[0]!.seed), [])
  const sim = useLiveSim({
    nCars: PLAYGROUND_CARS,
    trackSeed: firstSeed,
    conditions: {},
    layout: 'grid',
    driver: 'scripted',
    intro: false,
    seed: 3,
  })
  const [values, setValues] = useState<Partial<ConditionValues>>({})
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState<number>(1)

  useEffect(() => {
    loadSimConfig().then(setCfg, () => undefined)
    loadShowcase().then(
      (t) => {
        setTracks(t)
        setSeed((s) => s ?? t[0]!.seed)
      },
      () => undefined,
    )
  }, [])

  const groups = useMemo(() => (cfg ? controlsFrom(cfg) : null), [cfg])
  const defaults = useMemo<Partial<ConditionValues>>(
    () =>
      cfg
        ? {
            rain: 0,
            puddles: 0,
            windMps: 0,
            windDirDeg: 0,
            gustMps: 0,
            massMult: cfg.conditions.mass_mult.nominal,
            powerMult: cfg.conditions.power_mult.nominal,
            brakeMult: cfg.conditions.brake_mult.nominal,
            dragAreaM2: cfg.conditions.drag_area_m2.nominal,
            wearPerKm: cfg.conditions.tire_wear_per_km.nominal,
            steerTauS: cfg.conditions.steer_tau_s.nominal,
            delaySteps: 0,
          }
        : {},
    [cfg],
  )

  const set = (key: Key, value: number) => {
    setValues((v) => ({ ...v, [key]: value }))
    void sim.setConditions({ [key]: value })
  }
  const pickTrack = (s: number) => {
    setSeed(s)
    void sim.setTrack(s)
  }
  const togglePlay = () => {
    void sim.control(playing ? 'pause' : 'resume')
    setPlaying(!playing)
  }
  const pickSpeed = (s: number) => {
    setSpeed(s)
    void sim.setSpeed(s)
  }
  const current = tracks.find((t) => t.seed === seed)

  return (
    <div className="mx-auto grid max-w-[1920px] gap-0 lg:grid-cols-[1fr_340px]">
      <section className="flex min-w-0 flex-col" aria-labelledby="pg-title">
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-3 md:px-6">
          <h1 id="pg-title" className="sr-only">
            {strings.playground.title}
          </h1>
          <label className="label" htmlFor="pg-track">
            {strings.playground.track}
          </label>
          <select
            id="pg-track"
            className="select"
            value={current ? current.seed : ''}
            onChange={(e) => pickTrack(Number(e.currentTarget.value))}
          >
            {!current && seed !== null && <option value="">{strings.playground.seed(seed)}</option>}
            {tracks.map((t) => (
              <option key={t.seed} value={t.seed}>
                {t.name}
              </option>
            ))}
          </select>
          {seed !== null && (
            <span className="numeral text-[14px] text-text-3">{strings.playground.seed(seed)}</span>
          )}
          <Button onClick={() => pickTrack(Math.floor(Math.random() * 2 ** 31))}>
            <DiceIcon />
            {strings.playground.newTrack}
          </Button>
          <span className="mx-1 hidden h-6 w-px bg-line md:block" aria-hidden="true" />
          <Button
            onClick={togglePlay}
            aria-label={playing ? strings.playground.pause : strings.playground.play}
          >
            <PlayIcon playing={playing} />
          </Button>
          <div className="segmented" role="group" aria-label={strings.playground.speed}>
            {SPEEDS.map((s) => (
              <button key={s} type="button" aria-pressed={speed === s} onClick={() => pickSpeed(s)}>
                {s}×
              </button>
            ))}
          </div>
          <Button variant="ghost" onClick={() => void sim.control('reset')}>
            {strings.playground.restart}
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2 text-[14px] md:px-6">
          <span className="label">{strings.playground.driver}</span>
          <div className="segmented" role="group" aria-label={strings.playground.driver}>
            <button type="button" aria-pressed="true">
              {strings.playground.drivers.scripted}
            </button>
            {(['student', 'blind', 'teacher'] as const).map((d) => (
              <button key={d} type="button" disabled aria-pressed="false">
                {strings.playground.drivers[d]}
              </button>
            ))}
          </div>
          <span className="text-text-3">{strings.playground.driversSoon}</span>
        </div>
        <LiveCanvas
          canvasRef={sim.canvasRef}
          phase={sim.status.phase}
          label={strings.playground.canvasLabel}
          className="aspect-[4/3] w-full lg:aspect-auto lg:h-[calc(100svh-56px-110px)] lg:min-h-[480px]"
        />
      </section>
      <aside
        className="border-line bg-asphalt-800 lg:h-[calc(100svh-56px)] lg:overflow-y-auto lg:border-l"
        aria-label="Conditions"
      >
        <p className="px-5 pt-5 text-[14px] text-text-2">{strings.playground.intro}</p>
        {groups &&
          (['weather', 'car', 'electronics'] as const).map((g, i) => (
            <details key={g} open={i === 0} className="group border-b border-line">
              <summary className="cursor-pointer list-none px-5 py-4 text-[16px] font-[620] text-paint">
                {strings.playground.groups[g]}
              </summary>
              <div className="flex flex-col gap-5 px-5 pb-6">
                {groups[g].map((c) => (
                  <Slider
                    key={c.key}
                    label={c.label}
                    value={values[c.key] ?? defaults[c.key] ?? c.min}
                    min={c.min}
                    max={c.max}
                    step={c.step}
                    trained={c.trained}
                    onChange={(v) => set(c.key, v)}
                    format={c.format}
                    valueText={c.spoken}
                  />
                ))}
              </div>
            </details>
          ))}
      </aside>
    </div>
  )
}
