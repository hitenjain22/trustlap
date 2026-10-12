import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '../components/Button.tsx'
import { DiceIcon, PlayIcon } from '../components/icons.tsx'
import { LiveCanvas } from '../components/LiveCanvas.tsx'
import { TelemetryPanel } from '../components/TelemetryPanel.tsx'
import { PuddleTestPanel } from '../components/PuddleTestPanel.tsx'
import { TrackDrawer } from '../components/TrackDrawer.tsx'
import { Slider } from '../components/Slider.tsx'
import { strings } from '../copy/strings.ts'
import {
  loadCalibration,
  loadShowcase,
  loadSimConfig,
  type Calibration,
  type ShowcaseTrack,
} from '../data/artifacts.ts'
import { useLiveSim } from '../hooks/useLiveSim.ts'
import { scorePuddleTest } from '../state/puddleTest.ts'
import { parsePlayground, serializePlayground } from '../state/urlState.ts'
import type { SimConfig } from '../sim/config.ts'
import { CrashCause, type ConditionValues, type DriverKind } from '../worker/protocol.ts'

/** Condition presets (SPEC 10.2); anything a preset doesn't name resets to nominal. */
const PRESETS = [
  { id: 'dry', values: {} },
  { id: 'monsoon', values: { rain: 1, puddles: 8, windMps: 8, gustMps: 2 } },
  { id: 'ice', values: { muDry: 0.25 } },
  { id: 'broken', values: { steerTauS: 0.15, delaySteps: 2, obsNoise: 0.05 } },
  {
    id: 'chaos',
    values: {
      rain: 0.9,
      puddles: 6,
      windMps: 10,
      gustMps: 3,
      massMult: 1.2,
      powerMult: 0.8,
      brakeMult: 0.75,
      dragAreaM2: 1.3,
      wearPerKm: 0.2,
      steerTauS: 0.12,
      delaySteps: 2,
      obsNoise: 0.04,
    },
  },
] as const satisfies readonly { id: string; values: Partial<ConditionValues> }[]

const PLAYGROUND_CARS = 1 // the focused car; ghosts are added per driver
const DRIVER_ORDER: DriverKind[] = ['student', 'blind', 'teacher', 'scripted'] // SPEC 10.2
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
        key: 'muDry',
        label: 'Road grip (dry)',
        step: 0.01,
        ...span(c.mu_dry),
        format: (v) => `μ ${v.toFixed(2)}`,
        spoken: (v) => `dry road grip ${v.toFixed(2)}`,
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

/** Playground (SPEC 10.2): live canvas, driver + ghosts, condition sliders. Telemetry lands in M5. */
export default function Playground() {
  // Shareable state from the URL (SPEC 10.5); defaults when absent.
  const url = useMemo(() => parsePlayground(window.location.search), [])
  const [cfg, setCfg] = useState<SimConfig | null>(null)
  const [tracks, setTracks] = useState<ShowcaseTrack[]>([])
  const [seed, setSeed] = useState<number | null>(
    url.track && 'seed' in url.track ? url.track.seed : null,
  )
  const firstSeed = useMemo(
    () =>
      loadShowcase().then((t) => {
        if (url.track && 'seed' in url.track) return url.track.seed
        const slug = url.track && 'slug' in url.track ? url.track.slug : null
        return (t.find((x) => x.slug === slug) ?? t[0]!).seed
      }),
    [url],
  )
  const sim = useLiveSim({
    nCars: PLAYGROUND_CARS,
    trackSeed: firstSeed,
    conditions: url.values,
    layout: 'grid',
    driver: url.driver ?? 'student',
    ghosts: url.ghosts ?? ['blind'],
    showFocus: true,
    telemetry: true,
    intro: false,
    seed: 3,
  })
  const [driver, setDriver] = useState<DriverKind>(url.driver ?? 'student')
  const [ghosts, setGhosts] = useState<DriverKind[]>(url.ghosts ?? ['blind'])
  const [calibration, setCalibration] = useState<Calibration | null>(null)
  const [governorOn, setGovernorOn] = useState(url.governor ?? false)
  const [tau, setTau] = useState(url.tau ?? 0.2)
  const [toast, setToast] = useState<string | null>(null)
  // Crash toast: shown while the focused car has crashed more times than the user has seen.
  // The worker's counts restart whenever the cars are re-placed, so those actions reset this.
  const [crashSeen, setCrashSeen] = useState(0)
  const [camera, setCamera] = useState<'overview' | 'chase'>('overview')
  const [showForces, setShowForces] = useState(false)
  const [showKeys, setShowKeys] = useState(false)
  const [drawing, setDrawing] = useState(false)
  const [custom, setCustom] = useState(false) // a hand-drawn track or the puddle test's straight
  const [puddleTest, setPuddleTest] = useState(false)
  const drawButton = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    loadCalibration().then(
      (c) => {
        setCalibration(c)
        if (c && url.tau === null) setTau(c.tau_default)
      },
      () => undefined,
    )
  }, [url.tau])
  const applyGovernor = (on: boolean, t: number) => {
    setGovernorOn(on)
    setTau(t)
    void sim.setGovernor(on, t)
  }
  const available = (d: DriverKind) => sim.status.available.includes(d)
  const pickDriver = (d: DriverKind) => {
    const g = ghosts.filter((x) => x !== d)
    setDriver(d)
    setGhosts(g)
    setCrashSeen(0)
    void sim.setDrivers(d, g)
  }
  const toggleGhost = (d: DriverKind) => {
    const g = ghosts.includes(d) ? ghosts.filter((x) => x !== d) : [...ghosts, d]
    setGhosts(g)
    setCrashSeen(0)
    void sim.setDrivers(driver, g)
  }
  const [values, setValues] = useState<Partial<ConditionValues>>(url.values)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState<number>(url.speed ?? 1)

  // Once the sim is up, apply URL settings the worker doesn't take at init.
  const ready = sim.status.phase === 'ready'
  useEffect(() => {
    if (!ready) return
    if (speed !== 1) void sim.setSpeed(speed)
    if (governorOn) void sim.setGovernor(governorOn, tau)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready])

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
            muDry: cfg.conditions.mu_dry.nominal,
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
            obsNoise: 0,
          }
        : {},
    [cfg],
  )

  const set = (key: Key, value: number) => {
    setValues((v) => ({ ...v, [key]: value }))
    void sim.setConditions({ [key]: value })
  }
  const applyPreset = (preset: (typeof PRESETS)[number]) => {
    const next = { ...defaults, ...preset.values }
    setValues(next)
    void sim.setConditions(next)
  }
  const pickTrack = (s: number) => {
    setSeed(s)
    setCustom(false)
    setPuddleTest(false)
    setCrashSeen(0)
    void sim.setTrack(s)
  }
  const runPuddleTest = () => {
    setShowKeys(false)
    setPuddleTest(true)
    setCustom(true)
    setCrashSeen(0)
    setDriver('student')
    setGhosts([])
    setValues({ ...defaults, puddles: 1 })
    setSpeed(0.5)
    void sim.setSpeed(0.5)
    if (!playing) {
      setPlaying(true)
      void sim.control('resume')
    }
    void sim.startPuddleTest()
  }
  const endPuddleTest = () => {
    setPuddleTest(false)
    setSpeed(1)
    void sim.setSpeed(1)
    if (!playing) {
      setPlaying(true)
      void sim.control('resume')
    }
    if (tracks[0]) pickTrack(tracks[0].seed)
  }
  const restartLap = () => {
    setCrashSeen(0)
    void sim.control('reset')
  }
  const showToast = (msg: string, ms = 2500) => {
    setToast(msg)
    setTimeout(() => setToast(null), ms)
  }
  // Drawing hides the sim, so pause it meanwhile.
  const openDrawer = () => {
    setShowKeys(false)
    setDrawing(true)
    if (playing) void sim.control('pause')
  }
  const closeDrawer = () => {
    setDrawing(false)
    if (playing) void sim.control('resume')
    requestAnimationFrame(() => drawButton.current?.focus())
  }
  const raceDrawn = (points: Float64Array, widthM: number) => {
    setCustom(true)
    setPuddleTest(false)
    setCrashSeen(0)
    void sim.setCustomTrack(points, widthM)
    closeDrawer()
    showToast(strings.playground.drawnNoLink, 4000)
  }
  const togglePlay = () => {
    void sim.control(playing ? 'pause' : 'resume')
    setPlaying(!playing)
  }
  const pickSpeed = (s: number) => {
    setSpeed(s)
    void sim.setSpeed(s)
  }
  // The seed on screen: the user's pick, else whatever the sim started with.
  const shownSeed = custom ? null : (seed ?? sim.status.track?.seed ?? null)
  const current = custom ? undefined : tracks.find((t) => t.seed === shownSeed)

  const t = sim.telemetry
  const crashCount = t?.crash.count ?? 0
  const crash =
    t && crashCount > crashSeen && Number.isFinite(t.crash.speedMps)
      ? strings.playground.crashToast({
          kmh: Math.round(t.crash.speedMps * 3.6),
          driver: strings.playground.drivers[t.kind].toLowerCase(),
          puddle: (t.crash.cause & CrashCause.Puddle) !== 0,
          corner: (t.crash.cause & CrashCause.Corner) !== 0,
          spun: (t.crash.cause & CrashCause.Spun) !== 0,
          stalled: (t.crash.cause & CrashCause.Stalled) !== 0,
          wet: (t.crash.cause & CrashCause.Wet) !== 0,
          governor:
            t.kind === 'student' && calibration
              ? (t.crash.cause & CrashCause.GovernorOn) !== 0
                ? 'on'
                : 'off'
              : null,
        })
      : null
  // Renderer view options (no worker round trip).
  const { setView } = sim
  useEffect(() => {
    setView({
      camera,
      showForces,
      haloTau: calibration ? tau : null,
      labels: strings.playground.drivers,
    })
  }, [setView, camera, showForces, calibration, tau, ready])

  // Puddle test: pause 1.5 s after the verdict so the chart and result stay on screen.
  const ph = puddleTest ? sim.telemetry?.history : undefined
  const puddleDone = ph
    ? ['noticed', 'missed'].includes(scorePuddleTest(ph.t, ph.muRear, ph.muHatRear).state)
    : false
  const { control } = sim
  useEffect(() => {
    if (!puddleDone || !playing) return
    const id = setTimeout(() => {
      void control('pause')
      setPlaying(false)
    }, 1500)
    return () => clearTimeout(id)
  }, [puddleDone, playing, control])

  // The crash toast hides itself after 6 s (or when dismissed).
  useEffect(() => {
    if (!crash) return
    const id = setTimeout(() => setCrashSeen(crashCount), 6000)
    return () => clearTimeout(id)
  }, [crash, crashCount])

  // Keep the URL in sync (replaceState, debounced 300 ms).
  const query = serializePlayground({
    trackSeed: shownSeed,
    trackSlug: current?.slug ?? null,
    values,
    driver,
    ghosts,
    governor: governorOn,
    tau,
    speed,
  })
  useEffect(() => {
    const id = setTimeout(
      () => window.history.replaceState(null, '', `${window.location.pathname}?${query}`),
      300,
    )
    return () => clearTimeout(id)
  }, [query])
  const copyLink = async () => {
    const link = `${window.location.origin}${window.location.pathname}?${query}`
    try {
      await navigator.clipboard.writeText(link)
      showToast(
        custom ? strings.playground.drawnNoLink : strings.playground.linkCopied,
        custom ? 4000 : 2500,
      )
    } catch {
      showToast(strings.playground.linkCopyFailed)
    }
  }

  // Keyboard shortcuts (SPEC 10.4), ignored while typing in a form control.
  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => undefined)
  useEffect(() => {
    keyHandler.current = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (
        drawing ||
        e.metaKey ||
        e.ctrlKey ||
        e.altKey ||
        el.closest('input, select, textarea, [contenteditable]')
      )
        return
      if (e.key === ' ') {
        e.preventDefault()
        togglePlay()
      } else if (e.key === 'r' || e.key === 'R') restartLap()
      else if (e.key === 'n' || e.key === 'N') pickTrack(Math.floor(Math.random() * 2 ** 31))
      else if (e.key === 'd' || e.key === 'D') {
        if (cfg) openDrawer()
      } else if (e.key === 'c' || e.key === 'C') {
        setCamera((c) => (c === 'chase' ? 'overview' : 'chase'))
      } else if (e.key === 'f' || e.key === 'F') setShowForces((v) => !v)
      else if (e.key === '?') setShowKeys((v) => !v)
      else if (/^[1-4]$/.test(e.key)) {
        const d = DRIVER_ORDER[Number(e.key) - 1]!
        if (available(d)) pickDriver(d)
      }
    }
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => keyHandler.current(e)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

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
            {custom && (
              <option value="">
                {puddleTest ? strings.puddleTest.track : strings.playground.drawnTrack}
              </option>
            )}
            {!current && shownSeed !== null && (
              <option value="">{strings.playground.seed(shownSeed)}</option>
            )}
            {tracks.map((t) => (
              <option key={t.seed} value={t.seed}>
                {t.name}
              </option>
            ))}
          </select>
          {shownSeed !== null && (
            <span className="numeral hidden text-[14px] text-text-3 2xl:inline">
              {strings.playground.seed(shownSeed)}
            </span>
          )}
          <Button onClick={() => pickTrack(Math.floor(Math.random() * 2 ** 31))}>
            <DiceIcon />
            {strings.playground.newTrack}
          </Button>
          <Button ref={drawButton} variant="ghost" disabled={!cfg || !ready} onClick={openDrawer}>
            {strings.playground.drawTrack}
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
          <Button variant="ghost" onClick={restartLap}>
            {strings.playground.restart}
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2 text-[14px] md:px-6">
          <span className="label">{strings.playground.driver}</span>
          <div className="segmented" role="group" aria-label={strings.playground.driver}>
            {DRIVER_ORDER.map((d) => (
              <button
                key={d}
                type="button"
                aria-pressed={driver === d}
                disabled={!available(d)}
                title={available(d) ? undefined : strings.playground.notTrainedYet}
                onClick={() => pickDriver(d)}
              >
                <span className={`driver-swatch driver-swatch--${d}`} aria-hidden="true" />
                {strings.playground.drivers[d]}
              </button>
            ))}
          </div>
          <span className="label">{strings.playground.ghosts}</span>
          <div className="flex flex-wrap gap-2" role="group" aria-label={strings.playground.ghosts}>
            {DRIVER_ORDER.filter((d) => d !== driver && available(d)).map((d) => (
              <label key={d} className="chip">
                <input
                  type="checkbox"
                  checked={ghosts.includes(d)}
                  onChange={() => toggleGhost(d)}
                />
                <span
                  className={`driver-swatch driver-swatch--${d} driver-swatch--ghost`}
                  aria-hidden="true"
                />
                {strings.playground.drivers[d]}
              </label>
            ))}
          </div>
        </div>
        {/* While drawing, give small screens room to draw (the overlay covers this box). */}
        <div className={`relative ${drawing ? 'min-h-[min(560px,75svh)]' : ''}`}>
          <LiveCanvas
            canvasRef={sim.canvasRef}
            phase={sim.status.phase}
            label={strings.playground.canvasLabel}
            className="aspect-[4/3] w-full lg:aspect-auto lg:h-[calc(100svh-56px-110px-200px)] lg:min-h-[380px]"
          />
          {drawing && cfg && (
            <TrackDrawer
              cfg={cfg.track}
              onUse={raceDrawn}
              onRandom={() => {
                pickTrack(Math.floor(Math.random() * 2 ** 31))
                closeDrawer()
              }}
              onCancel={closeDrawer}
            />
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3 border-t border-line px-4 py-2 text-[14px] md:px-6">
          <span className="label">{strings.playground.camera}</span>
          <div className="segmented" role="group" aria-label={strings.playground.camera}>
            {(['overview', 'chase'] as const).map((c) => (
              <button
                key={c}
                type="button"
                aria-pressed={camera === c}
                onClick={() => setCamera(c)}
              >
                {strings.playground.cameras[c]}
              </button>
            ))}
          </div>
          <Button variant="ghost" disabled={!ready} onClick={runPuddleTest}>
            {strings.puddleTest.run}
          </Button>
          <label className="chip">
            <input
              type="checkbox"
              checked={showForces}
              onChange={(e) => setShowForces(e.currentTarget.checked)}
            />
            {strings.playground.showForces}
          </label>
        </div>
        {puddleTest && (
          <PuddleTestPanel t={sim.telemetry} onRunAgain={runPuddleTest} onEnd={endPuddleTest} />
        )}
        <TelemetryPanel t={sim.telemetry} tau={calibration ? tau : null} governorOn={governorOn} />
      </section>
      <aside
        className="border-line bg-asphalt-800 lg:h-[calc(100svh-56px)] lg:overflow-y-auto lg:border-l"
        aria-label="Conditions"
      >
        <div className="flex items-center justify-end gap-1 border-b border-line px-3 py-2">
          <Button variant="ghost" onClick={() => void copyLink()}>
            {strings.playground.copyLink}
          </Button>
          <div className="relative">
            <Button
              variant="ghost"
              aria-expanded={showKeys}
              aria-label={strings.playground.shortcuts}
              onClick={() => setShowKeys((v) => !v)}
            >
              ?
            </Button>
            {showKeys && (
              <div
                className="absolute right-0 z-20 mt-2 w-64 rounded-panel border border-line bg-asphalt-800 p-4 text-[13px]"
                role="dialog"
                aria-label={strings.playground.shortcuts}
              >
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5">
                  {strings.playground.keys.map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt>
                        <kbd className="rounded border border-line px-1.5 text-paint">{k}</kbd>
                      </dt>
                      <dd className="text-text-2">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </div>
        </div>
        <p className="px-5 pt-5 text-[14px] text-text-2">{strings.playground.intro}</p>
        <div className="px-5 pt-4">
          <p className="label">{strings.playground.presets}</p>
          <div
            className="mt-2 flex flex-wrap gap-2"
            role="group"
            aria-label={strings.playground.presets}
          >
            {PRESETS.map((p) => (
              <button key={p.id} type="button" className="chip" onClick={() => applyPreset(p)}>
                {strings.playground.presetNames[p.id]}
              </button>
            ))}
          </div>
        </div>
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
        {calibration && (
          <details className="group border-b border-line">
            <summary className="cursor-pointer list-none px-5 py-4 text-[16px] font-[620] text-paint">
              {strings.playground.groups.safety}
            </summary>
            <div className="flex flex-col gap-4 px-5 pb-6">
              <label className="flex items-center justify-between gap-3 text-[14px] text-paint">
                {strings.playground.governor}
                <input
                  type="checkbox"
                  role="switch"
                  checked={governorOn}
                  onChange={(e) => applyGovernor(e.currentTarget.checked, tau)}
                  className="size-5 accent-[var(--sodium)]"
                />
              </label>
              <Slider
                label={strings.playground.governorThreshold}
                value={tau}
                min={0.05}
                max={0.5}
                step={0.05}
                onChange={(v) => applyGovernor(governorOn, v)}
                format={(v) => `${Math.round(v * 100)}%`}
                valueText={(v) => `${Math.round(v * 100)} percent crash risk`}
              />
              <p className="text-[13px] text-text-3">{strings.playground.governorNote}</p>
            </div>
          </details>
        )}
      </aside>
      {crash && (
        <div
          role="status"
          className="fixed bottom-6 left-1/2 z-30 flex w-[min(92vw,560px)] -translate-x-1/2 flex-wrap items-center gap-3 rounded-panel border border-line bg-asphalt-800 px-4 py-3 text-[14px] text-paint"
        >
          <p className="min-w-0 flex-1">{crash}</p>
          <Button variant="primary" onClick={restartLap}>
            {strings.playground.restart}
          </Button>
          <Button variant="ghost" onClick={() => setCrashSeen(crashCount)}>
            {strings.playground.dismiss}
          </Button>
        </div>
      )}
      {toast && !crash && (
        <div
          role="status"
          className="fixed bottom-6 left-1/2 z-30 -translate-x-1/2 rounded-panel border border-line bg-asphalt-800 px-4 py-3 text-[14px] text-paint"
        >
          {toast}
        </div>
      )}
    </div>
  )
}
