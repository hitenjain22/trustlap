/**
 * Race (Can you beat the AI?): the visitor drives one lap against the Adaptive AI on the same track,
 * in the same weather, from the same start. Arrow keys / WASD, or on-screen buttons on touch
 * screens. Lap times come from the simulator's own lap timer; the result can be shared.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '../components/Button.tsx'
import { DiceIcon, LineIcon } from '../components/icons.tsx'
import { LiveCanvas } from '../components/LiveCanvas.tsx'
import { strings } from '../copy/strings.ts'
import { loadShowcase, type ShowcaseTrack } from '../data/artifacts.ts'
import { useLiveSim } from '../hooks/useLiveSim.ts'
import { useMediaQuery } from '../hooks/useMediaQuery.ts'
import { accent } from '../styles/accents.ts'
import type { ConditionValues } from '../worker/protocol.ts'

const R = strings.race
type Weather = 'dry' | 'rain' | 'ice'
const WEATHER: Record<Weather, { values: Partial<ConditionValues>; icon: string; tint: string }> = {
  dry: { values: { rain: 0, puddles: 0, muDry: 1.08 }, icon: 'dry', tint: 'var(--sodium)' },
  rain: { values: { rain: 0.6, puddles: 4, muDry: 1.08 }, icon: 'monsoon', tint: 'var(--sky)' },
  ice: { values: { rain: 0, puddles: 0, muDry: 0.25 }, icon: 'ice', tint: 'var(--teal)' },
}
type Phase = 'setup' | 'countdown' | 'racing' | 'finished'
const STEER_RATE = 7 // 1/s: how fast the wheel follows a held key (smooths digital input)

const fmt = (s: number) => {
  if (!Number.isFinite(s)) return '–'
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`
}

export default function Race() {
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const initialWeather = (['dry', 'rain', 'ice'] as const).find((w) => w === params.get('weather'))
  const [weather, setWeather] = useState<Weather>(initialWeather ?? 'rain')
  const [tracks, setTracks] = useState<ShowcaseTrack[]>([])
  const urlSeed = Number(params.get('track'))
  const [seed, setSeed] = useState<number | null>(
    Number.isFinite(urlSeed) && urlSeed > 0 ? urlSeed : null,
  )
  const firstSeed = useMemo(
    () => loadShowcase().then((t) => seed ?? t[0]!.seed),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )
  const sim = useLiveSim({
    nCars: 1,
    trackSeed: firstSeed,
    conditions: WEATHER[initialWeather ?? 'rain'].values,
    layout: 'grid',
    driver: 'human',
    ghosts: ['student'],
    showFocus: true,
    telemetry: true,
    intro: false,
    seed: 11,
    race: true,
  })
  const [phase, setPhase] = useState<Phase>('setup')
  const [count, setCount] = useState(3)
  const [copied, setCopied] = useState(false)
  const touch = useMediaQuery('(pointer: coarse)')
  const ready = sim.status.phase === 'ready'
  const { control, setView, setHumanInput, setConditions, setTrack } = sim

  useEffect(() => {
    loadShowcase().then(setTracks, () => undefined)
  }, [])
  // Chase camera on your car; the grid waits (paused) until the race starts.
  useEffect(() => {
    if (!ready) return
    setView({ camera: 'chase', labels: strings.playground.drivers, haloTau: null })
    void control('pause')
  }, [ready, setView, control])

  // Countdown 3, 2, 1, then reset the grid and go.
  useEffect(() => {
    if (phase !== 'countdown') return
    if (count === 0) {
      void control('reset').then(() => control('resume'))
      const id = setTimeout(() => setPhase('racing'), 0)
      return () => clearTimeout(id)
    }
    const id = setTimeout(() => setCount((c) => c - 1), 700)
    return () => clearTimeout(id)
  }, [phase, count, control])

  // Input: held keys (or touch buttons) → smoothed steering and throttle, sent every frame.
  const keys = useRef({ left: false, right: false, up: false, down: false })
  useEffect(() => {
    const map: Record<string, keyof typeof keys.current> = {
      ArrowLeft: 'left',
      a: 'left',
      A: 'left',
      ArrowRight: 'right',
      d: 'right',
      D: 'right',
      ArrowUp: 'up',
      w: 'up',
      W: 'up',
      ArrowDown: 'down',
      s: 'down',
      S: 'down',
    }
    const on = (down: boolean) => (e: KeyboardEvent) => {
      const k = map[e.key]
      if (!k || (e.target as HTMLElement).closest('input, select, textarea')) return
      e.preventDefault()
      keys.current[k] = down
    }
    const kd = on(true)
    const ku = on(false)
    window.addEventListener('keydown', kd)
    window.addEventListener('keyup', ku)
    return () => {
      window.removeEventListener('keydown', kd)
      window.removeEventListener('keyup', ku)
    }
  }, [])
  useEffect(() => {
    if (phase !== 'racing') {
      setHumanInput(0, 0)
      return
    }
    let raf = 0
    let last = performance.now()
    let steer = 0
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const k = keys.current
      const target = (k.left ? 1 : 0) - (k.right ? 1 : 0) // + is left
      steer += (target - steer) * Math.min(1, dt * STEER_RATE)
      setHumanInput(steer, (k.up ? 1 : 0) - (k.down ? 1 : 0))
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [phase, setHumanInput])

  // The race ends when your lap completes; freeze the scene shortly after.
  const t = sim.telemetry
  const finished =
    phase === 'racing' && t !== null && t.kind === 'human' && Number.isFinite(t.lastLap)
  useEffect(() => {
    if (!finished) return
    const id = setTimeout(() => {
      setPhase('finished')
      void control('pause')
    }, 900)
    return () => clearTimeout(id)
  }, [finished, control])

  const start = () => {
    setCopied(false)
    setCount(3)
    setPhase('countdown')
  }
  const pickWeather = (w: Weather) => {
    setWeather(w)
    void setConditions(WEATHER[w].values)
    setPhase('setup')
    void control('reset').then(() => control('pause'))
  }
  const pickTrack = (s: number) => {
    setSeed(s)
    setPhase('setup')
    void setTrack(s).then(() => control('pause'))
  }
  const currentSeed = seed ?? sim.status.track?.seed ?? null
  const current = tracks.find((x) => x.seed === currentSeed)

  // Result: your lap vs the AI's (if it finished first) and crashes on the way.
  const you = t?.lastLap ?? NaN
  const ai = t?.ghostLastLap ?? NaN
  const won = !Number.isFinite(ai) || you < ai
  const margin = Number.isFinite(ai) ? Math.abs(ai - you) : NaN
  const shareUrl = `${window.location.origin}/race?weather=${weather}${currentSeed !== null ? `&track=${currentSeed}` : ''}`
  const shareText = won
    ? R.shareWin(
        Number.isFinite(margin) ? margin.toFixed(1) : null,
        R.weather[weather].name.toLowerCase(),
      )
    : R.shareLose(margin.toFixed(1), R.weather[weather].name.toLowerCase())
  const share = async () => {
    const text = `${shareText} ${shareUrl}`
    try {
      if (navigator.share) await navigator.share({ text, url: shareUrl })
      else await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch {
      /* dismissed */
    }
  }

  const hold = (k: keyof typeof keys.current, down: boolean) => () => {
    keys.current[k] = down
  }
  const pad = (k: keyof typeof keys.current, label: string, icon: string) => (
    <button
      type="button"
      aria-label={label}
      className="grid size-16 touch-none place-items-center rounded-2xl border border-[var(--edge)] bg-asphalt/70 text-[22px] text-paint select-none active:bg-asphalt-700"
      onPointerDown={hold(k, true)}
      onPointerUp={hold(k, false)}
      onPointerLeave={hold(k, false)}
      onPointerCancel={hold(k, false)}
    >
      {icon}
    </button>
  )

  return (
    <div className="mx-auto max-w-[1440px] px-4 py-8 md:px-6 md:py-10">
      <p className="eyebrow" style={accent('var(--teal)')}>
        {R.eyebrow}
      </p>
      <h1 className="mt-2 text-[36px] leading-[1.08] font-bold [font-stretch:118%] md:text-[46px]">
        {R.title}
      </h1>
      <p className="mt-2 max-w-[68ch] text-text-2">{R.intro}</p>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <div className="flex flex-wrap gap-2" role="group" aria-label={R.weatherLabel}>
          {(['dry', 'rain', 'ice'] as const).map((w) => (
            <button
              key={w}
              type="button"
              aria-pressed={weather === w}
              disabled={phase === 'countdown' || phase === 'racing'}
              onClick={() => pickWeather(w)}
              style={accent(WEATHER[w].tint)}
              className={`card card--hover flex items-center gap-2 px-3 py-2 text-[14px] font-[560] disabled:opacity-60 ${weather === w ? 'card--selected text-paint' : 'text-text-2'}`}
            >
              <span className="icon-badge size-7">
                <LineIcon name={WEATHER[w].icon} />
              </span>
              {R.weather[w].name}
            </button>
          ))}
        </div>
        <label className="label ml-2" htmlFor="race-track">
          {strings.playground.track}
        </label>
        <select
          id="race-track"
          className="select"
          value={current ? current.seed : ''}
          disabled={phase === 'countdown' || phase === 'racing'}
          onChange={(e) => pickTrack(Number(e.currentTarget.value))}
        >
          {!current && currentSeed !== null && (
            <option value="">{strings.playground.seed(currentSeed)}</option>
          )}
          {tracks.map((x) => (
            <option key={x.seed} value={x.seed}>
              {x.name}
            </option>
          ))}
        </select>
        <Button
          variant="ghost"
          disabled={phase === 'countdown' || phase === 'racing'}
          onClick={() => pickTrack(Math.floor(Math.random() * 2 ** 31))}
        >
          <DiceIcon />
          {strings.playground.newTrack}
        </Button>
      </div>
      <p className="mt-2 text-[13px] text-text-3">{R.weather[weather].note}</p>

      <div className="relative mt-4 overflow-hidden rounded-panel border border-[var(--edge)]">
        <LiveCanvas
          canvasRef={sim.canvasRef}
          phase={sim.status.phase}
          label={R.canvasLabel}
          className="h-[min(68svh,720px)] min-h-[360px] w-full"
        />

        {/* Heads-up display while racing. */}
        {t && (phase === 'racing' || phase === 'finished') && (
          <div className="pointer-events-none absolute top-3 left-3 flex gap-3">
            <div className="card px-4 py-2">
              <p className="label">{R.hud.speed}</p>
              <p className="numeral text-[28px] leading-none text-paint">
                {Math.round(t.speedMps * 3.6)} <span className="text-[13px] text-text-2">km/h</span>
              </p>
            </div>
            <div className="card px-4 py-2">
              <p className="label">{R.hud.yourLap}</p>
              <p className="numeral text-[28px] leading-none text-paint">{fmt(t.lapTime)}</p>
            </div>
            <div className="card px-4 py-2" style={accent('var(--drv-student)')}>
              <p className="label">{strings.drivers.student.name}</p>
              <p className="numeral text-[20px] leading-tight text-paint">
                {Number.isFinite(ai) ? R.hud.aiDone(fmt(ai)) : R.hud.aiRacing}
              </p>
            </div>
          </div>
        )}

        {/* Start / countdown / result overlays. */}
        {phase === 'setup' && ready && (
          <div className="absolute inset-0 grid place-items-center bg-asphalt/45">
            <div
              className="card card--accent max-w-[420px] p-6 text-center"
              style={accent('var(--teal)')}
            >
              <p className="text-[18px] font-[620] text-paint">{R.ready}</p>
              <p className="mt-2 text-[14px] text-text-2">{touch ? R.keysTouch : R.keys}</p>
              <button type="button" className="btn-primary mt-5" onClick={start} autoFocus>
                {R.start}
              </button>
            </div>
          </div>
        )}
        {phase === 'countdown' && (
          <div
            className="pointer-events-none absolute inset-0 grid place-items-center"
            aria-live="assertive"
          >
            <span key={count} className="countdown numeral text-[120px] leading-none text-paint">
              {count > 0 ? count : R.go}
            </span>
          </div>
        )}
        {phase === 'finished' && t && (
          <div className="absolute inset-0 grid place-items-center bg-asphalt/55" role="status">
            <div
              className="card card--accent max-w-[460px] p-6 text-center"
              style={accent(won ? 'var(--teal)' : 'var(--coral)')}
            >
              <p className="eyebrow justify-center">{won ? R.youWin : R.aiWins}</p>
              <p className="mt-3 text-[15px] text-text-2">
                {won
                  ? Number.isFinite(margin)
                    ? R.winBy(margin.toFixed(1))
                    : R.winAiStillRacing
                  : R.loseBy(margin.toFixed(1))}
              </p>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-left">
                <div className="card px-3 py-2">
                  <dt className="label">{strings.drivers.human.name}</dt>
                  <dd className="numeral text-[24px] text-paint">{fmt(you)}</dd>
                  <dd className="text-[12px] text-text-3">
                    {R.crashes(t.crashesByDriver.human ?? 0)}
                  </dd>
                </div>
                <div className="card px-3 py-2">
                  <dt className="label">{strings.drivers.student.name}</dt>
                  <dd className="numeral text-[24px] text-paint">{fmt(ai)}</dd>
                  <dd className="text-[12px] text-text-3">
                    {R.crashes(t.crashesByDriver.student ?? 0)}
                  </dd>
                </div>
              </dl>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                <button type="button" className="btn-primary" onClick={start} autoFocus>
                  {R.again}
                </button>
                <Button onClick={() => void share()}>{copied ? R.copied : R.share}</Button>
              </div>
            </div>
          </div>
        )}

        {/* Touch controls. */}
        {touch && phase === 'racing' && (
          <div className="absolute inset-x-0 bottom-3 flex justify-between px-3">
            <div className="flex gap-2">
              {pad('left', R.pads.left, '◀')}
              {pad('right', R.pads.right, '▶')}
            </div>
            <div className="flex gap-2">
              {pad('down', R.pads.brake, '■')}
              {pad('up', R.pads.gas, '▲')}
            </div>
          </div>
        )}
      </div>
      {!touch && <p className="mt-3 text-[13px] text-text-3">{R.keys}</p>}
    </div>
  )
}
