/**
 * Home: the same live scene twice, side by side: Adaptive AI cars on the left, Basic AI cars on the
 * right, on the same track, in the same rain and puddles, started together with the same seed.
 * Mounted only when scrolled near (no cost at page load) and paused while off screen.
 */
import { useEffect, useRef, useState } from 'react'
import { strings } from '../copy/strings.ts'
import { loadShowcase } from '../data/artifacts.ts'
import { useLiveSim, type LiveSimOptions } from '../hooks/useLiveSim.ts'
import { accent } from '../styles/accents.ts'
import { Button } from './Button.tsx'
import { LiveCanvas } from './LiveCanvas.tsx'

const S = strings.sideBySide
const CARS = 8
const RAIN = 0.8 // the puddles suite's rain level (the wet suite spans 0.6–1.0)

function Field({
  opts,
  driver,
  run,
  restartAt,
}: {
  opts: Omit<LiveSimOptions, 'driver'>
  driver: 'student' | 'blind'
  run: boolean
  restartAt: number
}) {
  const sim = useLiveSim({ ...opts, driver })
  const { control } = sim
  const ready = sim.status.phase === 'ready'
  useEffect(() => {
    if (ready) void control(run ? 'resume' : 'pause')
  }, [ready, run, control])
  // Driving time counts from the last restart (restarts zero the crash count; the clock runs on).
  const [base, setBase] = useState({ timeS: 0 })
  const status = sim.status
  useEffect(() => {
    if (!ready || restartAt === 0) return
    void control('reset').then(() => setBase({ timeS: status.simTimeS }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, restartAt, control])
  const crashes = status.crashes
  const driven = Math.max(0, status.simTimeS - base.timeS)
  const tint = driver === 'student' ? 'var(--drv-student)' : 'var(--drv-blind)'
  return (
    <figure className="card card--accent overflow-hidden" style={accent(tint)}>
      <figcaption className="flex items-center justify-between gap-3 px-4 pt-4">
        <span className="flex items-center gap-2 text-[15px] font-[620] text-paint">
          <span className={`driver-swatch driver-swatch--${driver}`} aria-hidden="true" />
          {strings.drivers[driver].name}
        </span>
        <span className="numeral text-[15px] text-paint" aria-live="polite">
          {S.crashesIn(crashes, Math.round(driven))}
        </span>
      </figcaption>
      <p className="px-4 text-[13px] text-text-3">{strings.drivers[driver].what}</p>
      <LiveCanvas
        canvasRef={sim.canvasRef}
        phase={sim.status.phase}
        label={S.canvasLabel(strings.drivers[driver].name)}
        className="mt-3 aspect-[4/3] w-full"
      />
    </figure>
  )
}

function Fields() {
  const root = useRef<HTMLDivElement | null>(null)
  const [visible, setVisible] = useState(true)
  const [restartAt, setRestartAt] = useState(0)
  const [seed] = useState(() => loadShowcase().then((t) => (t[1] ?? t[0]!).seed))
  useEffect(() => {
    const el = root.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => setVisible(!!e?.isIntersecting), {
      threshold: 0.15,
    })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  const opts: Omit<LiveSimOptions, 'driver'> = {
    nCars: CARS,
    trackSeed: seed,
    conditions: { rain: RAIN },
    autoPuddles: true,
    layout: 'spread',
    intro: false,
    seed: 5,
    allowDegrade: false,
  }
  return (
    <div ref={root}>
      <div className="grid gap-4 md:grid-cols-2">
        <Field opts={opts} driver="student" run={visible} restartAt={restartAt} />
        <Field opts={opts} driver="blind" run={visible} restartAt={restartAt} />
      </div>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-[72ch] text-[13px] text-text-3">{S.note}</p>
        <Button variant="secondary" onClick={() => setRestartAt(Date.now())}>
          {S.restart}
        </Button>
      </div>
    </div>
  )
}

/** Section wrapper: mounts the two simulations only when the reader gets close. */
export function SideBySide() {
  const anchor = useRef<HTMLElement | null>(null)
  const [near, setNear] = useState(false)
  useEffect(() => {
    const el = anchor.current
    if (!el || near) return
    const io = new IntersectionObserver(([e]) => e?.isIntersecting && setNear(true), {
      rootMargin: '300px',
    })
    io.observe(el)
    return () => io.disconnect()
  }, [near])
  return (
    <section
      ref={anchor}
      className="mx-auto max-w-[1280px] px-4 pt-16 md:px-6 md:pt-24"
      aria-labelledby="sbs-title"
    >
      <p className="eyebrow" style={accent('var(--sky)')}>
        {S.eyebrow}
      </p>
      <h2 id="sbs-title" className="h2 mt-2">
        {S.title}
      </h2>
      <p className="mt-3 max-w-[68ch] text-text-2">{S.body(CARS, Math.round(RAIN * 100))}</p>
      <div className="mt-8 min-h-[420px]">{near && <Fields />}</div>
    </section>
  )
}
