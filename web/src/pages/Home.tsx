import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router'
import { Button } from '../components/Button.tsx'
import { DiceIcon } from '../components/icons.tsx'
import { LiveCanvas } from '../components/LiveCanvas.tsx'
import { Slider } from '../components/Slider.tsx'
import { REPO_URL, strings } from '../copy/strings.ts'
import {
  benchHeadline,
  bestFitTrack,
  loadBench,
  loadMetrics,
  loadShowcase,
  type Bench,
} from '../data/artifacts.ts'
import type { Metrics } from '../data/schemas.ts'
import { useLiveSim } from '../hooks/useLiveSim.ts'

const HERO_CARS = 48
// Phone budget (SPEC 10.2 mobile, 13): 16 cars and at most 400 rain streaks.
const PHONE_CARS = 16
const PHONE_RAIN_STREAKS = 400
const isPhone = () => window.matchMedia('(max-width: 767px)').matches
const INITIAL_RAIN = 0.4
/** Story steps that are live on the site (0-based). */
const BUILT_STEPS = [0, 1, 2, 3, 4]

/** Home: live hero with self-taught cars + the build story (SPEC 10.2). */
export default function Home() {
  const panelRef = useRef<HTMLDivElement | null>(null)
  // Pick the showcase track whose shape best fits the space next to the text (desktop) or the
  // canvas above it (mobile).
  const firstSeed = useMemo(
    () =>
      loadShowcase().then((tracks) => {
        const wide = window.matchMedia('(min-width: 768px)').matches
        const regionAspect = wide
          ? (window.innerWidth * 0.6) / (window.innerHeight - 56)
          : window.innerWidth / (window.innerHeight * 0.58)
        return bestFitTrack(tracks, regionAspect).seed
      }),
    [],
  )
  const [phone] = useState(isPhone) // decided once: the sim is sized at start-up
  const sim = useLiveSim({
    nCars: phone ? PHONE_CARS : HERO_CARS,
    maxRainStreaks: phone ? PHONE_RAIN_STREAKS : undefined,
    trackSeed: firstSeed,
    conditions: { rain: INITIAL_RAIN },
    autoPuddles: true,
    layout: 'grid',
    // The adaptive student when this build has it (falls back to the best shipped driver).
    // One ensemble member keeps 48 cars light enough for phones.
    driver: 'student',
    ensembleMembers: 1,
    intro: true,
    seed: 7,
    fitInsetLeft: () => {
      const panel = panelRef.current
      // Only when the text overlays the canvas (desktop layout).
      return panel && window.matchMedia('(min-width: 768px)').matches
        ? panel.getBoundingClientRect().right + 16
        : 0
    },
  })
  const [rain, setRain] = useState(INITIAL_RAIN)
  const { status } = sim

  const onRain = (value: number) => {
    setRain(value)
    void sim.setConditions({ rain: value })
  }
  const newTrack = () => void sim.setTrack(Math.floor(Math.random() * 2 ** 31))

  return (
    <>
      <section
        className="relative flex flex-col md:block md:h-[calc(100svh-56px)] md:min-h-[640px]"
        aria-label="Live simulator"
      >
        <LiveCanvas
          canvasRef={sim.canvasRef}
          phase={status.phase}
          label={strings.hero.canvasLabel(status.cars)}
          className="h-[58svh] min-h-[340px] md:absolute md:inset-0 md:h-full"
        />
        <div
          ref={panelRef}
          className="relative mx-4 -mt-10 mb-6 max-w-[560px] rounded-panel border border-line bg-asphalt/85 p-6 md:absolute md:bottom-8 md:left-[max(24px,calc((100vw-1280px)/2+24px))] md:m-0 md:bg-asphalt/80"
        >
          <h1 className="display">{strings.hero.headline}</h1>
          <p className="mt-4 text-text-2">{strings.hero.body}</p>
          <div className="mt-6 flex items-end gap-4">
            <div className="flex-1">
              <Slider
                label={strings.hero.rain}
                value={rain}
                min={0}
                max={1}
                step={0.01}
                onChange={onRain}
                format={(v) => `${Math.round(v * 100)}%`}
                valueText={(v) => `${Math.round(v * 100)} percent rain`}
                size="large"
              />
            </div>
            <Button onClick={newTrack} aria-label={strings.hero.newTrack}>
              <DiceIcon />
              <span className="hidden sm:inline">{strings.hero.newTrack}</span>
            </Button>
          </div>
          <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3">
            <Link to="/playground" className="btn-primary">
              {strings.hero.primary}
            </Link>
            <a href={REPO_URL} className="text-[15px] text-sodium underline">
              {strings.hero.secondary}
            </a>
          </div>
        </div>
        <p
          className="readout px-4 pb-4 text-right md:absolute md:right-6 md:bottom-8 md:p-0"
          aria-hidden="true"
        >
          {status.degraded ? strings.hero.degraded(status.cars) + ' · ' : ''}
          {strings.hero.readout(status.cars, status.leaderLap, status.crashes)}
        </p>
      </section>
      <Story />
    </>
  )
}

function Story() {
  const [bench, setBench] = useState<Bench | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  useEffect(() => {
    loadBench().then(setBench, () => setBench(null))
    loadMetrics().then(setMetrics, () => setMetrics(null))
  }, [])
  // Only claim a reaction time if the student noticed the puddle in most trials.
  const lat = metrics?.headline?.puddle_latency_ms
  const latency =
    lat?.median != null && (lat.noticed == null || !lat.trials || lat.noticed >= lat.trials / 2)
      ? lat.median
      : null
  const ece = metrics?.calibration?.ece
  const head = bench ? benchHeadline(bench) : null
  const steps = strings.story.steps
  return (
    <section
      className="mx-auto max-w-[1280px] px-4 py-16 md:px-6 md:py-24"
      aria-labelledby="story-title"
    >
      <h2 id="story-title" className="h2">
        {strings.story.title}
      </h2>
      <p className="mt-3 max-w-[68ch] text-text-2">{strings.story.intro}</p>
      <ol className="mt-12 grid gap-x-12 gap-y-10 md:grid-cols-2">
        {steps.map((step, i) => {
          const built = BUILT_STEPS.includes(i)
          const body =
            i === 0
              ? head
                ? steps[0].body(
                    head.maxCars.toLocaleString('en-US'),
                    Math.round(head.speedup).toLocaleString('en-US'),
                    head.device,
                  )
                : steps[0].pending
              : (step as { body: string }).body
          return (
            <li key={step.title} className="border-t border-line pt-6">
              <p className="flex items-center gap-3 text-[14px] text-text-3">
                <span className="numeral text-[20px] text-paint">{i + 1}</span>
                <span className={built ? 'status-chip status-chip--built' : 'status-chip'}>
                  {built ? strings.story.built : strings.story.inProgress}
                </span>
              </p>
              <h3 className="h3 mt-3">{step.title}</h3>
              <p className="mt-2 max-w-[52ch] text-text-2">
                {body}
                {i === 3 && latency != null && ` ${steps[3].measured(Math.round(latency))}`}
                {i === 4 && ece != null && ` ${steps[4].measured(ece.toFixed(3))}`}
              </p>
              {i === 0 && bench && (
                <p className="mt-3 text-[13px] text-text-3">
                  {strings.story.benchSource(bench.created_utc.slice(0, 10), bench.git_sha)}
                </p>
              )}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
