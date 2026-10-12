import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router'
import { Button } from '../components/Button.tsx'
import { DiceIcon, LineIcon } from '../components/icons.tsx'
import { accent, STEP_ACCENTS, STEP_ICONS } from '../styles/accents.ts'
import { LiveCanvas } from '../components/LiveCanvas.tsx'
import { Slider } from '../components/Slider.tsx'
import { REPO_URL, strings } from '../copy/strings.ts'
import { bestFitTrack, loadMetrics, loadShowcase } from '../data/artifacts.ts'
import type { Metrics } from '../data/schemas.ts'
import { compareConditions, TRAINED_CONDITIONS } from '../data/summary.ts'
import { useLiveSim } from '../hooks/useLiveSim.ts'

const HERO_CARS = 48
// Phone budget (SPEC 10.2 mobile, 13): 16 cars and at most 400 rain streaks.
const PHONE_CARS = 16
const PHONE_RAIN_STREAKS = 400
const isPhone = () => window.matchMedia('(max-width: 767px)').matches
const INITIAL_RAIN = 0.4

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
          className="relative mx-4 -mt-10 mb-6 max-w-[560px] rounded-panel border border-[var(--edge)] bg-asphalt/85 p-6 shadow-[inset_0_1px_0_rgb(255_255_255/0.05),0_30px_60px_-30px_rgb(0_0_0/0.9)] md:absolute md:bottom-8 md:left-[max(24px,calc((100vw-1280px)/2+24px))] md:m-0 md:bg-asphalt/80"
        >
          <h1 className="display">
            {strings.hero.headlineLead}{' '}
            <span className="text-gradient">{strings.hero.headlineAccent}</span>
          </h1>
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
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  useEffect(() => {
    loadMetrics().then(setMetrics, () => setMetrics(null))
  }, [])
  // The answer in one breath, generated from metrics.json (same rule as the Results page).
  const rows = metrics ? compareConditions(metrics, TRAINED_CONDITIONS) : []
  const names = (v: 'better' | 'worse') =>
    rows
      .filter((r) => r.verdict === v)
      .map((r) => strings.conditions[r.suite as keyof typeof strings.conditions].name.toLowerCase())
  const better = names('better')
  const worse = names('worse')
  const H = strings.how
  return (
    <section
      className="mx-auto max-w-[1280px] px-4 py-16 md:px-6 md:py-24"
      aria-labelledby="story-title"
    >
      <h2 id="story-title" className="h2">
        {strings.story.title}
      </h2>
      <p className="mt-3 max-w-[68ch] text-text-2">{strings.story.intro}</p>
      <ol className="mt-10 grid gap-4 md:grid-cols-5">
        {H.steps.map((step, i) => (
          <li key={step.tab} className="flex">
            <Link
              to={`/how-it-works#step-${i + 1}`}
              className="card card--hover card--accent flex flex-1 flex-col p-5 no-underline"
              style={accent(STEP_ACCENTS[i]!)}
            >
              <span className="flex items-center gap-3">
                <span className="icon-badge">
                  <LineIcon name={STEP_ICONS[i]!} />
                </span>
                <span className="text-[13px] font-[600] text-text-3">{`Step ${i + 1}`}</span>
              </span>
              <span className="mt-4 text-[17px] leading-snug font-[620] text-paint">
                {step.title}
              </span>
              <span className="mt-2 flex-1 text-[14px] text-text-2">{step.summary}</span>
              <span className="mt-4 text-[14px] font-[560]" style={{ color: STEP_ACCENTS[i] }}>
                {strings.story.readStep(i + 1)} →
              </span>
            </Link>
          </li>
        ))}
      </ol>
      {rows.length > 0 && (
        <div className="card card--accent mt-10 p-6" style={accent('var(--teal)')}>
          <h3 className="eyebrow">{strings.story.answerTitle}</h3>
          <p className="mt-2 max-w-[72ch] text-text-2">
            {strings.story.answerLead}{' '}
            {better.length > 0 && strings.story.answerBetter(better.join(', '))}{' '}
            {worse.length > 0 && strings.story.answerWorse(worse.join(', '))}
          </p>
          <Link to="/results" className="mt-4 inline-block text-sodium underline">
            {strings.story.answerLink}
          </Link>
        </div>
      )}
    </section>
  )
}
