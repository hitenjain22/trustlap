/** All user-facing strings live here (SPEC 11.11). */
export const strings = {
  wordmark: 'TrustLap',
  nav: {
    playground: 'Playground',
    lab: 'Training lab',
    results: 'Results',
    how: 'How it works',
    github: 'Source code on GitHub',
    menu: 'Menu',
  },
  hero: {
    headline: 'These cars taught themselves to drive in the rain.',
    // Honest for M3: the cars run the "blind" driver, which learned by trial and error but has
    // no grip sense yet (that's the student, M5).
    body: 'Every car here is a neural network running in your browser. It learned to drive by trial and error, through rain, wind and worn tires. Turn up the rain and watch how it copes.',
    rain: 'Rain',
    newTrack: 'New track',
    primary: 'Open the playground',
    secondary: 'Read the build log on GitHub',
    canvasLabel: (cars: number) =>
      `A top-down race track with ${cars} cars driving laps. Rain and puddles change how much grip they have.`,
    readout: (cars: number, lap: number, crashes: number) =>
      `${cars} cars · lap ${lap} · ${crashes} ${crashes === 1 ? 'crash' : 'crashes'}`,
    degraded: (cars: number) => `Running ${cars} cars for smoother playback`,
    liveLabel: 'Running live in your browser',
  },
  story: {
    title: 'How it’s built',
    intro: 'Five pieces, built in order. Each one is live on this site as soon as it works.',
    built: 'Built',
    inProgress: 'In progress',
    steps: [
      {
        title: 'A physics engine built for thousands of cars',
        body: (cars: string, speedup: string, device: string) =>
          `Tire slip, drag, weight, wind and puddles, simulated for ${cars} cars at once on one ${device} GPU, ${speedup}× faster than the same code on a CPU. The browser runs an exact port of it.`,
        pending:
          'Tire slip, drag, weight, wind and puddles, simulated for thousands of cars at once on one GPU.',
      },
      {
        title: 'A population that teaches itself',
        body: 'Next: a population of drivers trains side by side. The weakest copy the strongest and tweak how they learn.',
      },
      {
        title: 'A world that gets harder as they improve',
        body: 'The simulator raises the difficulty on its own as the drivers improve: more rain, worn tires, laggy steering, gusty wind.',
      },
      {
        title: 'A driver that feels the road',
        body: 'Next: a second network reads the last second of motion and estimates the grip under each axle, without being told the road is wet.',
      },
      {
        title: 'A driver that knows when it’s unsure',
        body: 'Next: five copies of the sensing network vote. When they disagree, the car slows down before it crashes.',
      },
    ],
    benchSource: (date: string, sha: string) =>
      `From bench.json · measured ${date} · commit ${sha}`,
  },
  playground: {
    title: 'Playground',
    intro:
      'Change the conditions and watch how the cars cope. The shaded part of each slider is the range the drivers will train on.',
    track: 'Track',
    newTrack: 'New track',
    seed: (seed: number) => `Seed ${seed}`,
    driver: 'Driver',
    drivers: { scripted: 'Scripted', student: 'Student', blind: 'Blind', teacher: 'Teacher' },
    ghosts: 'Ghosts',
    notTrainedYet: 'Not trained yet',
    canvasLabelLive: (driver: string) =>
      `A ${driver} car driving on the selected track, with any ghost cars you added.`,
    play: 'Play',
    pause: 'Pause',
    speed: 'Speed',
    restart: 'Restart lap',
    groups: { weather: 'Weather', car: 'Car', electronics: 'Electronics' },
    canvasLabel: 'A self-taught car driving on the selected track, with a scripted ghost car.',
  },
  results: {
    title: 'Results',
    intro: (tracks: number, seeds: number) =>
      `Every driver raced the same ${tracks} unseen test tracks under each condition, ${seeds} condition draws per track: ${tracks * seeds} laps per driver per condition. Ranges are 95% confidence intervals from a paired bootstrap over tracks.`,
    introMissing: 'The evaluation hasn’t been run for this build yet.',
    notComputed: 'Not computed yet.',
    showTable: 'Show table',
    showChart: 'Show chart',
    source: (date: string, sha: string) => `From metrics.json · generated ${date} · commit ${sha}`,
    benchSource: (date: string, sha: string) =>
      `From bench.json · measured ${date} · commit ${sha}`,
    tiles: {
      crashReduction: 'Fewer crashes than the blind driver',
      crashReductionNote: 'Student vs blind, everything-at-once conditions',
      teacherGap: 'Teacher gap left',
      teacherGapNote:
        'Share of the teacher’s advantage over the blind driver the student did not recover (0% = matches the teacher)',
      gripMae: 'Grip estimate error',
      gripMaeNote: 'Mean absolute error of the student’s grip estimate, training-range conditions',
      latency: 'Time to notice a puddle',
      latencyNote: (lo: number, hi: number) => `Median; middle half ${lo}–${hi} ms`,
      cars: 'Cars simulated at once',
      carsNote: (speedup: string) => `On one GPU, ${speedup}× faster than the same code on a CPU`,
      compute: 'Training compute',
      computeNote: (minutes: string, hw: string) => `${minutes} minutes on a ${hw}`,
    },
    figures: {
      completion: {
        neutral: 'Lap completion by condition',
        title: (wins: number, total: number) =>
          wins === total
            ? `The adaptive student finishes at least as many laps as the blind driver in every condition`
            : `The adaptive student finishes more laps than the blind driver in ${wins} of ${total} conditions`,
        subtitle:
          'Lap completion rate by condition (unseen test tracks), sorted by the student’s advantage over the blind driver.',
        axis: 'Laps completed',
      },
      delta: {
        neutral: 'Student vs blind, paired difference',
        title: (better: number, total: number) =>
          `Paired difference: the student’s completion rate is clearly above the blind driver’s in ${better} of ${total} conditions`,
        subtitle:
          'Student minus blind, same tracks and conditions. Whiskers that cross zero mean no clear difference.',
        axis: 'Completion difference (percentage points)',
      },
      lapTime: {
        neutral: 'Paired lap time',
        title: 'Lap times on laps every driver finished',
        subtitle:
          'Paired lap time: only laps that all five drivers completed, so slow-but-safe drivers aren’t flattered by survivor bias.',
        axis: 'Lap time (s)',
      },
      ood: {
        neutral: 'Out-of-distribution conditions',
        title: 'Never trained on these: how each driver copes far outside its training range',
        subtitle:
          'Out-of-distribution conditions: ice, extreme lag, very noisy sensors, very heavy car.',
        axis: 'Laps completed',
      },
      puddle: {
        title: (ms: number) =>
          `The student notices a puddle under the rear wheels in about ${ms} ms`,
        titleMissing: 'Feeling a puddle',
        subtitle:
          'True rear grip (blue) vs the student’s estimate (amber) as the car crosses a single puddle on a straight. Time 0 is when the true grip first drops by more than 0.1.',
        x: 'Time from entering the puddle (ms)',
        y: 'Rear grip μ',
      },
      reliability: {
        neutral: 'Does the crash-risk estimate mean what it says?',
        title: (ece: string) =>
          `When the car says “x% risk”, it crashes about x% of the time (calibration error ${ece})`,
        subtitle:
          'Predicted vs observed chance of crashing within 2 s, 10 equal-size bins (test tracks). The dashed diagonal is perfect calibration.',
        x: 'Predicted crash risk',
        y: 'Observed crash rate',
      },
      governor: {
        neutral: 'Safety vs speed',
        title: 'The safety governor trades lap time for fewer crashes',
        subtitle:
          'Each point is a risk threshold τ: above it the car eases off. Everything-at-once conditions, paired lap times.',
        x: 'Paired lap time (s)',
        y: 'Crash rate',
      },
      bench: {
        neutral: 'GPU benchmark',
        title: (speedup: string, cars: string) =>
          `One GPU runs ${cars} cars at once, ${speedup}× faster than the same code on its CPU`,
        subtitle:
          'Simulation throughput (car-steps per second) vs number of cars, log–log. Measured points only, nothing extrapolated.',
        x: 'Cars simulated at once',
        y: 'Car-steps per second',
      },
      generalization: {
        neutral: 'Unseen vs seen tracks',
        title: 'Unseen tracks are about as hard as the ones it trained on',
        subtitle:
          'Completion on training-set tracks vs held-out test tracks, everything-at-once conditions.',
        axis: 'Laps completed',
      },
    },
    limitsTitle: 'What this doesn’t show',
    limits: [
      'Everything here is simulated. A real car has suspension, load transfer, four tires and sensors that fail in ways this model doesn’t.',
      'The world is flat and two-dimensional, with one kind of car.',
      'All five students share one teacher’s driving policy, so their disagreement measures uncertainty about the road, not about the driving itself.',
      'Training ran at reduced scale on a laptop CPU (exact compute above). Bigger runs could change every number.',
      'Results that don’t favor the student are shown as they came out.',
    ],
  },
  controls: {
    outsideTraining: 'Outside what it trained on.',
  },
  states: {
    loadError: 'Couldn’t load the simulator files. Check your connection and reload.',
    reload: 'Reload',
    noCanvas: 'This page needs a browser with Canvas and Web Worker support to run the simulator.',
  },
  notFound: {
    title: 'This page drove off the track.',
    body: 'There’s nothing at this address.',
    home: 'Back to the start line',
  },
  footer: {
    credit: 'Built by Hiten Jain, UC Davis.',
    attribution:
      'Tire model: Pacejka’s “Magic Formula”. Training recipe and methods credited as they land.',
    source: 'Source on GitHub',
    version: (artifact: string, sha: string) => `Artifacts ${artifact} · build ${sha}`,
  },
} as const

export const REPO_URL = 'https://github.com/hitenjain22/trustlap'
