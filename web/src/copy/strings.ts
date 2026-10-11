/** All user-facing strings live here (SPEC 11.11). */
export const strings = {
  wordmark: 'TrustLap',
  nav: {
    playground: 'Playground',
    github: 'Source code on GitHub',
    menu: 'Menu',
  },
  hero: {
    // Interim copy for M2: these cars run the hand-written baseline. The final headline
    // ("These cars taught themselves to drive in the rain.") arrives with the trained drivers.
    headline: 'Watch 48 cars race through the rain, live in your browser.',
    body: 'Every car runs the same tire physics the AI drivers train on, computed right here on your device. For now they follow a hand-written script that knows the exact grip. The self-taught drivers come next. Turn up the rain and watch them back off.',
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
        body: 'Next: the simulator raises the difficulty on its own, with more rain, worn tires, laggy steering and gusty wind.',
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
    driversSoon: 'The trained drivers arrive in the next build.',
    play: 'Play',
    pause: 'Pause',
    speed: 'Speed',
    restart: 'Restart lap',
    groups: { weather: 'Weather', car: 'Car', electronics: 'Electronics' },
    canvasLabel: 'Three scripted cars driving on the selected track.',
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
