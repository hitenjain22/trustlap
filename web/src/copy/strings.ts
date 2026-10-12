/** All user-facing strings live here (SPEC 11.11). */
/**
 * The four drivers in plain words (internal ids stay student / blind / teacher / scripted).
 * Used everywhere a visitor sees a driver: buttons, legends, tables, toasts.
 */
const DRIVERS = {
  student: {
    name: 'Adaptive AI',
    what: 'Feels the road: estimates the grip under its tires from the last second of motion.',
  },
  blind: {
    name: 'Basic AI',
    what: 'Trained the same way, but can’t sense grip. The comparison point.',
  },
  teacher: {
    name: 'Teacher AI',
    what: 'Is told the true conditions. A training aid: a real car can’t know them.',
  },
  scripted: {
    name: 'Rule-based driver',
    what: 'Hand-written rules, given the exact grip. A reference, not AI.',
  },
} as const

/** Evaluation conditions in plain words (metrics.json suite ids). */
const CONDITIONS = {
  dry: { name: 'Dry road', what: 'Nothing changed: the easy case' },
  wet: { name: 'Rain', what: 'Less grip everywhere' },
  puddles: { name: 'Puddles', what: 'Grip drops in patches' },
  worn: { name: 'Worn tires', what: 'Grip fades as the lap goes on' },
  wind: { name: 'Wind', what: 'Gusts push the car sideways' },
  heavy: { name: 'Heavy car', what: 'More weight, less power' },
  laggy: { name: 'Laggy steering', what: 'The wheel reacts late' },
  combined: { name: 'Everything at once', what: 'All of the above together' },
  ice: { name: 'Ice', what: 'Far less grip than it ever practised on' },
  extreme_lag: { name: 'Extreme lag', what: 'Controls react very late' },
  noisy_sensors: { name: 'Noisy sensors', what: 'Its readings are badly off' },
  very_heavy: { name: 'Very heavy car', what: 'Heavier than it practised with' },
} as const

export const strings = {
  drivers: DRIVERS,
  conditions: CONDITIONS,
  wordmark: 'TrustLap',
  /** Matches index.html's <title> (used on the home page). */
  siteTitle: 'TrustLap: AI cars that teach themselves to drive in the rain',
  pageTitles: {
    '/playground': 'Playground',
    '/lab': 'Training lab',
    '/results': 'Results',
    '/how-it-works': 'How it works',
  },
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
    // no grip sense yet (that's the Adaptive AI, M5).
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
    title: 'How it works, in five steps',
    intro: 'From an empty simulator to cars that feel the road. Each step is live on this site.',
    readStep: (n: number) => `Read step ${n}`,
    answerTitle: 'Does it work?',
    answerLead: 'Compared with the same AI without a sense of grip:',
    answerBetter: (list: string) => `clearly more laps finished: ${list};`,
    answerWorse: (list: string) => `clearly fewer: ${list}.`,
    answerLink: 'See the full results',
  },
  playground: {
    title: 'Playground',
    intro:
      'Pick a scenario or change the conditions, and watch how the cars cope. The shaded part of each slider is the range they practised in; outside it, they’ve never been.',
    scenariosTitle: 'Try a scenario',
    scenarios: {
      dry: {
        name: 'Dry practice',
        watch: 'The baseline: no rain, full grip. Compare the cars’ lines and lap times.',
      },
      monsoon: {
        name: 'Monsoon',
        watch: 'Heavy rain, puddles and wind. Watch the grip gauge drop and how each car slows.',
      },
      ice: {
        name: 'Black ice',
        watch: 'Far less grip than anything they practised on. Watch the crash-risk meter.',
      },
      broken: { name: 'Broken car', watch: 'Laggy steering, delayed controls and noisy sensors.' },
      chaos: {
        name: 'Everything at once',
        watch: 'Rain, wind, a heavier and weaker car, and laggy controls together.',
      },
      puddle: {
        name: 'Puddle test',
        watch: 'One puddle on a long straight. Does the Adaptive AI’s grip estimate drop?',
      },
    } as Record<string, { name: string; watch: string }>,
    hints: {
      rain: 'Wets the whole track: less grip everywhere.',
      muDry: 'How grippy the road is. Low is icy.',
      puddles: 'Slippery patches the cars have to cross.',
      windMps: 'A steady push sideways.',
      windDirDeg: 'Which way the wind blows.',
      gustMps: 'Random bursts on top of the wind.',
      massMult: 'A heavier car is slower to turn and stop.',
      powerMult: 'Engine power: how hard it accelerates.',
      brakeMult: 'How hard it can brake.',
      dragAreaM2: 'Air resistance: lowers the top speed.',
      wearPerKm: 'Tires lose grip as the lap goes on.',
      steerTauS: 'How slowly the wheels follow the steering.',
      delaySteps: 'The pause between deciding and the car responding.',
    } as Record<string, string>,
    track: 'Track',
    newTrack: 'New track',
    seed: (seed: number) => `Seed ${seed}`,
    driver: 'Driver',
    drivers: {
      scripted: DRIVERS.scripted.name,
      student: DRIVERS.student.name,
      blind: DRIVERS.blind.name,
      teacher: DRIVERS.teacher.name,
    },
    ghosts: 'Race against',
    notTrainedYet: 'Not trained yet',
    canvasLabelLive: (driver: string) =>
      `A ${driver} car driving on the selected track, with any ghost cars you added.`,
    play: 'Play',
    pause: 'Pause',
    speed: 'Speed',
    restart: 'Restart lap',
    groups: {
      weather: 'Weather',
      more: 'More conditions',
      car: 'Car',
      electronics: 'Electronics',
      safety: 'Safety',
    },
    governor: 'Safety governor',
    copyLink: 'Copy link',
    linkCopied: 'Link copied.',
    linkCopyFailed: 'Couldn’t copy. The address bar has the link.',
    shortcuts: 'Keyboard shortcuts',
    keys: [
      ['Space', 'Play / pause'],
      ['R', 'Restart lap'],
      ['N', 'New track'],
      ['D', 'Draw a track'],
      ['C', 'Switch camera (overview / chase)'],
      ['F', 'Show tire forces'],
      ['1–4', 'Adaptive, Basic, Teacher, Rule-based'],
      ['?', 'Show these shortcuts'],
    ] as [string, string][],
    governorNote:
      'Adaptive AI only: eases off the throttle when its crash-risk estimate passes the threshold.',
    governorThreshold: 'Risk threshold',
    canvasLabel: 'A self-taught car driving on the selected track, with a ghost car.',
    drawTrack: 'Draw a track',
    crashes: 'Crashes',
    camera: 'Camera',
    conditions: 'Conditions',
    cameras: { overview: 'Overview', chase: 'Chase' },
    showForces: 'Show tire forces',
    crashToast: (c: {
      kmh: number
      driver: string
      puddle: boolean
      corner: boolean
      spun: boolean
      stalled: boolean
      wet: boolean
      governor: 'on' | 'off' | null
    }) => {
      if (c.stalled) return `The ${c.driver} car stalled and stopped moving.`
      const how = c.spun ? 'Spun out' : 'Went off the track'
      const why = c.puddle ? ' after hitting a puddle' : c.wet ? ' on a wet road' : ''
      const where = c.corner ? ' mid-corner' : ' on a straight'
      const gov = c.governor === null ? '' : ` The safety governor was ${c.governor}.`
      return `${how} at ${c.kmh} km/h${why}${where}.${gov}`
    },
    dismiss: 'Dismiss',
    drawnTrack: 'Your drawn track',
    drawnNoLink: 'Drawn tracks can’t be shared by link yet. The link keeps everything else.',
  },
  puddleTest: {
    track: 'Puddle test straight',
    title: 'Puddle test',
    body: 'The Adaptive AI drives down a long straight at half speed and crosses one puddle (grip × 0.4). Does its grip estimate drop while it’s on the puddle? Scored with the same rule as the evaluation.',
    chartLabel: 'Rear grip since the test started: true grip and the Adaptive AI’s estimate.',
    truth: 'True rear grip',
    estimate: 'Adaptive AI’s estimate',
    waiting: 'Driving toward the puddle…',
    noticed: (ms: number) => `Noticed the slippery patch in ${ms} ms.`,
    missed:
      'Didn’t notice the slippery patch: its estimate didn’t drop by half the true drop while it was on the puddle. That matches the evaluation (Results).',
    noEstimate: 'This driver doesn’t estimate grip. The puddle test needs the Adaptive AI.',
    runAgain: 'Run again',
    end: 'End the puddle test',
  },
  drawTrack: {
    title: 'Draw a track',
    hint: 'Drag one loop and finish inside the dashed circle where you started. Or generate a random track.',
    notClosed: 'Finish near where you started to close the loop, then let go.',
    valid: 'Looks good. Use this track to race on it.',
    fixed: 'Smoothed. Use this track to race on it.',
    reasons: {
      self_intersection: 'The track crosses itself. Redraw it, or try Auto-fix.',
      pinch: 'Two parts of the track come too close together. Try Auto-fix.',
      too_tight: 'A corner is too tight for the cars. Try Auto-fix.',
      too_short: 'The track is too short. Try Auto-fix or draw a bigger loop.',
      too_long: 'The track is too long. Try Auto-fix.',
    } as Record<'self_intersection' | 'pinch' | 'too_tight' | 'too_short' | 'too_long', string>,
    use: 'Use this track',
    autoFix: 'Auto-fix',
    clear: 'Clear',
    random: 'Generate a random track',
    cancel: 'Cancel',
  },
  telemetry: {
    title: 'Telemetry for the focused car',
    waiting: 'Waiting for the first lap data…',
    unitToggle: 'Switch speed units',
    speedShort: 'Speed',
    gripShort: (est: string, truth: string) => `feels ${est} · real ${truth}`,
    lap: 'Lap',
    best: 'Best',
    delta: 'Δ vs ghost',
    crashed: 'Crashed',
    front: 'Front grip',
    rear: 'Rear grip',
    gripReading: (est: string, truth: string) => `feels ${est} · real ${truth}`,
    gripKey: 'Grip: the bar is what the car feels, the blue tick is the real grip.',
    moreTelemetry: 'More telemetry',
    lessTelemetry: 'Less telemetry',
    gripTrueOnly: (truth: string) => `real ${truth} · this driver can’t feel grip`,
    gripAria: (axle: string, est: string, truth: string) => `${axle}: feels ${est}, real ${truth}`,
    risk: 'Crash risk',
    slowsAbove: (pct: number) => `(slows above ${pct}%)`,
    governorOff: '(governor off)',
    riskStudentOnly: 'Only the Adaptive AI estimates its crash risk.',
    sparkSpeed: 'Speed, last 10 s',
    sparkGrip: 'Rear grip: true vs estimate',
    sparkRisk: 'Crash risk',
    steering: 'Steering (left ← → right)',
    pedals: 'Brake ← → throttle',
    steerAria: (pct: number) => `Steering ${Math.abs(pct)} percent ${pct >= 0 ? 'left' : 'right'}`,
    pedalAria: (pct: number) => (pct >= 0 ? `Throttle ${pct} percent` : `Brake ${-pct} percent`),
  },
  how: {
    title: 'How it works',
    lead: 'Five steps, from an empty simulator to cars that drive in the rain. Each step uses a published technique; what’s new is putting them together and testing them honestly.',
    next: 'Next step',
    detailsLabel: 'The technical details',
    steps: [
      {
        tab: 'Build a world',
        summary: 'A physics simulator with grip, rain, puddles, wind and worn tires.',
        title: 'Build a world to practise in',
        body: [
          'A physics simulator of a car on a race track: tires that grip and then slide, rain that lowers grip everywhere, puddles that lower it in patches, wind, worn tires, heavy cars and laggy steering.',
          'Learning to drive takes millions of laps, so the simulator runs thousands of cars at once on a graphics card. Your browser runs an exact copy of the same physics, which is why the cars on this site are the real thing.',
        ],
        stat: (cars: string, x: string) => ({
          value: cars,
          meaning: `cars simulated at once on one GPU, ${x}× faster than the same code on its CPU`,
        }),
        components: ['sim', 'browser'],
      },
      {
        tab: 'Practise',
        summary: 'AI drivers learn by trial and error, millions of laps over.',
        title: 'Practise millions of laps',
        body: [
          'Each AI driver starts knowing nothing. It drives, crashes, and slowly learns what works: reinforcement learning, rewarded for making progress around the track without crashing.',
          'Two tricks keep it learning. The world gets harder on its own as the drivers improve (more rain, worn tires, gusts). And a whole population trains side by side: every so often the weakest copies the strongest and tries new settings.',
        ],
        stat: (steps: string, minutes: string) => ({
          value: steps,
          meaning: `driving decisions practised in total, in ${minutes} minutes on one GPU`,
        }),
        components: ['ppo', 'curriculum'],
        link: { to: '/lab', label: 'Watch it learn in the Training lab' },
      },
      {
        tab: 'Feel the road',
        summary: 'The Adaptive AI learns to sense grip from how the car moves.',
        title: 'Learn to feel the road',
        body: [
          'First, a Teacher AI practises while being told the true conditions: the grip under each axle, the wind, the car’s weight. A real car can’t know these, so the teacher is only a training aid.',
          'Then the Adaptive AI learns to work those conditions out from how the car moved in the last second (speed, slide, steering), and drives the way the teacher would. The Basic AI trains the same way but never gets this sense.',
        ],
        stat: (err: string, dry: string, ice: string) => ({
          value: `± ${err}`,
          meaning: `average error of its grip estimate, on a scale where dry asphalt grips at about ${dry} and ice at ${ice}`,
        }),
        components: ['teacher', 'student'],
        link: { to: '/playground', label: 'Try the puddle test in the Playground' },
      },
      {
        tab: 'Know when unsure',
        summary: 'Five guesses that disagree mean it’s on unfamiliar ground.',
        title: 'Know when it’s unsure',
        body: [
          'Five copies of the road-feeling network make their guesses separately. When they disagree, the car is probably somewhere it doesn’t understand.',
          'That disagreement becomes a crash-risk percentage, and an optional safety governor eases off the throttle when the risk gets high.',
        ],
        stat: (ece: string) => ({
          value: ece,
          meaning:
            'calibration error: when it says “20% risk”, about 20% of those moments really end in a crash (0 would be perfect)',
        }),
        components: ['confidence'],
      },
      {
        tab: 'Test it fairly',
        summary: 'Every driver races the same unseen tracks, in every condition.',
        title: 'Test it fairly',
        body: [
          'Every driver races the same tracks it has never seen, in the same conditions, from the same start. That makes the comparison fair: any difference comes from the driver, not luck.',
          'Results come with 95% ranges, and the ones that don’t favor the Adaptive AI are shown too.',
        ],
        stat: (tracks: string, conds: string) => ({
          value: tracks,
          meaning: `unseen test tracks, each raced in ${conds} conditions`,
        }),
        components: [],
        link: { to: '/results', label: 'See the results' },
      },
    ],
    moreTitle: 'Model card and how to reproduce',
    labels: {
      what: 'What it is',
      why: 'Why it’s here',
      measured: 'What I measured',
      borrowed: 'Borrowed from',
    },
    components: [
      {
        id: 'sim',
        title: 'The simulator',
        what: 'A dynamic bicycle model: front and rear tires with Pacejka slip curves and a friction ellipse, drag, rolling resistance, wind and gusts, steering lag and input delay, puddles that change the grip under each axle. Written in JAX so one function steps tens of thousands of cars at once.',
        why: 'Learning needs billions of driving decisions. A GPU can only provide them if the whole world is array math, and the browser can only show the same car if the physics is simple enough to port exactly.',
        measured:
          'Throughput on a T4 GPU vs its CPU and a plain Python loop (Results, benchmark). The TypeScript port matches Python to within 1.4% of a 0.0001 tolerance on 2,000 random cases.',
        borrowed:
          'Pacejka’s “Magic Formula” tire model; the vectorized-environment pattern from PureJaxRL (Lu et al., 2022).',
      },
      {
        id: 'curriculum',
        title: 'The automatic curriculum',
        what: 'One difficulty dial from 0 to 1 widens every condition range: rain, grip, wind, tire wear, mass, power, brakes, lag, delay and sensor noise. It rises when drivers finish over 80% of laps twice in a row and backs off below 50%.',
        why: 'Starting in a storm, nothing learns. Starting dry and stopping there, nothing generalizes. The dial keeps the task hard but learnable.',
        measured:
          'The difficulty reached in training (Training lab). Even at the full training budget neither population reached the top of the dial, and the conditions beyond what they reached are where the drivers fail (Results).',
        borrowed:
          'Inspired by Automatic Domain Randomization (OpenAI et al., 2019); this is a single global dial, not the full per-parameter version.',
      },
      {
        id: 'ppo',
        title: 'Population-based PPO',
        what: 'Proximal Policy Optimization: the driver improves a little at a time, never changing too much in one update. Several drivers train side by side; every 25 updates the weakest copy the strongest and nudge their learning rate and exploration.',
        why: 'PPO is stable on continuous control. The population tunes its own learning settings instead of me guessing them.',
        measured:
          'Fitness curves, the family tree of copies and the hyperparameters the winners chose (Training lab).',
        borrowed:
          'PPO (Schulman et al., 2017); the single-jitted-loop training recipe from PureJaxRL (Lu et al., 2022); Population Based Training (Jaderberg et al., 2017).',
      },
      {
        id: 'teacher',
        title: 'The Teacher AI',
        what: 'A driver that is told the true conditions (grip under each axle, puddles ahead, mass, wind…) through a small encoder network that compresses them into 8 numbers.',
        why: 'It shows how well a driver can do when it knows everything, and its 8-number summary is what the Adaptive AI learns to estimate.',
        measured:
          'Completion in every condition (Results). It beats the Basic AI in the wet, on puddles, worn tires and heavier cars, but not in wind or with noisy sensors; each teacher trained for two-thirds of the Basic AI’s steps.',
        borrowed: 'Phase 1 of Rapid Motor Adaptation (Kumar, Fu, Pathak, Malik, 2021).',
      },
      {
        id: 'student',
        title: 'The Adaptive AI (the “student”)',
        what: 'A second network reads the last second of motion (20 steps of speed, slip, steering and pedal inputs) and estimates the teacher’s 8-number summary plus the grip under each axle. The teacher’s driving network then acts on that estimate.',
        why: 'A real car isn’t told the road is wet. It has to feel it from how the car responds.',
        measured:
          'Grip estimate error; whether it notices a single puddle (in controlled tests it doesn’t, compared against the same runs without the puddle); and completion vs the Basic AI: better on puddles, wet roads, worn tires and a very heavy car, worse on dry roads, in wind and with noisy sensors. Why it does better in the wet isn’t isolated yet (Results).',
        borrowed:
          'Phase 2 of Rapid Motor Adaptation (Kumar et al., 2021), trained DAgger-style (Ross et al., 2011).',
      },
      {
        id: 'confidence',
        title: 'Knowing when it’s unsure',
        what: 'Five copies of the Adaptive AI’s road-feeling network, trained with different seeds. When their estimates disagree, the car is probably somewhere it doesn’t understand. That disagreement is turned into a crash probability with isotonic regression; a safety governor eases off above a threshold.',
        why: 'A driver that can say “I’m not sure” is more useful than one that is confidently wrong.',
        measured:
          'Calibration (reliability diagram, ECE, Brier, AUROC) and the governor’s crash-vs-speed trade-off (Results). The estimate is calibrated but only weakly predictive; the governor helped clearly in one condition (laggy steering) and made little difference elsewhere.',
        borrowed:
          'Deep-ensemble uncertainty (Lakshminarayanan et al., 2017); isotonic calibration (Zadrozny & Elkan, 2002).',
      },
      {
        id: 'browser',
        title: 'Running in your browser',
        what: 'An exact TypeScript port of the simulator and a dependency-free neural-network runtime, inside a Web Worker. The page draws at your screen’s refresh rate and blends between 20 Hz physics steps.',
        why: 'So you can poke at it yourself: turn up the rain, change the conditions, and watch the real trained networks cope.',
        measured:
          'Parity tests on every network and on full closed-loop laps; 60 fps with 48 cars; Lighthouse 100 for performance and accessibility.',
        borrowed:
          'Comlink for worker messaging; everything else is plain TypeScript and Canvas 2D.',
      },
    ],
    card: {
      title: 'Model card',
      rows: [
        [
          'Inputs',
          'Driving: 33 numbers in the car’s frame (speed, slip, steering, last action, distance from the center line, heading error, 12 points of the road ahead, track width). Adaptive AI: plus the last 1 s of motion (200 numbers). Teacher AI: plus 14 true conditions.',
        ],
        [
          'Outputs',
          'Steering and throttle/brake in [−1, 1], 20 times a second. The Adaptive AI also outputs grip estimates; the ensemble outputs a crash-risk estimate.',
        ],
        [
          'Training data',
          'Simulated only: 2,000 procedurally generated tracks with randomized weather, grip, car and sensor conditions. No real-world data.',
        ],
        [
          'Evaluation',
          '200 held-out tracks never used for training or tuning, 12 condition suites (4 never trained on), paired across drivers, 95% bootstrap intervals.',
        ],
        [
          'Compute',
          'One NVIDIA T4 GPU on Kaggle: Basic AI population 8 × 150M steps, Teacher AI population 16 × 100M, the Adaptive AI’s five road-feeling networks 5 × 40M (the training budgets in the design). Total steps and wall time are on the Results page.',
        ],
        [
          'Intended use',
          'A demonstration and teaching tool for adaptive control and reinforcement learning.',
        ],
        ['Not for', 'Anything involving real vehicles.'],
        [
          'Known limits',
          'Flat 2D world, one car type, four-wheel effects collapsed into two axles, one training run per driver (no repeat seeds); the five copies share one driving policy, so their disagreement reflects uncertainty about the road, not about the driving.',
        ],
      ] as [string, string][],
      resultsLead: 'Every number on this site comes from the evaluation script.',
      resultsLink: 'See all results.',
    },
    repro: {
      title: 'Reproduce it',
      body: 'Python 3.12 with JAX for simulation and training; Node 22+ for the site. Full instructions are in docs/RUNBOOK.md.',
      commands: `# simulator, training, evaluation (sim/)
pip install -e "sim[dev]"
python -m trustlap.track --out runs/banks
bash scripts/full_pipeline.sh runs/full   # GPU: banks, blind, teacher PBT, students, evaluation
# (sim/notebooks/train_full.ipynb runs the same thing on Kaggle or Colab)

# website (web/)
npm ci && npm run dev`,
      repo: 'Source code and full history on GitHub',
    },
  },
  lab: {
    title: 'Training lab',
    intro:
      'How the Teacher AI learned: a population trained side by side on one GPU, copying each other’s best ideas while the simulator made the world harder.',
    source:
      'From population_curves.json, lineage.json, curriculum.json and checkpoints/ · recorded during training',
    watch: {
      title: 'Watch it learn',
      body: 'These 16 cars run the actual network saved at that point in training, live in your browser, in that moment’s conditions. Drag the slider or press play.',
      stepsLabel: 'training steps per driver',
      caption: (pct: number, d: number) =>
        `At this point it finished ${pct}% of its training laps, with the difficulty dial at ${Math.round(d * 100)}%.`,
      play: 'Play the timeline',
      pause: 'Pause the timeline',
      scrubber: 'Training checkpoint',
      canvasLabel: (steps: string) =>
        `Sixteen cars driving with the checkpoint saved after ${steps} training steps.`,
    },
    population: {
      title: (n: number) =>
        `${n} drivers trained side by side; the amber line is the family that won`,
      subtitle:
        'Fitness (meters of progress per second) of each population slot over training. Hover a line to pick it out.',
      summary: 'Fitness curves for every population member, with the winning lineage highlighted.',
      winner: 'Winning lineage',
      x: 'Training steps per driver',
      y: 'Progress (m/s)',
    },
    tree: {
      title: (n: number) => `The family tree: ${n} times a weaker driver copied a stronger one`,
      subtitle:
        'One lane per population slot. A curve is a copy: the weaker driver takes the stronger one’s network and nudges its learning settings. Amber is the winner’s ancestry.',
      summary:
        'Family tree of population-based training: lanes per slot, connectors for copy events.',
      slot: (n: number) => `Slot ${n}`,
      member: (id: number) => `Driver #${id}`,
      event: (
        p: number,
        c: number,
        steps: string,
        lr0: number,
        lr1: number,
        e0: number,
        e1: number,
      ) =>
        `#${p} copied into #${c} at ${steps} steps. Learning rate ${lr0.toExponential(1)} → ${lr1.toExponential(1)}, entropy ${e0.toExponential(1)} → ${e1.toExponential(1)}`,
    },
    curriculum: {
      title: (maxD: number) =>
        `The world got harder on its own, up to ${Math.round(maxD * 100)}% of full difficulty`,
      subtitle:
        'The difficulty dial over training: it rises when the drivers finish over 80% of laps twice in a row, and backs off below 50%.',
      summary: 'Step chart of curriculum difficulty against training steps.',
      y: 'Difficulty',
    },
    tuning: {
      title: 'How the winners tuned themselves',
      subtitle:
        'Learning rate and exploration (entropy) of the winning lineage over training. Two small charts, not one with two axes.',
      lr: 'Base learning rate',
      ent: 'Entropy coefficient',
    },
  },
  results: {
    title: 'Results',
    simple: {
      question: 'Does feeling the road help?',
      intro: (tracks: number) =>
        `We raced the Adaptive AI against the Basic AI on ${tracks} tracks neither had ever seen, in every condition. Same tracks, same weather, same starting grid: the only difference is that one can feel the grip.`,
      unit: 'Laps finished, out of 100',
      barLabel: (driver: string, laps: number) => `${driver}: ${laps} of 100 laps`,
      verdict: {
        better: 'Adaptive AI clearly better',
        worse: 'Adaptive AI clearly worse',
        same: 'No clear difference',
      },
      cards: {
        combinedLabel: 'In the hardest mix',
        fewer: (p: number) => `${p}% fewer crashes`,
        more: (p: number) => `${p}% more crashes`,
        combinedNote: (lo: number, hi: number) =>
          `than the Basic AI with rain, wind, worn tires and lag all at once. 95% range: ${lo}% to ${hi}%.`,
        gainLabel: (cond: string) => `Biggest gain: ${cond}`,
        lossLabel: (cond: string) => `Biggest cost: ${cond}`,
        lapsNote: (a: number, b: number) => `${a} vs ${b} laps finished out of 100.`,
        perHundred: 'laps per 100',
      },
      chartTitle: 'Laps finished in each condition',
      chartNote: (laps: number) =>
        `Each condition was raced ${laps} times per driver. “Clearly” means the 95% range of the difference doesn’t include zero.`,
      takeawaysTitle: 'What this means',
      better: (list: string) => `Clearly better in: ${list}.`,
      worse: (list: string) => `Clearly worse in: ${list}.`,
      same: (list: string) => `No clear difference in: ${list}.`,
      untrained: (list: string, best: number) =>
        `In conditions it never practised (${list}), no AI driver finishes more than ${best} laps out of 100: learning only covers what it practised.`,
      puddle: (of: number) =>
        `It does not notice a single puddle while driving through it (0 of ${of} test runs). Its better results in the wet come from somewhere else, not yet pinned down.`,
      risk: 'Its crash-risk estimate is honest on average (when it says 20%, about 20% of those moments end in a crash) but weak at telling risky moments from safe ones.',
      fullTitle: 'Full evaluation',
      fullSummary:
        'Every chart and number, with 95% confidence intervals: all four drivers, lap times, untrained conditions, puddle response, risk calibration, the safety governor, the GPU benchmark.',
    },
    intro: (tracks: number, seeds: number) =>
      `Every driver raced the same ${tracks} unseen test tracks under each condition, ${seeds} condition draws per track: ${tracks * seeds} laps per driver per condition. Ranges are 95% confidence intervals from a paired bootstrap over tracks.`,
    introMissing: 'The evaluation hasn’t been run for this build yet.',
    figuresHeading: 'Figures',
    notComputed: 'Not computed yet.',
    showTable: 'Show table',
    showChart: 'Show chart',
    source: (date: string, sha: string) => `From metrics.json · generated ${date} · commit ${sha}`,
    benchSource: (date: string, sha: string) =>
      `From bench.json · measured ${date} · commit ${sha}`,
    tiles: {
      crashReduction: 'Fewer crashes than the Basic AI',
      crashIncrease: 'More crashes than the Basic AI',
      crashReductionNote: 'Adaptive vs Basic AI, everything-at-once conditions',
      teacherGap: 'Gap to the Teacher AI',
      teacherGapUndefined:
        'Not meaningful here: the Teacher AI didn’t beat the Basic AI in these conditions, so there was no gap to close.',
      teacherGapNote:
        'Share of the Teacher AI’s advantage over the Basic AI the Adaptive AI did not recover (0% = matches the Teacher AI)',
      gripMae: 'Grip estimate error',
      gripMaeNote:
        'Mean absolute error of the Adaptive AI’s grip estimate, training-range conditions',
      latency: 'Time to notice a puddle',
      latencyNote: (lo: number, hi: number) => `Median; middle half ${lo}–${hi} ms`,
      notNoticed: 'Not noticed',
      notNoticedNote: (n: number, of: number) =>
        `Noticed a single puddle on a straight in ${n} of ${of} trials.`,
      noticedSomeNote: (n: number, of: number, lo: number, hi: number) =>
        `Noticed in ${n} of ${of} trials; median of those, middle half ${lo}–${hi} ms`,
      cars: 'Cars simulated at once',
      carsNote: (speedup: string) => `On one GPU, ${speedup}× faster than the same code on a CPU`,
      compute: 'Training compute',
      computeNote: (minutes: string, hw: string) => `${minutes} minutes · ${hw}`,
    },
    figures: {
      completion: {
        neutral: 'Lap completion by condition',
        title: (wins: number, total: number) =>
          wins === total
            ? `The Adaptive AI finishes at least as many laps as the Basic AI in every condition`
            : `The Adaptive AI finishes more laps than the Basic AI in ${wins} of ${total} conditions`,
        subtitle:
          'Lap completion rate by condition (unseen test tracks), sorted by the Adaptive AI’s advantage over the Basic AI.',
        axis: 'Laps completed',
      },
      delta: {
        neutral: 'Adaptive vs Basic AI, paired difference',
        title: (better: number, total: number) =>
          `Paired difference: the Adaptive AI’s completion rate is clearly above the Basic AI’s in ${better} of ${total} conditions`,
        subtitle:
          'Adaptive minus Basic AI, same tracks and conditions. Whiskers that cross zero mean no clear difference.',
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
        title: (ms: number, n?: number, of?: number) =>
          n != null && of != null && n < of
            ? `The Adaptive AI notices a single puddle in ${n} of ${of} trials, after about ${ms} ms when it does`
            : `The Adaptive AI notices a puddle under the rear wheels in about ${ms} ms`,
        titleNotNoticed: (n: number, of: number) =>
          `The Adaptive AI doesn’t notice a single puddle on a straight (${n} of ${of} trials noticed): it passes in under half a second and barely changes the motion the Adaptive AI reads`,
        titleMissing: 'Feeling a puddle',
        subtitle:
          'True rear grip (blue) vs the Adaptive AI’s estimate (amber) as the car crosses a single puddle on a straight. Time 0 is when the true grip first drops by more than 0.1.',
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
        titleNoEffect:
          'In everything-at-once conditions, no governor threshold clearly reduces crashes',
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
      'The Adaptive AI’s risk estimate comes from five small networks that share one driving policy, so their disagreement measures uncertainty about the road, not about the driving itself.',
      'Each driver comes from one training run on one GPU (compute above). Other seeds or longer runs could change these numbers.',
      'Results that don’t favor the Adaptive AI are shown as they came out.',
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
export const SITE_URL = 'https://trustlap.vercel.app'
