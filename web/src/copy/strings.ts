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
        body: 'A population of drivers trains side by side. Every few minutes the weakest copy the strongest and tweak how they learn.',
      },
      {
        title: 'A world that gets harder as they improve',
        body: 'The simulator raises the difficulty on its own as the drivers improve: more rain, worn tires, laggy steering, gusty wind.',
      },
      {
        title: 'A driver that feels the road',
        body: 'A second network reads the last second of motion and estimates the grip under each axle. No one tells it the road is wet.',
        measured: (ms: number) =>
          `On a straight it notices a puddle under the rear wheels in about ${ms} ms.`,
      },
      {
        title: 'A driver that knows when it’s unsure',
        body: 'Five copies of the sensing network vote. When they disagree, the car knows it is somewhere unfamiliar.',
        measured: (ece: string) =>
          `Its crash-risk estimate is calibrated (error ${ece}), though only weakly predictive at this training budget.`,
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
    groups: { weather: 'Weather', car: 'Car', electronics: 'Electronics', safety: 'Safety' },
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
      ['1–4', 'Student, blind, teacher, scripted'],
      ['?', 'Show these shortcuts'],
    ] as [string, string][],
    presets: 'Presets',
    presetNames: {
      dry: 'Dry practice',
      monsoon: 'Monsoon',
      ice: 'Black ice',
      broken: 'Broken car',
      chaos: 'Everything at once',
    } as Record<string, string>,
    governorNote:
      'Student only: eases off the throttle when its crash-risk estimate passes the threshold.',
    governorThreshold: 'Risk threshold',
    canvasLabel: 'A self-taught car driving on the selected track, with a scripted ghost car.',
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
    run: 'Run the puddle test',
    track: 'Puddle test straight',
    title: 'Puddle test',
    body: 'The student drives down a long straight at half speed and crosses one puddle (grip × 0.4). Does its grip estimate drop while it’s on the puddle? Scored with the same rule as the evaluation.',
    chartLabel: 'Rear grip since the test started: true grip and the student’s estimate.',
    truth: 'True rear grip',
    estimate: 'Student’s estimate',
    waiting: 'Driving toward the puddle…',
    noticed: (ms: number) => `Noticed the slippery patch in ${ms} ms.`,
    missed:
      'Didn’t notice the slippery patch: its estimate didn’t drop by half the true drop while it was on the puddle. That matches the evaluation (Results).',
    noEstimate: 'This driver doesn’t estimate grip. The puddle test needs the student.',
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
    gripShort: (est: string, truth: string) => `est ${est} / true ${truth}`,
    lap: 'Lap',
    best: 'Best',
    delta: 'Δ vs ghost',
    crashed: 'Crashed',
    front: 'Front grip',
    rear: 'Rear grip',
    gripReading: (est: string, truth: string) => `est ${est} / true ${truth}`,
    gripTrueOnly: (truth: string) => `true ${truth} · this driver doesn’t estimate grip`,
    gripAria: (axle: string, est: string, truth: string) =>
      `${axle}: estimated ${est}, true ${truth}`,
    risk: 'Crash risk',
    slowsAbove: (pct: number) => `(slows above ${pct}%)`,
    governorOff: '(governor off)',
    riskStudentOnly: 'Only the student ensemble estimates its crash risk.',
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
    intro:
      'Six pieces, built in order. Each one combines a published technique with a custom simulator; nothing here is a new algorithm. What’s new is putting them together and measuring them honestly.',
    pipeline: [
      'Simulator',
      'Curriculum',
      'Population (PPO)',
      'Teacher',
      'Student ensemble',
      'Browser',
    ],
    pipelineLabel:
      'Pipeline: simulator, then curriculum, then a population trained with PPO, then the teacher, then the student ensemble, then the browser.',
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
        title: 'The teacher',
        what: 'A driver that is told the true conditions (grip under each axle, puddles ahead, mass, wind…) through a small encoder network that compresses them into 8 numbers.',
        why: 'It shows how well a driver can do when it knows everything, and its 8-number summary is what the student learns to estimate.',
        measured:
          'Completion in every condition (Results). It beats the blind driver in the wet, on puddles, worn tires and heavier cars, but not in wind or with noisy sensors; each teacher trained for two-thirds of the blind driver’s steps.',
        borrowed: 'Phase 1 of Rapid Motor Adaptation (Kumar, Fu, Pathak, Malik, 2021).',
      },
      {
        id: 'student',
        title: 'The student: feeling the road',
        what: 'A second network reads the last second of motion (20 steps of speed, slip, steering and pedal inputs) and estimates the teacher’s 8-number summary plus the grip under each axle. The teacher’s driving network then acts on that estimate.',
        why: 'A real car isn’t told the road is wet. It has to feel it from how the car responds.',
        measured:
          'Grip estimate error; whether it notices a single puddle (in controlled tests it doesn’t, compared against the same runs without the puddle); and completion vs the blind driver: better on puddles, wet roads, worn tires and a very heavy car, worse on dry roads, in wind and with noisy sensors. Why it does better in the wet isn’t isolated yet (Results).',
        borrowed:
          'Phase 2 of Rapid Motor Adaptation (Kumar et al., 2021), trained DAgger-style (Ross et al., 2011).',
      },
      {
        id: 'confidence',
        title: 'Knowing when it’s unsure',
        what: 'Five students trained with different seeds. When their estimates disagree, the car is probably somewhere it doesn’t understand. That disagreement is turned into a crash probability with isotonic regression; a safety governor eases off above a threshold.',
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
          'Driving: 33 numbers in the car’s frame (speed, slip, steering, last action, distance from the center line, heading error, 12 points of the road ahead, track width). Student: plus the last 1 s of motion (200 numbers). Teacher: plus 14 true conditions.',
        ],
        [
          'Outputs',
          'Steering and throttle/brake in [−1, 1], 20 times a second. The student also outputs grip estimates; the ensemble outputs a crash-risk estimate.',
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
          'One NVIDIA T4 GPU on Kaggle: blind population 8 × 150M steps, teacher population 16 × 100M, students 5 × 40M (the training budgets in the design). Total steps and wall time are on the Results page.',
        ],
        [
          'Intended use',
          'A demonstration and teaching tool for adaptive control and reinforcement learning.',
        ],
        ['Not for', 'Anything involving real vehicles.'],
        [
          'Known limits',
          'Flat 2D world, one car type, four-wheel effects collapsed into two axles, one training run per driver (no repeat seeds); the students share one teacher policy, so their disagreement reflects uncertainty about the road, not about the driving.',
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
      'How the teacher driver learned: a population trained side by side on one GPU, copying each other’s best ideas while the simulator made the world harder.',
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
      crashIncrease: 'More crashes than the blind driver',
      crashReductionNote: 'Student vs blind, everything-at-once conditions',
      teacherGap: 'Teacher gap left',
      teacherGapUndefined:
        'Not meaningful here: the teacher didn’t beat the blind driver in these conditions, so there was no gap to close.',
      teacherGapNote:
        'Share of the teacher’s advantage over the blind driver the student did not recover (0% = matches the teacher)',
      gripMae: 'Grip estimate error',
      gripMaeNote: 'Mean absolute error of the student’s grip estimate, training-range conditions',
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
        title: (ms: number, n?: number, of?: number) =>
          n != null && of != null && n < of
            ? `The student notices a single puddle in ${n} of ${of} trials, after about ${ms} ms when it does`
            : `The student notices a puddle under the rear wheels in about ${ms} ms`,
        titleNotNoticed: (n: number, of: number) =>
          `The student doesn’t notice a single puddle on a straight (${n} of ${of} trials noticed): it passes in under half a second and barely changes the motion the student reads`,
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
      'All five students share one teacher’s driving policy, so their disagreement measures uncertainty about the road, not about the driving itself.',
      'Each driver comes from one training run on one GPU (compute above). Other seeds or longer runs could change these numbers.',
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
