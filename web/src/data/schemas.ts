/**
 * zod mirrors of the result artifacts (SPEC 9.3). Every metric is optional: the UI shows
 * "Not computed yet" for anything missing, never a made-up number.
 */
import { z } from 'zod'

const num = z.number()
const nullableNum = z.number().nullable()
export const estimateSchema = z.object({ value: nullableNum, ci: z.array(nullableNum).length(2) })
export type Estimate = z.infer<typeof estimateSchema>

const driverSuiteSchema = z.object({
  completion: estimateSchema.optional(),
  crash: estimateSchema.optional(),
  paired_lap_time_s: estimateSchema.nullable().optional(),
  mean_speed_mps: estimateSchema.optional(),
  grip_mae: estimateSchema.optional(),
  n_laps: z.number().int().optional(),
  n_paired_laps: z.number().int().optional(),
})
export type DriverSuite = z.infer<typeof driverSuiteSchema>

export const metricsSchema = z.object({
  schema_version: z.literal(1),
  generated_utc: z.string(),
  git_sha: z.string().optional(),
  protocol: z
    .object({
      test_tracks: z.number().int(),
      condition_seeds: z.number().int(),
      laps_per_driver_per_suite: z.number().int(),
      bootstrap_resamples: z.number().int(),
      governor_tau: num.nullable(),
      calibration_fit: z.string(),
      eval_wall_minutes: num.optional(),
    })
    .partial()
    .optional(),
  headline: z
    .object({
      max_parallel_cars: num,
      gpu_speedup_vs_cpu: num,
      train_env_steps_total: num,
      train_wall_minutes: num,
      train_hardware: z.string(),
      student_crash_reduction_vs_blind_combined: estimateSchema,
      teacher_gap_combined: estimateSchema,
      grip_mae: nullableNum,
      puddle_latency_ms: z.object({
        median: nullableNum,
        iqr: z.array(num).length(2).nullable(),
        trials: z.number().int().optional(),
        noticed: z.number().int().optional(),
        control_noticed: z.number().int().optional(),
        never_settled: z.number().int().optional(),
        definition: z.string().optional(),
      }),
    })
    .partial()
    .optional(),
  suites: z.record(z.string(), z.record(z.string(), driverSuiteSchema)).optional(),
  paired_deltas: z
    .array(
      z.object({
        suite: z.string(),
        a: z.string(),
        b: z.string(),
        metric: z.string(),
        value: nullableNum,
        ci: z.array(nullableNum),
      }),
    )
    .optional(),
  calibration: z
    .object({
      bins: z.array(
        z.object({ u_mean: num.optional(), p_pred: num, p_obs: num, n: z.number().int() }),
      ),
      ece: num,
      brier: num,
      auroc: nullableNum,
      tau_default: num.nullable().optional(),
      /** The earlier disagreement-only risk model, scored on the same test steps. */
      baseline: z.object({ name: z.string(), ece: num, brier: num, auroc: nullableNum }),
    })
    .partial()
    .optional(),
  /** Independent training seeds (sim/trustlap/replicate.py); charts use the primary seed. */
  replication: z
    .object({
      seeds: z.array(z.number().int()),
      primary_seed: z.number().int(),
      agreement: z.record(
        z.string(),
        z.object({
          primary: z.enum(['better', 'worse', 'same']).nullable(),
          same_in: z.number().int(),
          of: z.number().int(),
        }),
      ),
    })
    .optional(),
  governor_sweep: z
    .array(
      z.object({ tau: num.nullable(), crash: estimateSchema, paired_lap_time_s: estimateSchema }),
    )
    .optional(),
  puddle_step: z
    .object({ t_ms: z.array(num), mu_true: z.array(num), mu_hat: z.array(num) })
    .optional(),
  generalization: z
    .object({
      train_bank: z.record(z.string(), z.object({ completion: estimateSchema })),
      test_bank: z.record(z.string(), z.object({ completion: estimateSchema })),
    })
    .optional(),
})
export type Metrics = z.infer<typeof metricsSchema>

export const curriculumSchema = z.object({
  run: z.string(),
  points: z.array(
    z.object({ update: z.number(), env_steps: num, d: num, success: nullableNum.optional() }),
  ),
})
export type Curriculum = z.infer<typeof curriculumSchema>

export const populationSchema = z.object({
  members: z.number().int(),
  points: z.array(
    z.object({
      update: z.number(),
      env_steps: num,
      fitness: z.array(num),
      ids: z.array(z.number().int()),
      lr: z.array(num),
      ent: z.array(num),
      completion: z.array(num),
    }),
  ),
})
export type Population = z.infer<typeof populationSchema>

const hp = z.object({ lr: num, ent: num })
export const lineageSchema = z.object({
  ids: z.array(z.number().int()),
  next_id: z.number().int(),
  winner_id: z.number().int().nullable().optional(),
  events: z.array(
    z.object({
      update: z.number(),
      env_steps: num,
      slot: z.number().int(),
      child_id: z.number().int(),
      parent_id: z.number().int(),
      prev_id: z.number().int(),
      hp_before: hp,
      hp_after: hp,
      fitness_parent: num,
      fitness_child_before: num,
    }),
  ),
  snapshots: z.array(
    z.object({
      update: z.number(),
      env_steps: num,
      members: z.array(z.object({ id: z.number().int(), fitness: num, lr: num, ent: num })),
    }),
  ),
})
export type Lineage = z.infer<typeof lineageSchema>

export const checkpointsSchema = z.object({
  checkpoints: z.array(
    z.object({
      i: z.number().int(),
      name: z.string(),
      update: z.number(),
      env_steps: num,
      difficulty: num,
      mean_return: num,
      completion: num,
    }),
  ),
})
export type Checkpoints = z.infer<typeof checkpointsSchema>
