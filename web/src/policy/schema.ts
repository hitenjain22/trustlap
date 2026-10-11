/** zod schema for exported network files (`policies/<name>.json`, SPEC 9.2). */
import { z } from 'zod'

const layer = z.object({
  name: z.string(),
  in: z.number().int().positive(),
  out: z.number().int().positive(),
  activation: z.enum(['elu', 'tanh', 'linear', 'sigmoid']),
  byte_offset: z.number().int().nonnegative(),
})
const norm = z.object({ mean: z.array(z.number()), std: z.array(z.number()) })

export const networkFileSchema = z.object({
  schema_version: z.literal(1),
  name: z.string(),
  bin: z.string(),
  bytes: z.number().int().positive(),
  sha256: z.string().length(64),
  kind: z.enum(['blind', 'teacher', 'student']),
  networks: z.record(z.string(), z.array(layer)),
  obs_clip: z.number(),
  obs_norm: norm.optional(),
  privileged_norm: norm.optional(),
  log_std: z.array(z.number()).optional(),
  training: z.record(z.string(), z.unknown()).optional(),
})
export type NetworkFile = z.infer<typeof networkFileSchema>
