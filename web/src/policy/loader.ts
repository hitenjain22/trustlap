/** Load exported networks: validate the json, check the .bin's sha256, slice the layers. */
import type { DenseLayer } from './mlp.ts'
import { networkFileSchema, type NetworkFile } from './schema.ts'

export interface LoadedNetwork {
  meta: NetworkFile
  layers: Record<string, DenseLayer[]>
}

async function sha256Hex(buf: ArrayBuffer): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle
  if (!subtle) return null // non-secure contexts: skip (the host still serves over https)
  const digest = await subtle.digest('SHA-256', buf)
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Build layers from a parsed json + its binary (also used directly by tests). */
export async function parseNetwork(rawJson: unknown, bin: ArrayBuffer): Promise<LoadedNetwork> {
  const meta = networkFileSchema.parse(rawJson)
  if (bin.byteLength !== meta.bytes)
    throw new Error(`${meta.bin}: ${bin.byteLength} bytes, expected ${meta.bytes}`)
  const hash = await sha256Hex(bin)
  if (hash !== null && hash !== meta.sha256) throw new Error(`${meta.bin}: checksum mismatch`)
  const layers: Record<string, DenseLayer[]> = {}
  for (const [net, table] of Object.entries(meta.networks)) {
    layers[net] = table.map((l) => {
      const kernel = new Float32Array(bin, l.byte_offset, l.in * l.out)
      const bias = new Float32Array(bin, l.byte_offset + 4 * l.in * l.out, l.out)
      return { name: l.name, in: l.in, out: l.out, activation: l.activation, kernel, bias }
    })
  }
  return { meta, layers }
}

/** Fetch `<base>/policies/<name>.json` and its .bin. */
export async function loadNetwork(base: string, name: string): Promise<LoadedNetwork> {
  const res = await fetch(`${base}/policies/${name}.json`)
  if (!res.ok) throw new Error(`${name}.json: HTTP ${res.status}`)
  const json = await res.json()
  const bin = await fetch(`${base}/policies/${networkFileSchema.shape.bin.parse(json.bin)}`)
  if (!bin.ok) throw new Error(`${json.bin}: HTTP ${bin.status}`)
  return parseNetwork(json, await bin.arrayBuffer())
}
