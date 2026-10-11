// Size budgets, enforced in CI after `npm run build`.
//  - Initial JS (gzipped) ≤ 200 KB (SPEC 10.7): JS referenced by dist/index.html.
//  - Artifacts (SPEC 9.5): total ≤ 15 MB, any single file ≤ 5 MB,
//    bench/metrics/population_curves/lineage JSON ≤ 1 MB each.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { gzipSync } from 'node:zlib'

const KB = 1024
const MB = 1024 * KB
const DIST = 'dist'
const ARTIFACTS = 'public/artifacts'
const failures = []

function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    return entry.isDirectory() ? walk(path) : [path]
  })
}

// Initial JS: scripts and modulepreloads linked from index.html.
const html = readFileSync(join(DIST, 'index.html'), 'utf8')
const jsPaths = [...html.matchAll(/(?:src|href)="\/([^"]+\.js)"/g)].map((m) => m[1])
const initialJsGz = jsPaths.reduce(
  (sum, p) => sum + gzipSync(readFileSync(join(DIST, p))).length,
  0,
)
console.log(`initial JS (gzip): ${(initialJsGz / KB).toFixed(1)} KB / 200 KB`)
if (initialJsGz > 200 * KB) failures.push('initial JS exceeds 200 KB gzipped')

// Artifacts.
const files = walk(ARTIFACTS).filter((p) => !p.endsWith('.gitkeep'))
let total = 0
for (const path of files) {
  const bytes = statSync(path).size
  total += bytes
  const name = relative(ARTIFACTS, path)
  if (bytes > 5 * MB) failures.push(`${name} is ${(bytes / MB).toFixed(2)} MB (limit 5 MB)`)
  if (/(^|\/)(bench|metrics|population_curves|lineage)\.json$/.test(name) && bytes > MB) {
    failures.push(`${name} is ${(bytes / MB).toFixed(2)} MB (limit 1 MB)`)
  }
}
console.log(`artifacts: ${files.length} files, ${(total / MB).toFixed(2)} MB / 15 MB`)
if (total > 15 * MB) failures.push('artifacts exceed 15 MB in total')

if (failures.length > 0) {
  console.error(`Size budget failed:\n  - ${failures.join('\n  - ')}`)
  process.exit(1)
}
console.log('Size budgets OK')
