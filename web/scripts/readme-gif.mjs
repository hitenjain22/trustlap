// README GIF (SPEC 16.1): the live hero as the rain rises from dry to 60% and the cars adapt.
// Captures the canvas pixels frame by frame, then assembles a GIF with Pillow (via uvx, so the
// project gains no dependency). Run against `npm run preview`:
//   node scripts/readme-gif.mjs http://localhost:4173 ../docs/media/hero.gif
import { execFileSync } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium } from '@playwright/test'

const [base = 'http://localhost:4173', out = '../docs/media/hero.gif'] = process.argv.slice(2)
const FRAMES = 90
const RAIN_MAX = 0.6
const dir = mkdtempSync(join(tmpdir(), 'trustlap-gif-'))

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 760 }, deviceScaleFactor: 1 })
await page.goto(base + '/')
const rain = page.getByRole('slider', { name: 'Rain' })
await rain.fill('0')
await page.waitForTimeout(12000) // let the field leave the grid and spread out
const canvas = page.locator('canvas').first()
const times = []
const t0 = Date.now()
for (let i = 0; i < FRAMES; i++) {
  // Hold dry for the first 20 frames, then raise the rain steadily.
  const r = Math.min(RAIN_MAX, Math.max(0, ((i - 20) / (FRAMES - 35)) * RAIN_MAX))
  await rain.fill(String(Math.round(r * 100) / 100)) // "0.00" is rejected by fill(); "0" works
  const url = await canvas.evaluate((c) => c.toDataURL('image/png'))
  writeFileSync(
    join(dir, `f${String(i).padStart(3, '0')}.png`),
    Buffer.from(url.split(',')[1], 'base64'),
  )
  times.push(Date.now() - t0)
  await page.waitForTimeout(60)
}
await browser.close()
// Real elapsed time between captures becomes each frame's duration, so playback is true speed.
const durations = times.map((t, i) => (i === 0 ? times[1] - times[0] : t - times[i - 1]))
writeFileSync(join(dir, 'durations.json'), JSON.stringify(durations))
const py = `
import json, sys, glob, os
from PIL import Image
d = sys.argv[1]
# Keep every other frame (durations merged, so playback stays real-time) to stay small.
files = sorted(glob.glob(os.path.join(d, 'f*.png')))[::2]
raw = json.load(open(os.path.join(d, 'durations.json')))
dur = [sum(raw[i:i + 2]) for i in range(0, len(raw), 2)]
frames = []
for f in files:
    im = Image.open(f).convert('RGB')
    # The hero keeps its left side clear for the headline text; keep the track side.
    im = im.crop((round(im.width * 0.44), 0, im.width, im.height))
    w = 640
    im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    frames.append(im.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
frames[0].save(sys.argv[2], save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
print('wrote', sys.argv[2], round(os.path.getsize(sys.argv[2]) / 1e6, 2), 'MB')
`
execFileSync('uvx', ['--with', 'pillow', 'python', '-c', py, dir, out], { stdio: 'inherit' })
