// Demo video (SPEC 16.2): a scripted walk through the live site, recorded with Playwright.
// Captions describe what is on screen, never an outcome the run might not show. Run against
// `npm run preview`; the .webm is NOT committed (upload it and link it from the README):
//   node scripts/demo-video.mjs http://localhost:4173 ~/Downloads/trustlap-demo.webm
import { renameSync } from 'node:fs'
import { chromium } from '@playwright/test'

const [base = 'http://localhost:4173', out = 'trustlap-demo.webm'] = process.argv.slice(2)
const size = { width: 1280, height: 720 }
const browser = await chromium.launch()
const context = await browser.newContext({
  viewport: size,
  deviceScaleFactor: 1,
  recordVideo: { dir: 'demo-video-tmp', size },
})
const page = await context.newPage()
const wait = (ms) => page.waitForTimeout(ms)

/** A caption bar over the page (added by this script, not part of the site). */
async function caption(text) {
  await page.evaluate((t) => {
    let el = document.getElementById('demo-caption')
    if (!el) {
      el = document.createElement('div')
      el.id = 'demo-caption'
      el.style.cssText =
        'position:fixed;left:50%;top:76px;transform:translateX(-50%);z-index:9999;max-width:900px;' +
        'padding:12px 20px;border-radius:10px;background:rgba(20,24,29,.92);border:1px solid #3a424d;' +
        'color:#e9edf0;font:600 20px/1.35 Archivo,system-ui,sans-serif;text-align:center;pointer-events:none'
      document.body.appendChild(el)
    }
    el.textContent = t
  }, text)
}

// 1. Hero, rain rising (~16 s).
await page.goto(base + '/')
const rain = page.getByRole('slider', { name: 'Rain' })
await rain.fill('0')
await wait(5000)
await caption(
  'Every car is a neural network running in your browser. It learned to drive by itself.',
)
await wait(3000)
await caption(
  'Turning up the rain: the cars have to feel the lower grip through how the car moves.',
)
for (let i = 1; i <= 30; i++) {
  await rain.fill(String(Math.round(i * 2) / 100)) // 0.02 … 0.6
  await wait(250)
}
await wait(2000)

// 2. Training lab: the same network at different points in training (~14 s).
await page.goto(base + '/lab')
await caption('Training lab: 16 cars running the network saved at different points in training.')
await page.getByRole('slider', { name: 'Training checkpoint' }).fill('0')
await wait(3000)
await page.getByRole('button', { name: 'Play the timeline' }).click()
await wait(11000)

// 3. Puddle test (~22 s).
await page.goto(base + '/playground')
await caption('Puddle test: one puddle on a long straight. Does the grip estimate drop?')
await page.getByRole('button', { name: 'Run the puddle test' }).click({ timeout: 15000 })
await page
  .getByRole('status')
  .filter({ hasText: /Noticed the slippery patch|Didn’t notice/ })
  .waitFor({ timeout: 40000 })
await page
  .getByRole('status')
  .filter({ hasText: /Noticed the slippery patch|Didn’t notice/ })
  .evaluate((el) => el.scrollIntoView({ block: 'end' }))
await caption('Scored with the same rule as the evaluation, whatever the result.')
await wait(5000)

// 4. Black ice, outside training, with the safety governor on (~16 s).
await page.getByRole('button', { name: 'End the puddle test' }).click()
await page.evaluate(() => window.scrollTo(0, 0))
await page.getByText('Safety', { exact: true }).click()
await page.getByRole('switch', { name: 'Safety governor' }).check()
await page.getByRole('checkbox', { name: 'Basic AI' }).check() // the Basic AI as a ghost
await page.getByRole('button', { name: 'Chase' }).click()
await page.getByRole('button', { name: 'Black ice' }).click()
await page.evaluate(() => window.scrollTo(0, 0))
await caption('Black ice, outside what it trained on, with the Basic AI as a ghost.')
await wait(14000)

// 5. Results (~10 s).
await page.goto(base + '/results')
await caption('Results: 200 unseen tracks, 12 conditions, losses included.')
await wait(3000)
await page.mouse.wheel(0, 650)
await wait(6000)

// 6. End card (~4 s).
await page.setContent(`<body style="margin:0;height:100vh;display:grid;place-items:center;background:#1c2127;
  color:#e9edf0;font-family:system-ui,sans-serif;text-align:center"><div>
  <div style="font-size:44px;font-weight:800;color:#f2b544">TrustLap</div>
  <div style="margin-top:12px;font-size:28px">trustlap.vercel.app</div></div></body>`)
await wait(4000)

const video = page.video()
await context.close()
renameSync(await video.path(), out)
await browser.close()
console.log(`wrote ${out}`)
