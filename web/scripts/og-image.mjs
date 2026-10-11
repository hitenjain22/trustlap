// Social preview image (SPEC 11.10): load the hero at rain 70%, let the cars run, screenshot the
// live canvas, then composite the wordmark + headline. Run against `npm run preview`:
//   node scripts/og-image.mjs http://localhost:4173 public/og.png
import { chromium } from '@playwright/test'

const [base = 'http://localhost:4173', out = 'public/og.png'] = process.argv.slice(2)
const browser = await chromium.launch()
const page = await browser.newPage({
  viewport: { width: 1600, height: 1000 },
  deviceScaleFactor: 1,
})
await page.goto(base + '/')
await page.getByRole('slider', { name: 'Rain' }).fill('0.7')
await page.waitForTimeout(9000)
// Read the canvas pixels themselves (an element screenshot would include the overlaid text).
const dataUrl = await page
  .locator('canvas')
  .first()
  .evaluate((c) => c.toDataURL('image/png'))
const frame = Buffer.from(dataUrl.split(',')[1], 'base64')
const card = await browser.newPage({ viewport: { width: 1200, height: 630 } })
await card.goto(base + '/fonts/OFL.txt') // same origin, so the self-hosted font loads
await card.setContent(`<!doctype html><html><head><style>
  @font-face { font-family: Archivo; src: url('${base}/fonts/archivo-latin-var.woff2') format('woff2'); font-weight: 100 900; font-stretch: 62% 125%; }
  html, body { margin: 0; width: 1200px; height: 630px; background: #1c2127; overflow: hidden; }
  .bg { position: absolute; inset: 0; background: url(data:image/png;base64,${frame.toString('base64')}) right center / cover no-repeat; }
  .panel { position: absolute; left: 48px; bottom: 48px; width: 560px; padding: 32px; border-radius: 12px; border: 1px solid #3a424d; background: rgba(28,33,39,0.88); font-family: Archivo, sans-serif; color: #e9edf0; }
  .mark { font-weight: 800; font-stretch: 125%; font-size: 30px; color: #f2b544; }
  h1 { margin: 14px 0 0; font-weight: 760; font-stretch: 125%; font-size: 52px; line-height: 1.02; letter-spacing: -0.02em; }
</style></head><body><div class="bg"></div><div class="panel"><div class="mark">TrustLap</div><h1>These cars taught themselves to drive in the rain.</h1></div></body></html>`)
await card.waitForTimeout(800)
await card.screenshot({ path: out })
await browser.close()
console.log(`wrote ${out}`)
