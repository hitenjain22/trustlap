import { expect, test, type Page } from '@playwright/test'

const meanSpeed = async (page: Page) =>
  Number(await page.locator('canvas').first().getAttribute('data-mean-speed'))

test('home: live hero canvas draws and cars drive', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('link', { name: 'TrustLap' })).toBeVisible()
  await expect(page.getByRole('heading', { level: 1 })).toContainText('rain')
  const canvas = page.locator('canvas').first()
  await expect(canvas).toHaveAttribute('data-mean-speed', /\d/, { timeout: 15_000 })
  // Non-blank: pixel variance over a downscaled copy of the canvas.
  const variance = await canvas.evaluate((c: HTMLCanvasElement) => {
    const probe = document.createElement('canvas')
    probe.width = 64
    probe.height = 64
    const ctx = probe.getContext('2d')!
    ctx.drawImage(c, 0, 0, 64, 64)
    const px = ctx.getImageData(0, 0, 64, 64).data
    let sum = 0
    let sq = 0
    for (let i = 0; i < px.length; i += 4) {
      const v = px[i]! + px[i + 1]! + px[i + 2]!
      sum += v
      sq += v * v
    }
    const n = px.length / 4
    return sq / n - (sum / n) ** 2
  })
  expect(variance).toBeGreaterThan(50)
})

test('home: turning the rain up slows the cars within a second', async ({ page }) => {
  test.setTimeout(40_000)
  await page.goto('/')
  const slider = page.getByRole('slider', { name: 'Rain' })
  await slider.fill('0')
  // Cars launch from a standing grid; their mean speed settles after ~11 s on a dry track.
  await page.waitForTimeout(12_000)
  const dry = await meanSpeed(page)
  await slider.fill('1')
  await page.waitForTimeout(1000)
  const wet = await meanSpeed(page)
  expect(dry).toBeGreaterThan(15)
  // Measured: −4.5% after 1 s (and still falling); steady-state noise is about ±0.5%.
  expect(wet).toBeLessThan(dry * 0.98)
})

test('playground: sliders show the training band and warn outside it', async ({ page }) => {
  await page.goto('/playground')
  const wind = page.getByRole('slider', { name: 'Wind', exact: true })
  await expect(wind).toBeVisible({ timeout: 10_000 })
  await expect(page.getByText('Outside what it trained on.')).toHaveCount(0)
  await wind.fill('15')
  await expect(page.getByText('Outside what it trained on.')).toHaveCount(1)
  await expect(page.locator('canvas').first()).toHaveAttribute('data-mean-speed', /\d/, {
    timeout: 15_000,
  })
})

test('unknown routes show the not-found page via the SPA fallback', async ({ page }) => {
  await page.goto('/no-such-page')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('drove off the track')
})
