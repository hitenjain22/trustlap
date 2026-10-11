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

test('home: the adaptive cars slow down when it starts raining (within their training range)', async ({
  page,
}) => {
  test.setTimeout(45_000)
  await page.goto('/')
  const slider = page.getByRole('slider', { name: 'Rain' })
  await slider.fill('0')
  // Cars launch from a standing grid; their mean speed settles after ~11 s on a dry track.
  await page.waitForTimeout(12_000)
  const dry = await meanSpeed(page)
  // 45% rain is inside the range the student trained on. It has to *feel* the lower grip
  // through its motion history, so it slows over a few seconds rather than instantly.
  // Measured locally: −2% after 2.5 s, −10% after 5 s, no crashes.
  await slider.fill('0.45')
  await page.waitForTimeout(5000)
  const wet = await meanSpeed(page)
  expect(dry).toBeGreaterThan(15)
  expect(wet).toBeLessThan(dry * 0.95)
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
