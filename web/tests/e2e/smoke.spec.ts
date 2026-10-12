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
  test.setTimeout(75_000)
  await page.goto('/')
  const slider = page.getByRole('slider', { name: 'Rain' })
  await slider.fill('0')
  // Cars launch from a standing grid; their mean speed settles after ~11 s of *simulated* time.
  // Wait on the sim's step counter (20 Hz), not the wall clock: a busy machine runs the sim
  // slower than real time, and measuring "dry" mid-launch would hide the slowdown.
  const simSeconds = async () =>
    Number(await page.locator('canvas').first().getAttribute('data-step')) / 20
  await expect.poll(simSeconds, { timeout: 40_000, intervals: [500] }).toBeGreaterThan(12)
  const dry = await meanSpeed(page)
  expect(dry).toBeGreaterThan(15)
  // 45% rain is inside the range the student trained on. It has to *feel* the lower grip
  // through its motion history, so it slows over a few seconds rather than instantly.
  // Measured locally (GPU-trained students): −1 to −3% after 1 s, −13 to −18% after 4 s of sim.
  await slider.fill('0.45')
  await expect
    .poll(() => meanSpeed(page), { timeout: 15_000, intervals: [500] })
    .toBeLessThan(dry * 0.95)
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

test('playground: draw a loop, validate it, race on it', async ({ page }) => {
  await page.goto('/playground')
  const draw = page.getByRole('button', { name: 'Draw a track' })
  await expect(draw).toBeEnabled({ timeout: 15_000 })
  await draw.click()
  const dialog = page.getByRole('dialog', { name: 'Draw a track' })
  await expect(dialog).toBeFocused()
  const box = (await dialog.boundingBox())!
  const cx = box.x + box.width / 2
  const cy = box.y + box.height / 2
  const loop = async (fx: (t: number) => number, fy: (t: number) => number) => {
    await page.mouse.move(cx + fx(0), cy + fy(0))
    await page.mouse.down()
    for (let i = 1; i <= 120; i++) {
      const t = (2 * Math.PI * i) / 120
      await page.mouse.move(cx + fx(t), cy + fy(t))
    }
    await page.mouse.up()
  }
  const rx = box.width * 0.3
  const ry = box.height * 0.3
  // A figure eight crosses itself.
  await loop(
    (t) => rx * Math.sin(t),
    (t) => ry * Math.sin(t) * Math.cos(t),
  )
  await expect(
    page.getByRole('status').filter({ hasText: /crosses itself|too close/ }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Use this track' })).toBeDisabled()
  await page.getByRole('button', { name: 'Clear' }).click()
  // An oval is valid.
  await loop(
    (t) => rx * Math.cos(t),
    (t) => ry * Math.sin(t),
  )
  await expect(page.getByRole('status').filter({ hasText: 'Looks good' })).toBeVisible()
  await page.getByRole('button', { name: 'Use this track' }).click()
  await expect(dialog).toHaveCount(0)
  const select = page.getByRole('combobox', { name: 'Track' })
  await expect(select).toHaveValue('')
  await expect(select.locator('option:checked')).toHaveText('Your drawn track')
  await expect(page.locator('canvas').first()).toHaveAttribute('data-mean-speed', /\d/)
  await expect(draw).toBeFocused()
  // Esc cancels the drawer.
  await draw.click()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
})

test('playground: the puddle test drives into a puddle and scores the result', async ({ page }) => {
  test.setTimeout(75_000)
  await page.goto('/playground')
  await page.getByRole('button', { name: /^Puddle test/ }).click({ timeout: 15_000 })
  await expect(page.getByLabel('Track').and(page.getByRole('combobox'))).toHaveValue('')
  // Either verdict is fine here; which one it is comes from the trained student.
  await expect(
    page.getByRole('status').filter({ hasText: /Noticed the slippery patch|Didn’t notice/ }),
  ).toBeVisible({ timeout: 50_000 })
  await page.getByRole('button', { name: 'End the puddle test' }).click()
  await expect(page.getByRole('heading', { name: 'Puddle test' })).toHaveCount(0)
})

test('each page sets its own title and canonical URL', async ({ page }) => {
  await page.goto('/results')
  await expect(page).toHaveTitle('Results · TrustLap')
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    'href',
    'https://trustlap.vercel.app/results',
  )
})

test('unknown routes show the not-found page via the SPA fallback', async ({ page }) => {
  await page.goto('/no-such-page')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('drove off the track')
})

test('results: the plain summary comes first, the full evaluation is folded', async ({ page }) => {
  await page.goto('/results')
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Does feeling the road help?')
  await expect(page.getByRole('heading', { name: 'Laps finished in each condition' })).toBeVisible()
  await expect(
    page.getByText(/Clearly (better|worse) in:|No clear difference in:/).first(),
  ).toBeVisible()
  // The detailed charts are there but collapsed until asked for.
  await expect(page.locator('details').first()).not.toHaveAttribute('open')
})

test('lab and how it works lead with the simple view', async ({ page }) => {
  await page.goto('/lab')
  await expect(page.getByRole('heading', { name: 'It got better with practice' })).toBeVisible({
    timeout: 15_000,
  })
  await page.goto('/how-it-works#step-3')
  await expect(
    page.getByRole('heading', { level: 2, name: 'Learn to feel the road' }),
  ).toBeVisible()
  await page.getByRole('button', { name: /Next step/ }).click()
  await expect(page.getByRole('heading', { level: 2, name: 'Know when it’s unsure' })).toBeVisible()
})
