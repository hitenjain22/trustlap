import { expect, test } from '@playwright/test'

test('home page renders the wordmark and headline', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('link', { name: 'TrustLap' })).toBeVisible()
  await expect(page.getByRole('heading', { level: 1 })).toContainText('drive in the rain')
})

test('unknown routes show the not-found page via the SPA fallback', async ({ page }) => {
  await page.goto('/no-such-page')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('drove off the track')
})
