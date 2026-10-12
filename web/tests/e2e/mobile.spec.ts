import { expect, test } from '@playwright/test'

test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true })

test('phone: no page scrolls sideways', async ({ page }) => {
  for (const path of ['/', '/playground', '/lab', '/results', '/how-it-works']) {
    await page.goto(path)
    await page.waitForTimeout(1500) // charts measure their width after mount
    const width = await page.evaluate(() => document.documentElement.scrollWidth)
    expect(width, path).toBeLessThanOrEqual(390)
  }
})

test('phone: the menu links every page and the conditions sheet has tabs', async ({ page }) => {
  await page.goto('/playground')
  await page.getByRole('button', { name: 'Menu' }).click()
  await expect(page.locator('#mobile-nav').getByRole('link')).toHaveCount(5)
  await page.getByRole('button', { name: 'Menu' }).click()
  await page.getByRole('button', { name: 'Conditions' }).click()
  await page.getByRole('tab', { name: 'More conditions' }).click()
  await expect(page.getByRole('slider', { name: 'Steering lag' })).toBeVisible()
  await expect(page.getByRole('slider', { name: 'Rain', exact: true })).toHaveCount(0)
})
