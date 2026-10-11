import { defineConfig, devices } from '@playwright/test'

const PORT = 4173

/** E2E tests run against the production build served by `vite preview` (build first). */
export default defineConfig({
  testDir: 'tests/e2e',
  // One at a time: the live-simulation tests measure speeds, and parallel runs compete for CPU,
  // which trips the hero's slow-device fallback and makes those measurements flaky.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['html', { open: 'never' }], ['list']] : 'list',
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `npm run preview -- --port ${PORT} --strictPort`,
    port: PORT,
    reuseExistingServer: !process.env.CI,
  },
})
