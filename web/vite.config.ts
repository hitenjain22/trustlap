import { execSync } from 'node:child_process'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

function buildSha(): string {
  const fromVercel = process.env.VERCEL_GIT_COMMIT_SHA
  if (fromVercel) return fromVercel.slice(0, 7)
  try {
    return execSync('git rev-parse --short HEAD').toString().trim()
  } catch {
    return 'dev'
  }
}

// https://vite.dev/config/
export default defineConfig({
  // Vercel serves from the domain root. Changing this breaks asset URLs (SPEC M0).
  base: '/',
  plugins: [react(), tailwindcss()],
  define: { __BUILD_SHA__: JSON.stringify(buildSha()) },
  worker: { format: 'es' },
  test: {
    include: ['tests/unit/**/*.test.ts', 'tests/parity/**/*.test.ts'],
    environment: 'node',
  },
})
