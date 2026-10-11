import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  // Vercel serves from the domain root. Changing this breaks asset URLs (SPEC M0).
  base: '/',
  plugins: [react(), tailwindcss()],
  test: {
    include: ['tests/unit/**/*.test.ts', 'tests/parity/**/*.test.ts'],
    environment: 'node',
  },
})
