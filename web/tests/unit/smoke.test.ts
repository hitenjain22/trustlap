import { describe, expect, it } from 'vitest'
import { strings } from '../../src/copy/strings.ts'

describe('smoke', () => {
  it('exports the copy table', () => {
    expect(strings.wordmark).toBe('TrustLap')
  })
})
