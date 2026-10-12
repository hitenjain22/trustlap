/**
 * Which companion accent carries which idea, so the same idea keeps the same color on every
 * page (UI tint and icons only; data series use the driver colors).
 */
export const STEP_ACCENTS = [
  'var(--sodium)', // build a world
  'var(--violet)', // practise (the AI learning)
  'var(--teal)', // feel the road (grip)
  'var(--sky)', // know when unsure
  'var(--coral)', // test it fairly
] as const

export const SCENARIO_ACCENTS: Record<string, string> = {
  dry: 'var(--sodium)',
  monsoon: 'var(--sky)',
  ice: 'var(--teal)',
  broken: 'var(--coral)',
  chaos: 'var(--violet)',
  puddle: 'var(--sky)',
}

/** Inline style that sets the --accent custom property. */
export const accent = (color: string) => ({ '--accent': color }) as React.CSSProperties

/** Icon names (components/icons.tsx LineIcon) for the five How it works steps. */
export const STEP_ICONS = ['world', 'practise', 'grip', 'unsure', 'test'] as const
