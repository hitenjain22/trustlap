/** Driver identity for charts (SPEC 11.3): token color + dash pattern + display name. */
export type ChartDriver =
  'student' | 'student_governed' | 'hybrid' | 'teacher' | 'blind' | 'scripted'

export const DRIVER_STYLE: Record<
  ChartDriver,
  {
    color: string
    dash: string
    label: string
    marker: 'circle' | 'square' | 'diamond' | 'triangle' | 'ring'
  }
> = {
  student: { color: 'var(--drv-student)', dash: '', label: 'Adaptive AI', marker: 'circle' },
  student_governed: {
    color: 'var(--drv-student)',
    dash: '2 3',
    label: 'Adaptive AI + safety governor',
    marker: 'ring',
  },
  hybrid: {
    color: 'var(--drv-student)',
    dash: '6 3',
    label: 'Adaptive AI + dry-road fallback',
    marker: 'diamond',
  },
  teacher: {
    color: 'var(--drv-teacher)',
    dash: '2 4',
    label: 'Teacher AI (told the conditions)',
    marker: 'diamond',
  },
  blind: { color: 'var(--drv-blind)', dash: '6 4', label: 'Basic AI', marker: 'square' },
  scripted: {
    color: 'var(--drv-scripted)',
    dash: '10 4',
    label: 'Rule-based (told the grip)',
    marker: 'triangle',
  },
}

export const SUITE_LABEL: Record<string, string> = {
  dry: 'Dry',
  wet: 'Wet',
  puddles: 'Puddles',
  wind: 'Wind + gusts',
  worn: 'Worn tires',
  laggy: 'Laggy steering',
  heavy: 'Heavy, low power',
  combined: 'Everything at once',
  ice: 'Ice',
  extreme_lag: 'Extreme lag',
  noisy_sensors: 'Noisy sensors',
  very_heavy: 'Very heavy',
}
