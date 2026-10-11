/** Driver identity for charts (SPEC 11.3): token color + dash pattern + display name. */
export type ChartDriver = 'student' | 'student_governed' | 'teacher' | 'blind' | 'scripted'

export const DRIVER_STYLE: Record<
  ChartDriver,
  {
    color: string
    dash: string
    label: string
    marker: 'circle' | 'square' | 'diamond' | 'triangle' | 'ring'
  }
> = {
  student: { color: 'var(--drv-student)', dash: '', label: 'Student (adaptive)', marker: 'circle' },
  student_governed: {
    color: 'var(--drv-student)',
    dash: '2 3',
    label: 'Student + safety governor',
    marker: 'ring',
  },
  teacher: {
    color: 'var(--drv-teacher)',
    dash: '2 4',
    label: 'Teacher (knows conditions)',
    marker: 'diamond',
  },
  blind: { color: 'var(--drv-blind)', dash: '6 4', label: 'Blind', marker: 'square' },
  scripted: {
    color: 'var(--drv-scripted)',
    dash: '10 4',
    label: 'Scripted (knows grip)',
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
