/** Canvas colors, read from the CSS design tokens (SPEC 11.2) so there is one source of truth. */
export interface Palette {
  runoff: string
  track: string
  paint: string
  line: string
  kerbRed: string
  grip4: string
  rain: string
  sodium: string
  drvStudent: string
  drvTeacher: string
  drvBlind: string
  drvScripted: string
  human: string // the visitor's own car (Race page): teal, outside the driver data palette
  skid: string
}

export function readPalette(): Palette {
  const css = getComputedStyle(document.documentElement)
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback
  return {
    runoff: v('--runoff', '#181c21'),
    track: v('--track', '#2a3038'),
    paint: v('--paint', '#e9edf0'),
    line: v('--line', '#3a424d'),
    kerbRed: v('--kerb-red', '#c8463c'),
    grip4: v('--grip-4', '#3f7fbf'),
    rain: v('--rain', '#9fc5e8'),
    sodium: v('--sodium', '#f2b544'),
    drvStudent: v('--drv-student', '#c98500'),
    drvTeacher: v('--drv-teacher', '#9085e9'),
    drvBlind: v('--drv-blind', '#d55181'),
    drvScripted: v('--drv-scripted', '#8a949d'),
    human: v('--teal', '#38d1bd'),
    skid: '#0f1215',
  }
}

/** "#rrggbb" + alpha → "rgba(...)". */
export function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.replace('#', ''), 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`
}
