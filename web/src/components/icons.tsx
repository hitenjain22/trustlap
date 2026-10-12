/** Small inline icons (decorative; buttons carry text labels). */
export function DiceIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
    >
      <rect x="3.5" y="3.5" width="17" height="17" rx="3.5" />
      <circle cx="8.5" cy="8.5" r="1.3" fill="currentColor" stroke="none" />
      <circle cx="15.5" cy="15.5" r="1.3" fill="currentColor" stroke="none" />
      <circle cx="15.5" cy="8.5" r="1.3" fill="currentColor" stroke="none" />
      <circle cx="8.5" cy="15.5" r="1.3" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function GitHubIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  )
}

export function PlayIcon({ playing }: { playing: boolean }) {
  return playing ? (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <rect x="3" y="2" width="3.5" height="12" rx="1" />
      <rect x="9.5" y="2" width="3.5" height="12" rx="1" />
    </svg>
  ) : (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M4 2.5v11a.5.5 0 00.76.43l9-5.5a.5.5 0 000-.86l-9-5.5A.5.5 0 004 2.5z" />
    </svg>
  )
}

/** Line icons for the steps and scenarios (20 px, currentColor, decorative). */
const PATHS: Record<string, string> = {
  world:
    'M10 2a8 8 0 1 0 0 16a8 8 0 0 0 0-16M2.5 10h15M10 2.5c2.2 2.1 3.2 4.6 3.2 7.5s-1 5.4-3.2 7.5M10 2.5C7.8 4.6 6.8 7.1 6.8 10s1 5.4 3.2 7.5',
  practise:
    'M3.5 10a6.5 6.5 0 0 1 11.3-4.4M16.5 10a6.5 6.5 0 0 1-11.3 4.4M15 2.5v3.6h-3.6M5 17.5v-3.6h3.6',
  grip: 'M2.5 7c2.5-2 5-2 7.5 0s5 2 7.5 0M2.5 12c2.5-2 5-2 7.5 0s5 2 7.5 0',
  unsure: 'M3 14a7 7 0 1 1 14 0M10 14l3.5-4.5M2.5 14h2M15.5 14h2',
  test: 'M4.5 17.5V3M4.5 3.5h10l-2 3.5 2 3.5h-10',
  dry: 'M10 6.5a3.5 3.5 0 1 0 0 7a3.5 3.5 0 0 0 0-7M10 2v1.8M10 16.2V18M2 10h1.8M16.2 10H18M4.3 4.3l1.3 1.3M14.4 14.4l1.3 1.3M15.7 4.3l-1.3 1.3M5.6 14.4l-1.3 1.3',
  monsoon:
    'M5.5 12.5a3.5 3.5 0 0 1 .7-6.9a5 5 0 0 1 9.4 1.6a2.8 2.8 0 0 1-.6 5.3M7 15l-1 2.5M10.5 15l-1 2.5M14 15l-1 2.5',
  ice: 'M10 2v16M3.1 6l13.8 8M3.1 14l13.8-8M8 3.5l2 2 2-2M8 16.5l2-2 2 2',
  broken: 'M13.5 3a4 4 0 0 0-4.8 5.3L3 14l3 3 5.7-5.7A4 4 0 0 0 17 6.5l-2.4 2.4-2.4-.7-.7-2.4z',
  chaos: 'M11 2L4 11h5l-1 7 7-9h-5z',
  puddle: 'M10 2.5s-5 5.6-5 9a5 5 0 0 0 10 0c0-3.4-5-9-5-9',
}

export function LineIcon({ name }: { name: keyof typeof PATHS | string }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name] ?? ''} />
    </svg>
  )
}
