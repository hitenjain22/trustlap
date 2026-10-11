/** Chart tooltip: positioned in the chart's box, text in text tokens (never series colors). */
export interface TipState {
  x: number
  y: number
  lines: string[]
}

export function Tooltip({ tip, width }: { tip: TipState | null; width: number }) {
  if (!tip) return null
  const left = Math.min(Math.max(tip.x + 12, 0), width - 220)
  return (
    <div
      className="pointer-events-none absolute z-10 max-w-[220px] rounded-control border border-line bg-asphalt-800 px-3 py-2 text-[13px] text-paint"
      style={{ left, top: Math.max(0, tip.y - 12) }}
      role="status"
    >
      {tip.lines.map((l, i) => (
        <div key={i} className={i === 0 ? 'font-[560]' : 'text-text-2'}>
          {l}
        </div>
      ))}
    </div>
  )
}
