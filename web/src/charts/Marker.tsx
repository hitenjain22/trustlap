import type { ChartDriver } from './drivers.ts'
import { DRIVER_STYLE } from './drivers.ts'

/** Shape per driver so identity never relies on color alone. 8 px+ with a 2 px surface ring. */
export function Marker({
  driver,
  x,
  y,
  size = 9,
}: {
  driver: ChartDriver
  x: number
  y: number
  size?: number
}) {
  const s = DRIVER_STYLE[driver]
  const r = size / 2
  const common = { stroke: 'var(--asphalt)', strokeWidth: 2, paintOrder: 'stroke' as const }
  switch (s.marker) {
    case 'square':
      return (
        <rect x={x - r} y={y - r} width={size} height={size} rx={1.5} fill={s.color} {...common} />
      )
    case 'diamond':
      return (
        <path
          d={`M${x} ${y - r * 1.2}L${x + r * 1.2} ${y}L${x} ${y + r * 1.2}L${x - r * 1.2} ${y}Z`}
          fill={s.color}
          {...common}
        />
      )
    case 'triangle':
      return (
        <path
          d={`M${x} ${y - r * 1.15}L${x + r * 1.1} ${y + r * 0.85}L${x - r * 1.1} ${y + r * 0.85}Z`}
          fill={s.color}
          {...common}
        />
      )
    case 'ring':
      return (
        <circle
          cx={x}
          cy={y}
          r={r - 0.5}
          fill="var(--asphalt)"
          stroke={s.color}
          strokeWidth={2.5}
        />
      )
    default:
      return <circle cx={x} cy={y} r={r} fill={s.color} {...common} />
  }
}
