/**
 * Line chart (SPEC 11.6): 2 px lines, optional markers and CI crosses, linear or log axes,
 * crosshair + tooltip, direct labels at line ends for ≤ 4 series. One y-axis, always.
 */
import { scaleLinear, scaleLog } from 'd3-scale'
import { line as d3line, curveLinear, curveStepAfter } from 'd3-shape'
import { useState } from 'react'
import { Tooltip, type TipState } from './Tooltip.tsx'
import { useWidth } from './useWidth.ts'

export interface Series {
  id: string
  label: string
  color: string
  dash?: string
  points: { x: number; y: number; xCi?: [number, number]; yCi?: [number, number]; note?: string }[]
  markers?: boolean
  step?: boolean
  width?: number
  opacity?: number
}

export function LineChart({
  series,
  xLabel,
  yLabel,
  formatX,
  formatY,
  logX = false,
  logY = false,
  yDomain,
  xDomain,
  height = 300,
  vline,
  diagonal = false,
  directLabels = true,
  onHoverSeries,
}: {
  series: Series[]
  xLabel: string
  yLabel: string
  formatX: (v: number) => string
  formatY: (v: number) => string
  logX?: boolean
  logY?: boolean
  yDomain?: [number, number]
  xDomain?: [number, number]
  height?: number
  vline?: { x: number; label: string }
  diagonal?: boolean
  directLabels?: boolean
  onHoverSeries?: (id: string | null) => void
}) {
  const { ref, width } = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<TipState | null>(null)
  const [cross, setCross] = useState<number | null>(null)
  const labelRoom = directLabels && series.length <= 4 && width > 520 ? 110 : 16
  const margin = { top: 12, right: labelRoom, bottom: 42, left: 56 }
  const all = series.flatMap((s) => s.points)
  const xs = all.flatMap((p) => [p.x, ...(p.xCi ?? [])]).filter(Number.isFinite)
  const ys = all.flatMap((p) => [p.y, ...(p.yCi ?? [])]).filter(Number.isFinite)
  const xd =
    xDomain ??
    (logX ? [Math.min(...xs) / 1.5, Math.max(...xs) * 1.5] : [Math.min(...xs), Math.max(...xs)])
  // Linear axes start at zero; log axes can't, so they span the data.
  const yd =
    yDomain ??
    (logY ? [Math.min(...ys) / 1.5, Math.max(...ys) * 1.5] : [Math.min(0, ...ys), Math.max(...ys)])
  const x = (logX ? scaleLog() : scaleLinear())
    .domain(xd)
    .range([margin.left, width - margin.right])
  const y = (logY ? scaleLog() : scaleLinear())
    .domain(yd)
    .range([height - margin.bottom, margin.top])
  if (!logX) (x as ReturnType<typeof scaleLinear<number, number>>).nice()
  if (!logY) (y as ReturnType<typeof scaleLinear<number, number>>).nice()
  const xTicks = logX
    ? x.ticks(5).filter((t) => Math.log10(t) % 1 === 0)
    : x.ticks(width < 520 ? 4 : 6)
  const yTicks = logY ? y.ticks(5).filter((t) => Math.log10(t) % 1 === 0) : y.ticks(5)

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - box.left + margin.left
    const xv = x.invert(px)
    // Nearest point per series to the crosshair.
    const lines: string[] = [`${xLabel}: ${formatX(xv)}`]
    for (const s of series) {
      let best = s.points[0]
      for (const p of s.points) if (best && Math.abs(p.x - xv) < Math.abs(best.x - xv)) best = p
      if (best)
        lines.push(
          `${s.label}: ${formatY(best.y)}${best.yCi ? ` (${formatY(best.yCi[0])}–${formatY(best.yCi[1])})` : ''}${best.note ? ` · ${best.note}` : ''}`,
        )
    }
    setCross(px)
    setTip({ x: px, y: margin.top + 10, lines: lines.slice(0, 7) })
  }

  return (
    <div ref={ref} className="relative" onMouseLeave={() => (setTip(null), setCross(null))}>
      <svg width={width} height={height} className="block overflow-visible">
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line
              x1={margin.left}
              x2={width - margin.right}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--asphalt-700)"
            />
            <text
              x={margin.left - 8}
              y={y(t) + 4}
              textAnchor="end"
              className="fill-[var(--text-3)] text-[12px] tabular-nums"
            >
              {formatY(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text
            key={`x${t}`}
            x={x(t)}
            y={height - margin.bottom + 18}
            textAnchor="middle"
            className="fill-[var(--text-3)] text-[12px] tabular-nums"
          >
            {formatX(t)}
          </text>
        ))}
        <text
          x={(margin.left + width - margin.right) / 2}
          y={height - 6}
          textAnchor="middle"
          className="fill-[var(--text-3)] text-[12px]"
        >
          {xLabel}
        </text>
        <text
          transform={`translate(14 ${(margin.top + height - margin.bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
          className="fill-[var(--text-3)] text-[12px]"
        >
          {yLabel}
        </text>
        {diagonal && (
          <line
            x1={x(xd[0])}
            y1={y(xd[0])}
            x2={x(xd[1])}
            y2={y(xd[1])}
            stroke="var(--text-3)"
            strokeDasharray="4 4"
          />
        )}
        {vline && (
          <g>
            <line
              x1={x(vline.x)}
              x2={x(vline.x)}
              y1={margin.top}
              y2={height - margin.bottom}
              stroke="var(--text-3)"
              strokeDasharray="3 3"
            />
            <text
              x={x(vline.x) + 6}
              y={margin.top + 12}
              className="fill-[var(--text-2)] text-[12px]"
            >
              {vline.label}
            </text>
          </g>
        )}
        {series.map((s) => {
          const path = d3line<Series['points'][number]>()
            .defined((p) => Number.isFinite(p.y))
            .x((p) => x(p.x))
            .y((p) => y(p.y))
            .curve(s.step ? curveStepAfter : curveLinear)(s.points)
          const last = [...s.points].reverse().find((p) => Number.isFinite(p.y))
          return (
            <g
              key={s.id}
              opacity={s.opacity ?? 1}
              onMouseEnter={() => onHoverSeries?.(s.id)}
              onMouseLeave={() => onHoverSeries?.(null)}
            >
              {s.points.map((p, i) => (
                <g key={i}>
                  {p.yCi && (
                    <line
                      x1={x(p.x)}
                      x2={x(p.x)}
                      y1={y(p.yCi[0])}
                      y2={y(p.yCi[1])}
                      stroke={s.color}
                      strokeWidth={1.5}
                    />
                  )}
                  {p.xCi && (
                    <line
                      x1={x(p.xCi[0])}
                      x2={x(p.xCi[1])}
                      y1={y(p.y)}
                      y2={y(p.y)}
                      stroke={s.color}
                      strokeWidth={1.5}
                    />
                  )}
                </g>
              ))}
              {path && (
                <path
                  d={path}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={s.width ?? 2}
                  strokeDasharray={s.dash}
                  strokeLinejoin="round"
                />
              )}
              {s.markers &&
                s.points.map((p, i) =>
                  Number.isFinite(p.y) ? (
                    <circle
                      key={i}
                      cx={x(p.x)}
                      cy={y(p.y)}
                      r={4.5}
                      fill={s.color}
                      stroke="var(--asphalt)"
                      strokeWidth={2}
                    />
                  ) : null,
                )}
              {directLabels && series.length <= 4 && width > 520 && last && (
                <text
                  x={x(last.x) + 8}
                  y={y(last.y) + 4}
                  className="fill-[var(--text-2)] text-[12px]"
                >
                  {s.label}
                </text>
              )}
            </g>
          )
        })}
        {cross !== null && (
          <line
            x1={cross}
            x2={cross}
            y1={margin.top}
            y2={height - margin.bottom}
            stroke="var(--text-3)"
            strokeWidth={1}
          />
        )}
        <rect
          x={margin.left}
          y={margin.top}
          width={Math.max(0, width - margin.left - margin.right)}
          height={height - margin.top - margin.bottom}
          fill="transparent"
          onMouseMove={onMove}
          style={{ pointerEvents: onHoverSeries ? 'none' : 'all' }}
        />
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  )
}
