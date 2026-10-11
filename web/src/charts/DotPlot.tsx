/**
 * Dot plot with 95% CI whiskers (SPEC 11.6): one row per category, one dot per driver,
 * 1.5 px whiskers with 6 px caps, ≥ 8 px markers with a 2 px surface ring, per-mark tooltips.
 * `zero` draws an emphasized zero line for paired differences (diverging).
 */
import { scaleLinear } from 'd3-scale'
import { useState } from 'react'
import type { ChartDriver } from './drivers.ts'
import { DRIVER_STYLE } from './drivers.ts'
import { Marker } from './Marker.tsx'
import { Tooltip, type TipState } from './Tooltip.tsx'
import { useWidth } from './useWidth.ts'

export interface DotDatum {
  row: string
  driver: ChartDriver
  value: number
  ci: [number, number]
}

export function DotPlot({
  rows,
  data,
  domain,
  format,
  axisLabel,
  zero = false,
  rowHeight = 34,
}: {
  rows: string[]
  data: DotDatum[]
  domain: [number, number]
  format: (v: number) => string
  axisLabel: string
  zero?: boolean
  rowHeight?: number
}) {
  const { ref, width } = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<TipState | null>(null)
  const labelW = width < 520 ? 104 : 150
  const margin = { top: 8, right: 16, bottom: 40, left: labelW }
  const drivers = [...new Set(data.map((d) => d.driver))]
  const height = margin.top + rows.length * rowHeight + margin.bottom
  const x = scaleLinear()
    .domain(domain)
    .nice()
    .range([margin.left, width - margin.right])
  const ticks = x.ticks(width < 520 ? 4 : 6)
  const offset = (d: ChartDriver) => {
    const i = drivers.indexOf(d)
    return drivers.length > 1
      ? (i - (drivers.length - 1) / 2) * Math.min(8, (rowHeight - 12) / drivers.length)
      : 0
  }
  return (
    <div ref={ref} className="relative" onMouseLeave={() => setTip(null)}>
      <svg width={width} height={height} className="block overflow-visible">
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={x(t)}
              x2={x(t)}
              y1={margin.top}
              y2={height - margin.bottom}
              stroke="var(--asphalt-700)"
              strokeWidth={zero && t === 0 ? 0 : 1}
            />
            <text
              x={x(t)}
              y={height - margin.bottom + 18}
              textAnchor="middle"
              className="fill-[var(--text-3)] text-[12px] tabular-nums"
            >
              {format(t)}
            </text>
          </g>
        ))}
        {zero && (
          <line
            x1={x(0)}
            x2={x(0)}
            y1={margin.top}
            y2={height - margin.bottom}
            stroke="var(--text-3)"
            strokeWidth={1.5}
          />
        )}
        <text
          x={(margin.left + width - margin.right) / 2}
          y={height - 6}
          textAnchor="middle"
          className="fill-[var(--text-3)] text-[12px]"
        >
          {axisLabel}
        </text>
        {rows.map((r, i) => {
          const cy = margin.top + i * rowHeight + rowHeight / 2
          return (
            <g key={r}>
              <text
                x={margin.left - 12}
                y={cy + 4}
                textAnchor="end"
                className="fill-[var(--paint)] text-[13px]"
              >
                {r}
              </text>
              {data
                .filter((d) => d.row === r)
                .map((d) => {
                  const y = cy + offset(d.driver)
                  const lo = x(Math.max(domain[0], d.ci[0]))
                  const hi = x(Math.min(x.domain()[1]!, d.ci[1]))
                  const color = DRIVER_STYLE[d.driver].color
                  return (
                    <g
                      key={d.driver}
                      onMouseEnter={() =>
                        setTip({
                          x: x(d.value),
                          y,
                          lines: [
                            `${DRIVER_STYLE[d.driver].label}, ${r}`,
                            `${format(d.value)} (95% CI ${format(d.ci[0])} to ${format(d.ci[1])})`,
                          ],
                        })
                      }
                    >
                      <rect
                        x={margin.left}
                        y={y - 8}
                        width={width - margin.left - margin.right}
                        height={16}
                        fill="transparent"
                      />
                      <line x1={lo} x2={hi} y1={y} y2={y} stroke={color} strokeWidth={1.5} />
                      <line
                        x1={lo}
                        x2={lo}
                        y1={y - 3}
                        y2={y + 3}
                        stroke={color}
                        strokeWidth={1.5}
                      />
                      <line
                        x1={hi}
                        x2={hi}
                        y1={y - 3}
                        y2={y + 3}
                        stroke={color}
                        strokeWidth={1.5}
                      />
                      <Marker driver={d.driver} x={x(d.value)} y={y} />
                    </g>
                  )
                })}
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} width={width} />
    </div>
  )
}
