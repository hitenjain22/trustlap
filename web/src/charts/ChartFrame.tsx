/**
 * Figure wrapper (SPEC 10.2 Results, 11.6): the title states the finding, the subtitle names
 * the method and n, then the legend, the chart (or its table), the "Show table" toggle and
 * the source line. Missing data renders "Not computed yet", never a fake chart.
 */
import { useId, useState, type ReactNode } from 'react'
import { strings } from '../copy/strings.ts'
import type { ChartDriver } from './drivers.ts'
import { DRIVER_STYLE } from './drivers.ts'
import { Marker } from './Marker.tsx'

export interface TableData {
  columns: string[]
  rows: (string | number)[][]
}

export function Legend({ drivers }: { drivers: ChartDriver[] }) {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-2 text-[13px] text-text-2" aria-label="Legend">
      {drivers.map((d) => (
        <li key={d} className="flex items-center gap-2">
          <svg width="30" height="12" aria-hidden="true">
            <line
              x1="1"
              x2="29"
              y1="6"
              y2="6"
              stroke={DRIVER_STYLE[d].color}
              strokeWidth="2"
              strokeDasharray={DRIVER_STYLE[d].dash}
            />
            <Marker driver={d} x={15} y={6} size={8} />
          </svg>
          {DRIVER_STYLE[d].label}
        </li>
      ))}
    </ul>
  )
}

export function ChartFrame({
  title,
  subtitle,
  summary,
  drivers,
  table,
  source,
  missing,
  children,
}: {
  title: string
  subtitle: string
  summary: string
  drivers?: ChartDriver[]
  table?: TableData | null
  source: string
  missing?: boolean
  children: ReactNode
}) {
  const [showTable, setShowTable] = useState(false)
  const id = useId()
  return (
    <figure className="border-t border-line pt-6" aria-labelledby={`${id}-t`}>
      <h3 id={`${id}-t`} className="h3 max-w-[60ch]">
        {title}
      </h3>
      <p className="mt-1 max-w-[72ch] text-[14px] text-text-2">{subtitle}</p>
      {missing ? (
        <p className="mt-6 rounded-panel border border-dashed border-line px-4 py-8 text-center text-text-3">
          {strings.results.notComputed}
        </p>
      ) : (
        <>
          {drivers && drivers.length > 1 && (
            <div className="mt-4">
              <Legend drivers={drivers} />
            </div>
          )}
          <div className="mt-4" role="img" aria-label={summary} hidden={showTable}>
            {children}
          </div>
          {showTable && table && <TableView table={table} caption={title} />}
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <p className="text-[13px] text-text-3">{source}</p>
            {table && (
              <button
                type="button"
                className="text-[14px] text-sodium underline"
                aria-pressed={showTable}
                onClick={() => setShowTable((s) => !s)}
              >
                {showTable ? strings.results.showChart : strings.results.showTable}
              </button>
            )}
          </div>
        </>
      )}
    </figure>
  )
}

export function TableView({ table, caption }: { table: TableData; caption: string }) {
  return (
    <div className="mt-4 max-w-full overflow-x-auto">
      <table className="w-full border-collapse text-left text-[14px] tabular-nums">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {table.columns.map((c) => (
              <th
                key={c}
                scope="col"
                className="border-b border-line px-2 py-2 font-[560] text-text-2"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((r, i) => (
            <tr key={i}>
              {r.map((cell, j) =>
                j === 0 ? (
                  <th
                    key={j}
                    scope="row"
                    className="border-b border-line/60 px-2 py-1.5 font-[450] text-paint"
                  >
                    {cell}
                  </th>
                ) : (
                  <td key={j} className="border-b border-line/60 px-2 py-1.5 text-paint">
                    {cell}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
