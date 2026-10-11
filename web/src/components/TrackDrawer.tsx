/**
 * Draw-a-track overlay (SPEC 6.1 drawn tracks, 10.2): the canvas becomes a faint grid, the user
 * drags a loop and releases near the start to close it. The stroke is simplified, scaled and
 * validated like a generated track, with a specific reason when it fails and an Auto-fix.
 * Keyboard users get "Generate a random track"; Esc cancels.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { strings } from '../copy/strings.ts'
import type { SimConfig } from '../sim/config.ts'
import { autoFixDrawing, boundaries, trackFromDrawing, type DrawnTrack } from '../sim/track.ts'
import { Button } from './Button.tsx'

const D = strings.drawTrack
const CLOSE_PX = 44 // release within this distance of the start to close the loop
const MIN_STEP_PX = 3

type Stage =
  | { kind: 'empty' }
  | { kind: 'drawing'; pts: number[] }
  | { kind: 'open'; pts: number[] } // released away from the start
  | { kind: 'result'; pts: number[]; drawn: DrawnTrack; fixed: boolean }

export function TrackDrawer({
  cfg,
  onUse,
  onRandom,
  onCancel,
}: {
  cfg: SimConfig['track']
  onUse: (controlPoints: Float64Array, widthM: number) => void
  onRandom: () => void
  onCancel: () => void
}) {
  const [stage, setStage] = useState<Stage>({ kind: 'empty' })
  const [size, setSize] = useState({ w: 800, h: 600 })
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => {
      if (e) setSize({ w: e.contentRect.width, h: e.contentRect.height })
    })
    ro.observe(el)
    el.focus()
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onCancel()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  const local = (e: React.PointerEvent) => {
    const r = e.currentTarget.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top] as const
  }
  // The stroke lives in a ref: pointer events can outpace React renders, and a stale state
  // closure would drop points.
  const stroke = useRef<number[]>([])
  const active = useRef(false)
  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    active.current = true
    stroke.current = [...local(e)]
    setStage({ kind: 'drawing', pts: stroke.current })
  }
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!active.current) return
    const [x, y] = local(e)
    const p = stroke.current
    if (Math.hypot(x - p[p.length - 2]!, y - p[p.length - 1]!) < MIN_STEP_PX) return
    stroke.current = [...p, x, y]
    setStage({ kind: 'drawing', pts: stroke.current })
  }
  const onUp = () => {
    if (!active.current) return
    active.current = false
    const p = stroke.current
    const n = p.length
    const closed = n >= 40 && Math.hypot(p[0]! - p[n - 2]!, p[1]! - p[n - 1]!) < CLOSE_PX
    if (!closed) {
      setStage({ kind: 'open', pts: p })
      return
    }
    // Trim the tail that wanders around the start (overshoot makes a hook); the loop closes
    // with a straight chord back to the start instead.
    let end = n
    while (end > 40 && Math.hypot(p[0]! - p[end - 2]!, p[1]! - p[end - 1]!) < CLOSE_PX * 0.6) {
      end -= 2
    }
    // Screen y points down; the world is y-up.
    const world = Float64Array.from(p.slice(0, end), (v, i) => (i % 2 ? -v : v))
    const drawn = trackFromDrawing(world, cfg)
    setStage(drawn ? { kind: 'result', pts: p, drawn, fixed: false } : { kind: 'open', pts: p })
  }

  const message =
    stage.kind === 'empty' || stage.kind === 'drawing'
      ? D.hint
      : stage.kind === 'open'
        ? D.notClosed
        : stage.drawn.reason
          ? D.reasons[stage.drawn.reason]
          : stage.fixed
            ? D.fixed
            : D.valid
  const result = stage.kind === 'result' ? stage.drawn : null

  // The validated track, fitted into the overlay for preview.
  const preview = useMemo(() => {
    if (!result) return null
    const { left, right } = boundaries(result.track)
    const c = result.track.centerline
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const arr of [left, right]) {
      for (let i = 0; i < arr.length; i += 2) {
        minX = Math.min(minX, arr[i]!)
        maxX = Math.max(maxX, arr[i]!)
        minY = Math.min(minY, arr[i + 1]!)
        maxY = Math.max(maxY, arr[i + 1]!)
      }
    }
    const pad = 48
    const s = Math.min((size.w - 2 * pad) / (maxX - minX), (size.h - 2 * pad) / (maxY - minY))
    const ox = size.w / 2 - ((minX + maxX) / 2) * s
    const oy = size.h / 2 + ((minY + maxY) / 2) * s
    const path = (arr: Float64Array | Float32Array) => {
      let d = ''
      for (let i = 0; i < arr.length; i += 2) {
        d += `${i ? 'L' : 'M'}${(ox + arr[i]! * s).toFixed(1)} ${(oy - arr[i + 1]! * s).toFixed(1)}`
      }
      return `${d}Z`
    }
    return { center: path(c), width: result.track.widthM * s }
  }, [result, size])

  const strokePts = stage.kind === 'empty' ? [] : stage.pts
  const strokePath = strokePts.reduce(
    (d, v, i) => (i % 2 ? `${d} ${v.toFixed(1)}` : `${d}${i ? 'L' : 'M'}${v.toFixed(1)}`),
    '',
  )

  return (
    <div
      ref={boxRef}
      className="absolute inset-0 z-10 bg-asphalt outline-none"
      tabIndex={-1}
      role="dialog"
      aria-label={D.title}
      aria-describedby="draw-msg"
    >
      <svg
        className="absolute inset-0 h-full w-full touch-none select-none"
        style={{ cursor: 'crosshair' }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        aria-hidden="true"
      >
        <defs>
          <pattern id="draw-grid" width="32" height="32" patternUnits="userSpaceOnUse">
            <path d="M32 0H0V32" fill="none" stroke="var(--line)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#draw-grid)" opacity="0.5" />
        {preview ? (
          <>
            <path
              d={preview.center}
              fill="none"
              stroke="var(--asphalt-700)"
              strokeWidth={Math.max(6, preview.width)}
              strokeLinejoin="round"
            />
            <path
              d={preview.center}
              fill="none"
              stroke={result?.reason ? 'var(--text-3)' : 'var(--paint)'}
              strokeWidth={1.5}
              strokeDasharray={result?.reason ? '6 6' : undefined}
            />
          </>
        ) : (
          strokePath && (
            <path
              d={strokePath}
              fill="none"
              stroke="var(--paint)"
              strokeWidth={3}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )
        )}
        {!preview && strokePts.length >= 2 && (
          <circle
            cx={strokePts[0]}
            cy={strokePts[1]}
            r={CLOSE_PX}
            fill="none"
            stroke="var(--sodium)"
            strokeDasharray="4 6"
          />
        )}
      </svg>

      <div className="pointer-events-none absolute inset-x-0 top-0 flex flex-col gap-3 p-4">
        <p
          id="draw-msg"
          role="status"
          className="max-w-[60ch] self-start rounded-panel border border-line bg-asphalt-800/95 px-3 py-2 text-[14px] text-paint"
        >
          {message}
        </p>
      </div>
      <div className="absolute inset-x-0 bottom-0 flex flex-wrap gap-2 p-4">
        <Button
          variant="primary"
          disabled={!result || result.reason !== null}
          onClick={() => result && onUse(result.controlPoints, result.track.widthM)}
        >
          {D.use}
        </Button>
        {result?.reason && (
          <Button
            variant="ghost"
            onClick={() => {
              if (stage.kind !== 'result') return
              setStage({ ...stage, drawn: autoFixDrawing(stage.drawn, cfg), fixed: true })
            }}
          >
            {D.autoFix}
          </Button>
        )}
        {stage.kind !== 'empty' && (
          <Button variant="ghost" onClick={() => setStage({ kind: 'empty' })}>
            {D.clear}
          </Button>
        )}
        <Button variant="ghost" onClick={onRandom}>
          {D.random}
        </Button>
        <Button variant="ghost" onClick={onCancel}>
          {D.cancel}
        </Button>
      </div>
    </div>
  )
}
