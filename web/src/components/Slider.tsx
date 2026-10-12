/**
 * Native range slider (SPEC 11.9) with the value + unit, a shaded band for the range the
 * drivers trained on, and a note when the value is outside it (SPEC 10.2).
 */
import { useId } from 'react'
import { strings } from '../copy/strings.ts'

export interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  format: (value: number) => string
  /** Spoken value, e.g. "60 percent rain". */
  valueText: (value: number) => string
  trained?: [number, number]
  size?: 'regular' | 'large'
  note?: string
  /** One plain line under the label: what changing this does. */
  hint?: string
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  valueText,
  trained,
  size = 'regular',
  note,
  hint,
}: SliderProps) {
  const id = useId()
  const pct = (v: number) => ((v - min) / (max - min)) * 100
  const outside = trained !== undefined && (value < trained[0] - 1e-9 || value > trained[1] + 1e-9)
  const band = trained
    ? { left: `${pct(trained[0])}%`, width: `${pct(trained[1]) - pct(trained[0])}%` }
    : null
  return (
    <div className={size === 'large' ? 'slider slider--large' : 'slider'}>
      <div className="flex items-baseline justify-between gap-3">
        <label
          htmlFor={id}
          className={size === 'large' ? 'text-[15px] font-[620] text-paint' : 'label'}
        >
          {label}
        </label>
        <output htmlFor={id} className="numeral text-[15px] text-paint">
          {format(value)}
        </output>
      </div>
      {hint && (
        <p id={`${id}-hint`} className="mt-0.5 text-[13px] text-text-3">
          {hint}
        </p>
      )}
      <div className="slider__track">
        {band && <span className="slider__band" style={band} aria-hidden="true" />}
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          aria-valuetext={valueText(value)}
          aria-describedby={hint ? `${id}-hint` : undefined}
          onChange={(e) => onChange(Number(e.currentTarget.value))}
        />
      </div>
      {outside && (
        <p className="mt-1 text-[13px] text-sodium">{strings.controls.outsideTraining}</p>
      )}
      {note && !outside && <p className="mt-1 text-[13px] text-text-3">{note}</p>}
    </div>
  )
}
