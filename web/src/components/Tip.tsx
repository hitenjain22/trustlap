/**
 * Hover / focus tooltip: the wrapped control gets aria-describedby, so screen readers read the
 * same text. Shown on pointer hover or keyboard focus (CSS), positioned above the control.
 */
import { cloneElement, useId, type ReactElement } from 'react'

export function Tip({
  text,
  children,
}: {
  text: string
  children: ReactElement<{ 'aria-describedby'?: string }>
}) {
  const id = useId()
  return (
    <span className="tip-wrap">
      {cloneElement(children, { 'aria-describedby': id })}
      <span role="tooltip" id={id} className="tip">
        {text}
      </span>
    </span>
  )
}
