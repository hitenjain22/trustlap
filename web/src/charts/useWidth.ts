import { useEffect, useRef, useState } from 'react'

/** Track an element's content width (charts are sized in CSS px, responsive). */
export function useWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T | null>(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) =>
      setWidth(Math.max(280, Math.floor(entry!.contentRect.width))),
    )
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return { ref, width }
}
