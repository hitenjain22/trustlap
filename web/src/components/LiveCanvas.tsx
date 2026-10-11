/** The live simulator canvas plus its loading / error states (SPEC 10.3). */
import { useEffect } from 'react'
import { strings } from '../copy/strings.ts'
import { useUi } from '../state/store.ts'
import { Button } from './Button.tsx'

export function LiveCanvas({
  canvasRef,
  phase,
  label,
  className = '',
}: {
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  phase: 'loading' | 'ready' | 'error'
  label: string
  className?: string
}) {
  const { startLoading, doneLoading } = useUi()
  useEffect(() => {
    if (phase !== 'loading') return
    startLoading()
    return () => doneLoading()
  }, [phase, startLoading, doneLoading])

  if (typeof Worker === 'undefined') {
    return <p className="p-6 text-text-2">{strings.states.noCanvas}</p>
  }
  return (
    <div className={`relative ${className}`}>
      <canvas ref={canvasRef} role="img" aria-label={label} className="block h-full w-full" />
      {phase === 'error' && (
        <div className="absolute inset-0 grid place-items-center bg-asphalt/80 p-6">
          <div className="max-w-sm text-center">
            <p className="text-paint">{strings.states.loadError}</p>
            <Button className="mt-4" onClick={() => window.location.reload()}>
              {strings.states.reload}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
