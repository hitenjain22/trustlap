/** The same opening on every page: a colored eyebrow, the title, one paragraph of intro. */
import type { ReactNode } from 'react'
import { accent } from '../styles/accents.ts'

export function PageHeader({
  eyebrow,
  title,
  intro,
  tint = 'var(--sodium)',
}: {
  eyebrow: string
  title: ReactNode
  intro?: ReactNode
  tint?: string
}) {
  return (
    <header>
      <p className="eyebrow" style={accent(tint)}>
        {eyebrow}
      </p>
      <h1 className="mt-2 text-[40px] leading-[1.08] font-bold [font-stretch:118%] md:text-[49px]">
        {title}
      </h1>
      {intro && <p className="mt-3 max-w-[72ch] text-text-2">{intro}</p>}
    </header>
  )
}
