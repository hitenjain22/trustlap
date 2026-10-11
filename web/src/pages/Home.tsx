import { strings } from '../copy/strings.ts'

/** Placeholder home page (M0). Replaced by the live hero in M2. */
export default function Home() {
  return (
    <section className="mx-auto max-w-[1280px] px-4 py-16 md:px-6 md:py-24">
      <h1 className="max-w-[14ch] text-[clamp(44px,6vw,76px)] leading-[1.02] font-[760] tracking-[-0.02em] [font-stretch:125%]">
        {strings.placeholder.headline}
      </h1>
      <p className="mt-6 max-w-[68ch] text-text-2">{strings.placeholder.body}</p>
    </section>
  )
}
