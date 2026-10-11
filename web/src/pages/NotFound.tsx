import { Link } from 'react-router'
import { strings } from '../copy/strings.ts'

export default function NotFound() {
  return (
    <section className="mx-auto max-w-[1280px] px-4 py-16 md:px-6 md:py-24">
      <h1 className="text-[49px] leading-[1.08] font-bold [font-stretch:118%]">
        {strings.notFound.title}
      </h1>
      <p className="mt-4 text-text-2">{strings.notFound.body}</p>
      <p className="mt-6">
        <Link to="/">{strings.notFound.home}</Link>
      </p>
    </section>
  )
}
