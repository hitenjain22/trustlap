import { Link, Outlet } from 'react-router'
import { strings } from './copy/strings.ts'

/** App shell: top nav, page outlet, footer. Full nav links arrive with their pages. */
export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 h-14 border-b border-line bg-asphalt-800">
        <nav className="mx-auto flex h-full max-w-[1280px] items-center px-4 md:px-6">
          <Link
            to="/"
            className="text-xl font-extrabold tracking-tight text-paint no-underline [font-stretch:125%]"
          >
            {strings.wordmark}
          </Link>
        </nav>
      </header>
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-line px-4 py-8 text-sm text-text-3 md:px-6">
        <div className="mx-auto max-w-[1280px]">{strings.footer.credit}</div>
      </footer>
    </div>
  )
}
