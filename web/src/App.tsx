import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router'
import { GitHubIcon } from './components/icons.tsx'
import { REPO_URL, strings } from './copy/strings.ts'
import { ARTIFACT_VERSION } from './data/artifacts.ts'
import { useUi } from './state/store.ts'

/** App shell: sticky top nav with a loading line, page outlet, footer (SPEC 10.1). */
export default function App() {
  const loading = useUi((s) => s.loading > 0)
  // Small screens: the page links collapse into a menu (closes on navigation and Escape).
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()
  const [menuPath, setMenuPath] = useState(location.pathname)
  if (menuPath !== location.pathname) {
    setMenuPath(location.pathname)
    setMenuOpen(false)
  }
  useEffect(() => {
    if (!menuOpen) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setMenuOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [menuOpen])
  const links = (
    <>
      <NavLink to="/playground" className="navlink">
        {strings.nav.playground}
      </NavLink>
      <NavLink to="/lab" className="navlink">
        {strings.nav.lab}
      </NavLink>
      <NavLink to="/results" className="navlink">
        {strings.nav.results}
      </NavLink>
      <NavLink to="/how-it-works" className="navlink">
        {strings.nav.how}
      </NavLink>
    </>
  )
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-asphalt-800">
        <nav
          className="mx-auto flex h-14 max-w-[1280px] items-center gap-6 px-4 md:px-6"
          aria-label="Main"
        >
          <Link to="/" className="wordmark text-[22px] text-paint no-underline">
            {strings.wordmark}
          </Link>
          <div className="hidden flex-1 items-center gap-1 md:flex">{links}</div>
          <div className="flex-1 md:hidden" />
          <button
            type="button"
            className="min-h-11 rounded-control px-3 text-[15px] font-[560] text-text-2 hover:text-paint md:hidden"
            aria-expanded={menuOpen}
            aria-controls="mobile-nav"
            onClick={() => setMenuOpen((v) => !v)}
          >
            {strings.nav.menu}
          </button>
          <a
            href={REPO_URL}
            className="grid size-11 place-items-center rounded-control text-text-2 hover:text-paint"
            aria-label={strings.nav.github}
          >
            <GitHubIcon />
          </a>
        </nav>
        {menuOpen && (
          <div
            id="mobile-nav"
            className="flex flex-col gap-1 border-b border-line bg-asphalt-800 px-4 pb-3 md:hidden"
          >
            {links}
          </div>
        )}
        <div
          className={`progress-line ${loading ? 'progress-line--active' : ''}`}
          aria-hidden="true"
        />
      </header>
      <main id="main" className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-line px-4 py-10 text-sm text-text-2 md:px-6">
        <div className="mx-auto flex max-w-[1280px] flex-col gap-2">
          <p className="text-paint">
            {strings.footer.credit}{' '}
            <a href={REPO_URL} className="text-sodium underline">
              {strings.footer.source}
            </a>
          </p>
          <p>{strings.footer.attribution}</p>
          <p className="text-text-3">{strings.footer.version(ARTIFACT_VERSION, __BUILD_SHA__)}</p>
        </div>
      </footer>
    </div>
  )
}
