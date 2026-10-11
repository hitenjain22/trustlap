import { createBrowserRouter } from 'react-router'
import App from './App.tsx'
import { lazy, Suspense } from 'react'
import Home from './pages/Home.tsx'
import NotFound from './pages/NotFound.tsx'

const Playground = lazy(() => import('./pages/Playground.tsx'))
const Results = lazy(() => import('./pages/Results.tsx'))

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Home /> },
      {
        path: 'results',
        element: (
          <Suspense fallback={null}>
            <Results />
          </Suspense>
        ),
      },
      {
        path: 'playground',
        element: (
          <Suspense fallback={null}>
            <Playground />
          </Suspense>
        ),
      },
      { path: '*', element: <NotFound /> },
    ],
  },
])
