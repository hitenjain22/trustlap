import { createBrowserRouter } from 'react-router'
import App from './App.tsx'
import { lazy, Suspense } from 'react'
import Home from './pages/Home.tsx'
import NotFound from './pages/NotFound.tsx'

const Playground = lazy(() => import('./pages/Playground.tsx'))
const Results = lazy(() => import('./pages/Results.tsx'))
const Lab = lazy(() => import('./pages/Lab.tsx'))
const How = lazy(() => import('./pages/How.tsx'))

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Home /> },
      {
        path: 'how-it-works',
        element: (
          <Suspense fallback={null}>
            <How />
          </Suspense>
        ),
      },
      {
        path: 'lab',
        element: (
          <Suspense fallback={null}>
            <Lab />
          </Suspense>
        ),
      },
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
