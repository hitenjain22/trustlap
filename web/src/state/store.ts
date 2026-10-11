/** Global UI state (SPEC 3.2: Zustand). For now: how many things are loading (progress line). */
import { create } from 'zustand'

interface UiState {
  loading: number
  startLoading: () => void
  doneLoading: () => void
}

export const useUi = create<UiState>((set) => ({
  loading: 0,
  startLoading: () => set((s) => ({ loading: s.loading + 1 })),
  doneLoading: () => set((s) => ({ loading: Math.max(0, s.loading - 1) })),
}))
