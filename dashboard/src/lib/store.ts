/**
 * Global application state via Zustand.
 */

import { create } from "zustand";
import type { OverviewStats, Identity, RecognitionEvent } from "./api";

interface AppState {
  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // Overview stats (cached)
  stats: OverviewStats | null;
  setStats: (stats: OverviewStats) => void;

  // Active identities list
  identities: Identity[];
  setIdentities: (ids: Identity[]) => void;

  // Recent events feed
  recentEvents: RecognitionEvent[];
  addEvent: (event: RecognitionEvent) => void;
  setEvents: (events: RecognitionEvent[]) => void;

  // Live feed state
  isStreaming: boolean;
  setStreaming: (v: boolean) => void;
  fps: number;
  setFps: (v: number) => void;

  // Theme
  theme: "dark" | "light";
  toggleTheme: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  stats: null,
  setStats: (stats) => set({ stats }),

  identities: [],
  setIdentities: (identities) => set({ identities }),

  recentEvents: [],
  addEvent: (event) =>
    set((s) => ({
      recentEvents: [event, ...s.recentEvents].slice(0, 100),
    })),
  setEvents: (recentEvents) => set({ recentEvents }),

  isStreaming: false,
  setStreaming: (isStreaming) => set({ isStreaming }),
  fps: 0,
  setFps: (fps) => set({ fps }),

  theme: "dark",
  toggleTheme: () =>
    set((s) => {
      const next = s.theme === "dark" ? "light" : "dark";
      document.documentElement.classList.toggle("dark", next === "dark");
      return { theme: next };
    }),
}));
