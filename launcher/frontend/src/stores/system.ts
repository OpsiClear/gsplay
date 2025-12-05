/**
 * System stats store.
 */

import { createSignal } from "solid-js";
import { api, SystemStats } from "../api/client";

const [systemStats, setSystemStats] = createSignal<SystemStats | null>(null);

async function fetchSystemStats(): Promise<void> {
  try {
    const data = await api.system.getStats();
    setSystemStats(data);
  } catch (e) {
    console.error("Failed to fetch system stats:", e);
  }
}

export const systemStore = {
  stats: systemStats,
  fetchStats: fetchSystemStats,
};
