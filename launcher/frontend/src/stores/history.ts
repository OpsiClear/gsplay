/**
 * Launch history store.
 */

import { createSignal } from "solid-js";
import { api, BrowseLaunchRequest } from "../api/client";
import { showError } from "./error";
import { instancesStore } from "./instances";
import { gpuStore } from "./gpu";
import { launchSettingsStore } from "./launchSettings";
import { browseStore } from "./browse";

export interface LaunchHistoryEntry {
  id: string;
  timestamp: string;
  path: string;
  name: string;
  gpu: number | null;
  compact: boolean;
  view_only: boolean;
  viewer_id?: string;
  stream_token?: string;
}

interface LaunchHistoryStorage {
  rootPath: string;
  entries: LaunchHistoryEntry[];
}

const HISTORY_STORAGE_KEY = "gsplay_launch_history";

const [launchHistory, setLaunchHistory] = createSignal<LaunchHistoryEntry[]>([]);
const [historyLimit, setHistoryLimit] = createSignal(5);
const [historyExpanded, setHistoryExpanded] = createSignal(false);

function initHistory(currentRootPath: string | null, limit: number): void {
  setHistoryLimit(limit);
  if (!currentRootPath) {
    setLaunchHistory([]);
    return;
  }

  try {
    const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!stored) {
      setLaunchHistory([]);
      return;
    }
    const parsed: LaunchHistoryStorage = JSON.parse(stored);
    if (parsed.rootPath !== currentRootPath) {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
      setLaunchHistory([]);
      return;
    }
    setLaunchHistory(parsed.entries.slice(0, limit));
  } catch (e) {
    console.error("Failed to load launch history:", e);
    setLaunchHistory([]);
  }
}

function addHistoryEntry(entry: Omit<LaunchHistoryEntry, "id" | "timestamp">, rootPath: string): void {
  const newEntry: LaunchHistoryEntry = {
    ...entry,
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
  };

  const updated = [newEntry, ...launchHistory().filter((e) => e.path !== entry.path)].slice(0, historyLimit());
  setLaunchHistory(updated);

  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify({ rootPath, entries: updated } as LaunchHistoryStorage));
  } catch (e) {
    console.error("Failed to save launch history:", e);
  }
}

async function relaunchFromHistory(entry: LaunchHistoryEntry): Promise<void> {
  const config = browseStore.config();
  if (!config) return;

  const currentViewerId = launchSettingsStore.viewerId();
  const currentStreamToken = launchSettingsStore.streamToken();

  const data: BrowseLaunchRequest = {
    path: entry.path,
    name: entry.name,
    port: undefined,
    stream_port: -1,
    gpu: entry.gpu ?? gpuStore.selectedGpu(),
    compact: entry.compact,
    view_only: config.view_only || entry.view_only,
    viewer_id: currentViewerId || entry.viewer_id || undefined,
    stream_token: currentStreamToken || entry.stream_token || undefined,
  };

  try {
    await api.browse.launch(data, !!config.external_url);
    addHistoryEntry(
      {
        path: entry.path,
        name: entry.name,
        gpu: data.gpu ?? null,
        compact: entry.compact,
        view_only: entry.view_only,
        viewer_id: data.viewer_id,
        stream_token: data.stream_token,
      },
      config.root_path ?? ""
    );
    await instancesStore.fetchInstances();
  } catch (e) {
    showError(e instanceof Error ? e.message : "Failed to relaunch");
  }
}

function clearHistory(): void {
  setLaunchHistory([]);
  localStorage.removeItem(HISTORY_STORAGE_KEY);
}

function toggleHistoryExpanded(): void {
  setHistoryExpanded(!historyExpanded());
}

export const historyStore = {
  history: launchHistory,
  historyLimit,
  expanded: historyExpanded,
  initHistory,
  addHistoryEntry,
  relaunchFromHistory,
  clearHistory,
  toggleExpanded: toggleHistoryExpanded,
};
