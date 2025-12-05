/**
 * File browser store.
 */

import { createSignal } from "solid-js";
import { api, BrowseConfig, DirectoryEntry, BrowseLaunchRequest } from "../api/client";
import { showError } from "./error";
import { instancesStore } from "./instances";
import { gpuStore } from "./gpu";
import { launchSettingsStore } from "./launchSettings";
import { historyStore } from "./history";

const [browseConfig, setBrowseConfig] = createSignal<BrowseConfig | null>(null);
const [currentPath, setCurrentPath] = createSignal("");
const [entries, setEntries] = createSignal<DirectoryEntry[]>([]);
const [breadcrumbs, setBreadcrumbs] = createSignal<{ name: string; path: string }[]>([]);

async function fetchBrowseConfig(): Promise<void> {
  try {
    const data = await api.browse.getConfig();
    setBrowseConfig(data);
    if (data.enabled) {
      await navigateTo("");
    }
  } catch (e) {
    console.error("Failed to fetch browse config:", e);
  }
}

async function navigateTo(path: string): Promise<void> {
  try {
    const data = await api.browse.list(path);
    setCurrentPath(data.current_path);
    setEntries(data.entries);
    setBreadcrumbs(data.breadcrumbs);
  } catch (e) {
    showError(e instanceof Error ? e.message : "Failed to browse path");
  }
}

async function launchFolder(entry: DirectoryEntry): Promise<void> {
  const config = browseConfig();
  if (!config) return;

  const data: BrowseLaunchRequest = {
    path: entry.path,
    name: launchSettingsStore.instanceName() || entry.name,
    port: launchSettingsStore.port() || undefined,
    stream_port: -1,
    gpu: gpuStore.selectedGpu(),
    compact: launchSettingsStore.compact(),
    view_only: config.view_only || launchSettingsStore.viewOnly(),
    viewer_id: launchSettingsStore.viewerId() || undefined,
    stream_token: launchSettingsStore.streamToken() || undefined,
  };

  try {
    await api.browse.launch(data, !!config.external_url);
    historyStore.addHistoryEntry(
      {
        path: entry.path,
        name: data.name || entry.name,
        gpu: data.gpu ?? null,
        compact: data.compact ?? false,
        view_only: data.view_only ?? false,
        viewer_id: data.viewer_id ?? undefined,
        stream_token: data.stream_token ?? undefined,
      },
      config.root_path ?? ""
    );
    await instancesStore.fetchInstances();
  } catch (e) {
    showError(e instanceof Error ? e.message : "Failed to launch");
  }
}

export const browseStore = {
  config: browseConfig,
  currentPath,
  entries,
  breadcrumbs,
  fetchConfig: fetchBrowseConfig,
  navigateTo,
  launchFolder,
};
