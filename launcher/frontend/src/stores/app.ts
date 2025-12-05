/**
 * Application-level store and orchestration.
 * Re-exports all stores for convenient imports.
 */

// Re-export all stores
export { instancesStore } from "./instances";
export { gpuStore } from "./gpu";
export { systemStore } from "./system";
export { browseStore } from "./browse";
export { launchSettingsStore } from "./launchSettings";
export { historyStore, type LaunchHistoryEntry } from "./history";
export { consoleStore } from "./console";
export { orphanedStore } from "./orphaned";
export { streamStore } from "./stream";
export { error, showError, clearError } from "./error";

// Import for initialization
import { instancesStore } from "./instances";
import { gpuStore } from "./gpu";
import { systemStore } from "./system";
import { browseStore } from "./browse";
import { historyStore } from "./history";
import { error, showError, clearError } from "./error";

// ============================================================================
// App Store (Initialization and polling)
// ============================================================================

async function initialize(): Promise<void> {
  await Promise.all([
    systemStore.fetchStats(),
    gpuStore.fetchGpuInfo(),
    browseStore.fetchConfig(),
    instancesStore.fetchInstances(),
  ]);

  const config = browseStore.config();
  if (config) {
    historyStore.initHistory(config.root_path, config.history_limit);
  }
}

function startPolling(): () => void {
  const instancesInterval = setInterval(() => instancesStore.fetchInstances(), 5000);
  const systemInterval = setInterval(() => systemStore.fetchStats(), 3000);
  const gpuInterval = setInterval(() => gpuStore.fetchGpuInfo(), 10000);

  return () => {
    clearInterval(instancesInterval);
    clearInterval(systemInterval);
    clearInterval(gpuInterval);
  };
}

export const appStore = {
  error,
  showError,
  clearError,
  initialize,
  startPolling,
};
