/**
 * Global application stores for state management.
 */

import { createSignal, createEffect, onCleanup } from "solid-js";
import {
  api,
  Instance,
  GpuInfo,
  GpuInfoResponse,
  SystemStats,
  BrowseConfig,
  BrowseResponse,
  DirectoryEntry,
  ProcessInfo,
  BrowseLaunchRequest,
} from "../api/client";

// ============================================================================
// Instances Store
// ============================================================================

const [instances, setInstances] = createSignal<Instance[]>([]);
const [instancesLoading, setInstancesLoading] = createSignal(false);
const [selectedInstanceId, setSelectedInstanceId] = createSignal<string | null>(null);

async function fetchInstances(): Promise<void> {
  try {
    const data = await api.instances.list();
    setInstances(data.instances);
  } catch (e) {
    console.error("Failed to fetch instances:", e);
  }
}

async function stopInstance(id: string): Promise<void> {
  try {
    const updated = await api.instances.stop(id);
    setInstances((prev) => prev.map((i) => (i.id === id ? updated : i)));
  } catch (e) {
    appStore.showError(e instanceof Error ? e.message : "Failed to stop instance");
  }
}

async function deleteInstance(id: string): Promise<void> {
  try {
    await api.instances.delete(id);
    setInstances((prev) => prev.filter((i) => i.id !== id));
    if (selectedInstanceId() === id) {
      setSelectedInstanceId(null);
    }
  } catch (e) {
    appStore.showError(e instanceof Error ? e.message : "Failed to delete instance");
  }
}

function selectInstance(id: string): void {
  if (selectedInstanceId() === id) return;
  setSelectedInstanceId(id);
  consoleStore.loadLogs(id);
  consoleStore.startStream(id);
}

export const instancesStore = {
  instances,
  loading: instancesLoading,
  selectedInstanceId,
  fetchInstances,
  stopInstance,
  deleteInstance,
  selectInstance,
  setSelectedInstanceId,
};

// ============================================================================
// GPU Store
// ============================================================================

const [gpus, setGpus] = createSignal<GpuInfo[]>([]);
const [driverVersion, setDriverVersion] = createSignal("");
const [cudaVersion, setCudaVersion] = createSignal("");
const [selectedGpu, setSelectedGpu] = createSignal(0);

async function fetchGpuInfo(): Promise<void> {
  try {
    const data = await api.gpu.getInfo();
    setGpus(data.gpus);
    setDriverVersion(data.driver_version);
    setCudaVersion(data.cuda_version);
  } catch (e) {
    console.error("Failed to fetch GPU info:", e);
  }
}

export const gpuStore = {
  gpus,
  driverVersion,
  cudaVersion,
  selectedGpu,
  setSelectedGpu,
  fetchGpuInfo,
};

// ============================================================================
// System Stats Store
// ============================================================================

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

// ============================================================================
// Browse Store
// ============================================================================

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
    appStore.showError(e instanceof Error ? e.message : "Failed to browse path");
  }
}

async function launchFolder(entry: DirectoryEntry): Promise<void> {
  const config = browseConfig();
  if (!config) return;

  const data: BrowseLaunchRequest = {
    path: entry.path,
    name: launchSettingsStore.instanceName() || entry.name,
    port: launchSettingsStore.port() || undefined,
    stream_port: -1, // Auto-assign
    gpu: gpuStore.selectedGpu(),
    compact: launchSettingsStore.compact(),
    view_only: config.view_only || launchSettingsStore.viewOnly(),
    cache_size: 100,
  };

  try {
    await api.browse.launch(data, !!config.external_url);

    // Record in history after successful launch
    historyStore.addHistoryEntry(
      {
        path: entry.path,
        name: data.name || entry.name,
        gpu: data.gpu ?? null,
        compact: data.compact ?? false,
        view_only: data.view_only ?? false,
      },
      config.root_path ?? ""
    );

    await instancesStore.fetchInstances();
  } catch (e) {
    appStore.showError(e instanceof Error ? e.message : "Failed to launch");
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

// ============================================================================
// Launch Settings Store
// ============================================================================

const [instanceName, setInstanceName] = createSignal("");
const [port, setPort] = createSignal<number | null>(null);
const [compact, setCompact] = createSignal(true);
const [viewOnly, setViewOnly] = createSignal(true);

export const launchSettingsStore = {
  instanceName,
  setInstanceName,
  port,
  setPort,
  compact,
  setCompact,
  viewOnly,
  setViewOnly,
};

// ============================================================================
// History Store
// ============================================================================

export interface LaunchHistoryEntry {
  id: string;
  timestamp: string;
  path: string;
  name: string;
  gpu: number | null;
  compact: boolean;
  view_only: boolean;
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

    // Reset if root path changed
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

function addHistoryEntry(
  entry: Omit<LaunchHistoryEntry, "id" | "timestamp">,
  rootPath: string
): void {
  const newEntry: LaunchHistoryEntry = {
    ...entry,
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
  };

  // Remove existing entry with same path, add new at front, trim to limit
  const updated = [
    newEntry,
    ...launchHistory().filter((e) => e.path !== entry.path),
  ].slice(0, historyLimit());

  setLaunchHistory(updated);

  try {
    localStorage.setItem(
      HISTORY_STORAGE_KEY,
      JSON.stringify({
        rootPath,
        entries: updated,
      } as LaunchHistoryStorage)
    );
  } catch (e) {
    console.error("Failed to save launch history:", e);
  }
}

async function relaunchFromHistory(entry: LaunchHistoryEntry): Promise<void> {
  const config = browseConfig();
  if (!config) return;

  const data: BrowseLaunchRequest = {
    path: entry.path,
    name: entry.name,
    port: undefined, // Auto-assign new port
    stream_port: -1,
    gpu: entry.gpu ?? gpuStore.selectedGpu(),
    compact: entry.compact,
    view_only: config.view_only || entry.view_only,
    cache_size: 100,
  };

  try {
    await api.browse.launch(data, !!config.external_url);
    // Update history timestamp
    addHistoryEntry(
      {
        path: entry.path,
        name: entry.name,
        gpu: data.gpu ?? null,
        compact: entry.compact,
        view_only: entry.view_only,
      },
      config.root_path ?? ""
    );
    await instancesStore.fetchInstances();
  } catch (e) {
    appStore.showError(e instanceof Error ? e.message : "Failed to relaunch");
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

// ============================================================================
// Console Store
// ============================================================================

const [logLines, setLogLines] = createSignal<string[]>([]);
const [autoScroll, setAutoScroll] = createSignal(true);
const [consoleStatus, setConsoleStatus] = createSignal<"idle" | "streaming" | "error">("idle");
let logEventSource: EventSource | null = null;

async function loadLogs(instanceId: string): Promise<void> {
  try {
    const data = await api.instances.getLogs(instanceId, 200);
    setLogLines(data.lines);
  } catch (e) {
    console.error("Failed to load logs:", e);
    setLogLines([]);
  }
}

function startStream(instanceId: string): void {
  stopStream();
  setConsoleStatus("streaming");

  logEventSource = new EventSource(`/api/instances/${instanceId}/logs/stream`);

  logEventSource.addEventListener("log", (event) => {
    setLogLines((prev) => {
      const newLines = [...prev, event.data];
      // Limit to 1000 lines
      return newLines.slice(-1000);
    });
  });

  logEventSource.onerror = () => {
    setConsoleStatus("error");
    // Reconnect after delay
    setTimeout(() => {
      const selected = instancesStore.selectedInstanceId();
      if (selected === instanceId) {
        startStream(instanceId);
      }
    }, 3000);
  };
}

function stopStream(): void {
  if (logEventSource) {
    logEventSource.close();
    logEventSource = null;
  }
}

function clearConsole(): void {
  setLogLines([]);
}

function toggleAutoScroll(): void {
  setAutoScroll(!autoScroll());
}

export const consoleStore = {
  lines: logLines,
  autoScroll,
  status: consoleStatus,
  loadLogs,
  startStream,
  stopStream,
  clearConsole,
  toggleAutoScroll,
};

// ============================================================================
// Orphaned Processes Store
// ============================================================================

const [orphanedProcesses, setOrphanedProcesses] = createSignal<ProcessInfo[]>([]);
const [scanning, setScanning] = createSignal(false);

async function scanProcesses(): Promise<void> {
  setScanning(true);
  try {
    const data = await api.cleanup.scan();
    // Filter out managed PIDs
    const managedPids = new Set(
      instancesStore.instances()
        .filter((i) => i.pid)
        .map((i) => i.pid)
    );
    const orphaned = data.processes.filter((p) => !managedPids.has(p.pid));
    setOrphanedProcesses(orphaned);
  } catch (e) {
    appStore.showError(e instanceof Error ? e.message : "Failed to scan processes");
  } finally {
    setScanning(false);
  }
}

async function killOrphanedProcess(pid: number): Promise<void> {
  try {
    await api.cleanup.kill(pid);
    setOrphanedProcesses((prev) => prev.filter((p) => p.pid !== pid));
    await instancesStore.fetchInstances();
  } catch (e) {
    appStore.showError(e instanceof Error ? e.message : "Failed to kill process");
  }
}

async function killAllOrphaned(): Promise<void> {
  const processes = orphanedProcesses();
  for (const p of processes) {
    try {
      await api.cleanup.kill(p.pid);
    } catch (e) {
      console.error(`Failed to kill PID ${p.pid}:`, e);
    }
  }
  setOrphanedProcesses([]);
  await instancesStore.fetchInstances();
}

export const orphanedStore = {
  processes: orphanedProcesses,
  scanning,
  scan: scanProcesses,
  kill: killOrphanedProcess,
  killAll: killAllOrphaned,
};

// ============================================================================
// Stream Preview Store
// ============================================================================

const [activeStreams, setActiveStreams] = createSignal<Set<string>>(new Set());
const streamWebSockets = new Map<string, WebSocket>();
const [streamImages, setStreamImages] = createSignal<Map<string, string>>(new Map());
const [streamResolutions, setStreamResolutions] = createSignal<Map<string, string>>(new Map());
const [streamErrors, setStreamErrors] = createSignal<Set<string>>(new Set());
const [streamPanelCollapsed, setStreamPanelCollapsed] = createSignal(false);

function getStreamingInstances(): Instance[] {
  return instancesStore.instances().filter(
    (i) => ["running", "starting", "orphaned"].includes(i.status) && i.encoded_stream_path
  );
}

function startStreamPreview(instanceId: string): void {
  const instance = instancesStore.instances().find((i) => i.id === instanceId);
  if (!instance || !instance.encoded_stream_path) return;

  // Close existing
  if (streamWebSockets.has(instanceId)) {
    streamWebSockets.get(instanceId)?.close();
    streamWebSockets.delete(instanceId);
  }

  // Clear error state
  setStreamErrors((prev) => {
    const next = new Set(prev);
    next.delete(instanceId);
    return next;
  });

  setActiveStreams((prev) => new Set([...prev, instanceId]));

  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = instance.encoded_stream_path.replace(/^https?:/, wsProtocol).replace(/\/$/, "") + "/ws";
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log(`Stream WebSocket connected for ${instanceId}`);
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);

      // Cleanup old URL
      const oldUrl = streamImages().get(instanceId);
      if (oldUrl) URL.revokeObjectURL(oldUrl);

      setStreamImages((prev) => new Map(prev).set(instanceId, url));

      // Update resolution from image dimensions
      const img = new Image();
      img.onload = () => {
        if (img.naturalWidth && img.naturalHeight) {
          setStreamResolutions((prev) =>
            new Map(prev).set(instanceId, `${img.naturalWidth}×${img.naturalHeight}`)
          );
        }
        URL.revokeObjectURL(url);
      };
      // Create a separate blob URL for dimension check
      const checkUrl = URL.createObjectURL(blob);
      img.src = checkUrl;
    }
  };

  ws.onerror = () => {
    console.error(`Stream WebSocket error for ${instanceId}`);
    setStreamErrors((prev) => new Set([...prev, instanceId]));
  };

  ws.onclose = () => {
    console.log(`Stream WebSocket closed for ${instanceId}`);
    streamWebSockets.delete(instanceId);
  };

  streamWebSockets.set(instanceId, ws);
}

function stopStreamPreview(instanceId: string): void {
  setActiveStreams((prev) => {
    const next = new Set(prev);
    next.delete(instanceId);
    return next;
  });

  // Clear error state
  setStreamErrors((prev) => {
    const next = new Set(prev);
    next.delete(instanceId);
    return next;
  });

  const ws = streamWebSockets.get(instanceId);
  if (ws) {
    ws.close();
    streamWebSockets.delete(instanceId);
  }

  const url = streamImages().get(instanceId);
  if (url) {
    URL.revokeObjectURL(url);
    setStreamImages((prev) => {
      const next = new Map(prev);
      next.delete(instanceId);
      return next;
    });
  }

  setStreamResolutions((prev) => {
    const next = new Map(prev);
    next.delete(instanceId);
    return next;
  });
}

function toggleStreamPreview(instanceId: string): void {
  if (activeStreams().has(instanceId)) {
    stopStreamPreview(instanceId);
  } else {
    startStreamPreview(instanceId);
  }
}

function retryStream(instanceId: string): void {
  stopStreamPreview(instanceId);
  setTimeout(() => startStreamPreview(instanceId), 100);
}

function startAllStreams(): void {
  getStreamingInstances().forEach((i) => {
    if (!activeStreams().has(i.id)) {
      startStreamPreview(i.id);
    }
  });
}

function stopAllStreams(): void {
  [...activeStreams()].forEach((id) => stopStreamPreview(id));
}

function toggleStreamPanel(): void {
  setStreamPanelCollapsed(!streamPanelCollapsed());
}

export const streamStore = {
  activeStreams,
  streamImages,
  streamResolutions,
  streamErrors,
  panelCollapsed: streamPanelCollapsed,
  getStreamingInstances,
  start: startStreamPreview,
  stop: stopStreamPreview,
  toggle: toggleStreamPreview,
  retry: retryStream,
  startAll: startAllStreams,
  stopAll: stopAllStreams,
  togglePanel: toggleStreamPanel,
};

// ============================================================================
// App Store (Global)
// ============================================================================

const [error, setError] = createSignal<string | null>(null);

function showError(message: string): void {
  setError(message);
  setTimeout(() => setError(null), 5000);
}

function clearError(): void {
  setError(null);
}

// Initialize all stores
async function initialize(): Promise<void> {
  await Promise.all([
    systemStore.fetchStats(),
    gpuStore.fetchGpuInfo(),
    browseStore.fetchConfig(),
    instancesStore.fetchInstances(),
  ]);

  // Initialize history after browse config is loaded
  const config = browseStore.config();
  if (config) {
    historyStore.initHistory(config.root_path, config.history_limit);
  }
}

// Start polling intervals
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
