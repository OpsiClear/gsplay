/**
 * Console/logs store.
 */

import { createSignal } from "solid-js";
import { api } from "../api/client";

const [logLines, setLogLines] = createSignal<string[]>([]);
const [autoScroll, setAutoScroll] = createSignal(true);
const [consoleStatus, setConsoleStatus] = createSignal<"idle" | "streaming" | "error">("idle");
let logEventSource: EventSource | null = null;
let currentStreamInstanceId: string | null = null;

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
  currentStreamInstanceId = instanceId;
  setConsoleStatus("streaming");

  logEventSource = new EventSource(`/api/instances/${instanceId}/logs/stream`);

  logEventSource.addEventListener("log", (event) => {
    setLogLines((prev) => [...prev, event.data].slice(-1000));
  });

  logEventSource.onerror = () => {
    setConsoleStatus("error");
    setTimeout(() => {
      if (currentStreamInstanceId === instanceId) {
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
  currentStreamInstanceId = null;
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
