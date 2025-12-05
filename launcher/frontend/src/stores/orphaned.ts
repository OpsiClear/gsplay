/**
 * Orphaned processes store.
 */

import { createSignal } from "solid-js";
import { api, ProcessInfo } from "../api/client";
import { showError } from "./error";
import { instancesStore } from "./instances";

const [orphanedProcesses, setOrphanedProcesses] = createSignal<ProcessInfo[]>([]);
const [scanning, setScanning] = createSignal(false);

async function scanProcesses(): Promise<void> {
  setScanning(true);
  try {
    const data = await api.cleanup.scan();
    const managedPids = new Set(
      instancesStore.instances().filter((i) => i.pid).map((i) => i.pid)
    );
    const orphaned = data.processes.filter((p) => !managedPids.has(p.pid));
    setOrphanedProcesses(orphaned);
  } catch (e) {
    showError(e instanceof Error ? e.message : "Failed to scan processes");
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
    showError(e instanceof Error ? e.message : "Failed to kill process");
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
