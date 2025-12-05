/**
 * Instance management store.
 */

import { createSignal } from "solid-js";
import { api, Instance } from "../api/client";
import { showError } from "./error";
import { consoleStore } from "./console";

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
    showError(e instanceof Error ? e.message : "Failed to stop instance");
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
    showError(e instanceof Error ? e.message : "Failed to delete instance");
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
  setSelectedInstanceId,
  setInstances,
  fetchInstances,
  stopInstance,
  deleteInstance,
  selectInstance,
};
