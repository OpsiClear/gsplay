/**
 * SolidJS store for instance state management.
 */

import { createSignal } from "solid-js";
import { api, Instance, CreateInstanceRequest } from "../api/client";

// State signals
const [instances, setInstances] = createSignal<Instance[]>([]);
const [loading, setLoading] = createSignal(false);
const [error, setError] = createSignal<string | null>(null);

// Actions
async function fetchAll(): Promise<void> {
  setLoading(true);
  setError(null);
  try {
    const data = await api.instances.list();
    setInstances(data.instances);
  } catch (e) {
    setError(e instanceof Error ? e.message : "Failed to fetch instances");
  } finally {
    setLoading(false);
  }
}

async function create(data: CreateInstanceRequest): Promise<Instance | null> {
  setLoading(true);
  setError(null);
  try {
    const instance = await api.instances.create(data);
    setInstances((prev) => [...prev, instance]);
    return instance;
  } catch (e) {
    setError(e instanceof Error ? e.message : "Failed to create instance");
    return null;
  } finally {
    setLoading(false);
  }
}

async function stop(id: string): Promise<void> {
  setError(null);
  try {
    const updated = await api.instances.stop(id);
    setInstances((prev) =>
      prev.map((i) => (i.id === id ? updated : i))
    );
  } catch (e) {
    setError(e instanceof Error ? e.message : "Failed to stop instance");
  }
}

async function remove(id: string): Promise<void> {
  setError(null);
  try {
    await api.instances.delete(id);
    setInstances((prev) => prev.filter((i) => i.id !== id));
  } catch (e) {
    setError(e instanceof Error ? e.message : "Failed to delete instance");
  }
}

function clearError(): void {
  setError(null);
}

export const instancesStore = {
  // Reactive getters
  instances,
  loading,
  error,

  // Actions
  fetchAll,
  create,
  stop,
  remove,
  clearError,
};
