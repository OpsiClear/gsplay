/**
 * API client for the launcher backend.
 */

const API_BASE = "/api";

export interface Instance {
  id: string;
  name: string;
  status: "pending" | "starting" | "running" | "stopping" | "stopped" | "failed" | "orphaned";
  port: number;
  url: string;
  config_path: string;
  gpu: number | null;
  pid: number | null;
  created_at: string;
  started_at: string | null;
  stopped_at: string | null;
  error_message: string | null;
}

export interface InstanceList {
  instances: Instance[];
  total: number;
}

export interface CreateInstanceRequest {
  config_path: string;
  name?: string;
  port?: number | null;
  gpu?: number | null;
  cache_size?: number;
  view_only?: boolean;
  compact?: boolean;
  log_level?: string;
}

export interface PortInfo {
  next_available: number | null;
  range_start: number;
  range_end: number;
}

export interface Health {
  status: string;
  instance_count: number;
  active_count: number;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

export const api = {
  health: {
    check: () => request<Health>("/health"),
  },

  instances: {
    list: () => request<InstanceList>("/instances"),

    get: (id: string) => request<Instance>(`/instances/${id}`),

    create: (data: CreateInstanceRequest) =>
      request<Instance>("/instances", {
        method: "POST",
        body: JSON.stringify(data),
      }),

    stop: (id: string) =>
      request<Instance>(`/instances/${id}/stop`, { method: "POST" }),

    delete: (id: string) =>
      request<void>(`/instances/${id}`, { method: "DELETE" }),
  },

  ports: {
    getNext: () => request<PortInfo>("/ports/next"),
  },
};
