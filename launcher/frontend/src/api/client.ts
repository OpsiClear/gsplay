/**
 * API client for the launcher backend.
 */

const API_BASE = "/api";

// Instance types
export interface Instance {
  id: string;
  name: string;
  status: "pending" | "starting" | "running" | "stopping" | "stopped" | "failed" | "orphaned";
  port: number;
  url: string;
  stream_port: number;
  stream_url: string | null;
  encoded_stream_path: string | null;
  config_path: string;
  gpu: number | null;
  view_only: boolean;
  compact: boolean;
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
  view_only?: boolean;
  compact?: boolean;
  log_level?: string;
}

// GPU types
export interface GpuInfo {
  index: number;
  name: string;
  memory_used: number;
  memory_total: number;
  utilization: number;
  temperature: number;
}

export interface GpuInfoResponse {
  gpus: GpuInfo[];
  driver_version: string;
  cuda_version: string;
}

// System types
export interface SystemStats {
  cpu_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  memory_percent: number;
}

// Browse types
export interface BrowseConfig {
  enabled: boolean;
  root_path: string | null;
  default_custom_ip: string | null;
  external_url: string | null;
  view_only: boolean;
  history_limit: number;
}

export interface DirectoryEntry {
  name: string;
  path: string;
  is_directory: boolean;
  is_ply_folder: boolean;
  ply_count: number;
  total_size_mb: number;
  subfolder_count: number;
  modified_at: string | null;
}

export interface BrowseResponse {
  current_path: string;
  breadcrumbs: { name: string; path: string }[];
  entries: DirectoryEntry[];
  browse_enabled: boolean;
}

export interface BrowseLaunchRequest {
  path: string;
  name?: string;
  port?: number | null;
  stream_port?: number;
  gpu?: number | null;
  view_only?: boolean;
  compact?: boolean;
  custom_ip?: string | null;
}

// Port types
export interface PortInfo {
  next_available: number | null;
  range_start: number;
  range_end: number;
}

// Health types
export interface Health {
  status: string;
  instance_count: number;
  active_count: number;
}

// Cleanup types
export interface ProcessInfo {
  pid: number;
  port: number | null;
  config_path: string | null;
  memory_mb: number;
  status: string;
}

export interface CleanupResponse {
  processes: ProcessInfo[];
  total: number;
}

// Log types
export interface LogResponse {
  lines: string[];
  total_lines: number;
  offset: number;
  has_more: boolean;
}

// Control types
export interface ControlResponse {
  ok: boolean;
  centroid?: [number, number, number];
  translation_applied?: [number, number, number];
  filtered_count?: number;
  total_count?: number;
  error?: string;
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

    getLogs: (id: string, lines: number = 200) =>
      request<LogResponse>(`/instances/${id}/logs?lines=${lines}`),
  },

  gpu: {
    getInfo: () => request<GpuInfoResponse>("/gpu"),
  },

  system: {
    getStats: () => request<SystemStats>("/system"),
  },

  browse: {
    getConfig: () => request<BrowseConfig>("/browse/config"),

    list: (path: string = "") =>
      request<BrowseResponse>(`/browse?path=${encodeURIComponent(path)}`),

    launch: (data: BrowseLaunchRequest, useExternalUrl: boolean = false) => {
      const endpoint = useExternalUrl ? "/browse/launch?external_url=true" : "/browse/launch";
      return request<Instance>(endpoint, {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
  },

  cleanup: {
    scan: () => request<CleanupResponse>("/cleanup"),

    kill: (pid: number) =>
      request<void>("/cleanup/stop", {
        method: "POST",
        body: JSON.stringify({ pid, force: true }),
      }),
  },

  ports: {
    getNext: () => request<PortInfo>("/ports/next"),
  },

  control: {
    centerScene: (instanceId: string) =>
      request<ControlResponse>(`/instances/${instanceId}/control/center-scene`, {
        method: "POST",
      }),

    setTranslation: (instanceId: string, x: number, y: number, z: number) =>
      request<ControlResponse>(`/instances/${instanceId}/control/set-translation`, {
        method: "POST",
        body: JSON.stringify({ x, y, z }),
      }),
  },
};
