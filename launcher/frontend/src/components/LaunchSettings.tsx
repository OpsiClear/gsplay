/**
 * Launch settings component with GPU selector and options.
 */

import { Component, For, Show } from "solid-js";
import { gpuStore, launchSettingsStore, browseStore } from "../stores/app";

export const LaunchSettings: Component = () => {
  const gpus = () => gpuStore.gpus();
  const config = () => browseStore.config();

  return (
    <div class="card">
      <div class="card-header">Launch Settings</div>
      <div class="card-body">
        <div class="form-row">
          <div class="form-group">
            <label>GPU</label>
            <select
              class="gpu-select"
              value={gpuStore.selectedGpu()}
              onChange={(e) => gpuStore.setSelectedGpu(parseInt(e.currentTarget.value))}
            >
              <For each={gpus()}>
                {(gpu) => {
                  // Extract short GPU name (e.g., "L40S", "RTX 4090")
                  const shortName = gpu.name
                    .replace(/NVIDIA\s*/gi, "")
                    .replace(/GeForce\s*/gi, "")
                    .replace(/Tesla\s*/gi, "")
                    .replace(/Quadro\s*/gi, "")
                    .trim();
                  const memUsedGb = (gpu.memory_used / 1024).toFixed(1);
                  const memTotalGb = (gpu.memory_total / 1024).toFixed(0);
                  const memPct = Math.round((gpu.memory_used / gpu.memory_total) * 100);
                  return (
                    <option value={gpu.index}>
                      {gpu.index}: {shortName} • {memUsedGb}/{memTotalGb} GB ({memPct}%) • {gpu.temperature}°C
                    </option>
                  );
                }}
              </For>
            </select>
          </div>

          <div class="form-group">
            <label>Instance Name</label>
            <input
              type="text"
              placeholder="Auto"
              style={{ width: "150px" }}
              value={launchSettingsStore.instanceName()}
              onInput={(e) => launchSettingsStore.setInstanceName(e.currentTarget.value)}
            />
          </div>

          <div class="form-group">
            <label>Port</label>
            <input
              type="number"
              placeholder="Auto"
              style={{ width: "80px" }}
              value={launchSettingsStore.port() ?? ""}
              onInput={(e) => {
                const val = e.currentTarget.value;
                launchSettingsStore.setPort(val ? parseInt(val) : null);
              }}
            />
          </div>

          <div class="form-group">
            <label>Options</label>
            <div style={{ display: "flex", gap: "8px" }}>
              <button
                type="button"
                class={`toggle-btn ${launchSettingsStore.compact() ? "active" : ""}`}
                onClick={() => launchSettingsStore.setCompact(!launchSettingsStore.compact())}
              >
                Compact
              </button>
              <Show when={!config()?.view_only}>
                <button
                  type="button"
                  class={`toggle-btn ${launchSettingsStore.viewOnly() ? "active" : ""}`}
                  onClick={() => launchSettingsStore.setViewOnly(!launchSettingsStore.viewOnly())}
                >
                  View Only
                </button>
              </Show>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
