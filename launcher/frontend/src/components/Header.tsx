/**
 * Header component with system stats.
 */

import { Component, Show } from "solid-js";
import { systemStore, gpuStore, browseStore } from "../stores/app";

export const Header: Component = () => {
  const stats = () => systemStore.stats();
  const config = () => browseStore.config();

  return (
    <header>
      <div class="header-top">
        <h1>GSPlay</h1>
        <Show when={config()?.external_url}>
          <span class="external-badge">External</span>
        </Show>
      </div>
      <div class="header-stats">
        <Show when={stats()}>
          <span>CPU: {stats()!.cpu_percent.toFixed(0)}%</span>
          <span class="sep">•</span>
          <span>RAM: {stats()!.memory_used_gb}/{stats()!.memory_total_gb} GB</span>
        </Show>
        <Show when={gpuStore.driverVersion()}>
          <span class="sep">•</span>
          <span>Driver: {gpuStore.driverVersion()}</span>
          <span class="sep">•</span>
          <span>CUDA: {gpuStore.cudaVersion()}</span>
        </Show>
      </div>
    </header>
  );
};
