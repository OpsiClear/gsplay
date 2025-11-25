/**
 * Grid list component for displaying all instances.
 */

import { Component, For, Show, onMount, onCleanup } from "solid-js";
import { instancesStore } from "../stores/instances";
import { InstanceCard } from "./InstanceCard";

export const InstanceList: Component = () => {
  let interval: number | undefined;

  onMount(() => {
    // Initial fetch
    instancesStore.fetchAll();

    // Poll every 5 seconds
    interval = setInterval(() => {
      instancesStore.fetchAll();
    }, 5000) as unknown as number;
  });

  onCleanup(() => {
    if (interval) {
      clearInterval(interval);
    }
  });

  return (
    <div>
      <Show when={instancesStore.loading() && instancesStore.instances().length === 0}>
        <div style={{ "text-align": "center", padding: "40px", color: "#888" }}>
          Loading...
        </div>
      </Show>

      <Show when={instancesStore.instances().length === 0 && !instancesStore.loading()}>
        <div
          style={{
            "text-align": "center",
            padding: "40px",
            color: "#888",
            background: "#1a1a1a",
            "border-radius": "8px",
          }}
        >
          No viewer instances. Create one to get started.
        </div>
      </Show>

      <div
        style={{
          display: "grid",
          "grid-template-columns": "repeat(auto-fill, minmax(300px, 1fr))",
          gap: "16px",
        }}
      >
        <For each={instancesStore.instances()}>
          {(instance) => <InstanceCard instance={instance} />}
        </For>
      </div>
    </div>
  );
};
