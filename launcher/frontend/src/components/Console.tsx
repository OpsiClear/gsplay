/**
 * Console component with log streaming.
 */

import { Component, For, Show, createEffect, onCleanup } from "solid-js";
import { consoleStore, instancesStore } from "../stores/app";

export const Console: Component = () => {
  let contentRef: HTMLDivElement | undefined;

  const lines = () => consoleStore.lines();
  const status = () => consoleStore.status();
  const autoScroll = () => consoleStore.autoScroll();
  const selectedId = () => instancesStore.selectedInstanceId();

  const selectedInstance = () =>
    instancesStore.instances().find((i) => i.id === selectedId());

  // Auto-scroll when new lines added
  createEffect(() => {
    const _ = lines().length;
    if (autoScroll() && contentRef) {
      contentRef.scrollTop = contentRef.scrollHeight;
    }
  });

  // Cleanup on unmount
  onCleanup(() => {
    consoleStore.stopStream();
  });

  const getLineClass = (line: string): string => {
    const lower = line.toLowerCase();
    if (lower.includes("error") || lower.includes("exception") || lower.includes("traceback")) {
      return "log-line error";
    }
    if (lower.includes("warning") || lower.includes("warn")) {
      return "log-line warning";
    }
    if (lower.includes("info")) {
      return "log-line info";
    }
    return "log-line";
  };

  return (
    <div class="card">
      <div class="card-header">
        <span>
          <span class={`console-status ${status()}`}></span>
          Console{" "}
          <span class="console-instance-name">
            {selectedInstance() ? `— ${selectedInstance()!.name}` : "— Select an instance"}
          </span>
        </span>
        <span style={{ display: "flex", gap: "6px" }}>
          <button class="btn-secondary btn-sm" onClick={() => consoleStore.clearConsole()}>
            Clear
          </button>
          <button class="btn-secondary btn-sm" onClick={() => consoleStore.toggleAutoScroll()}>
            Auto-scroll: {autoScroll() ? "ON" : "OFF"}
          </button>
        </span>
      </div>
      <div class="card-body">
        <div ref={contentRef} class="card-inner compact console-content">
          <Show when={lines().length === 0}>
            <div class="empty-state" style={{ padding: "12px", "text-align": "center" }}>
              {selectedId() ? "No logs yet" : "Click on an instance row to view its console output"}
            </div>
          </Show>
          <For each={lines()}>{(line) => <div class={getLineClass(line)}>{line}</div>}</For>
        </div>
      </div>
    </div>
  );
};
