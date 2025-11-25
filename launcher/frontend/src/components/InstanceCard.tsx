/**
 * Card component for displaying a single viewer instance.
 */

import { Component, Show } from "solid-js";
import { Instance } from "../api/client";
import { instancesStore } from "../stores/instances";
import { StatusBadge } from "./StatusBadge";

interface InstanceCardProps {
  instance: Instance;
}

export const InstanceCard: Component<InstanceCardProps> = (props) => {
  const isActive = () =>
    props.instance.status === "running" || props.instance.status === "starting";

  const handleStop = async () => {
    await instancesStore.stop(props.instance.id);
  };

  const handleDelete = async () => {
    if (confirm(`Delete instance "${props.instance.name}"?`)) {
      await instancesStore.remove(props.instance.id);
    }
  };

  const handleOpen = () => {
    window.open(props.instance.url, "_blank");
  };

  return (
    <div
      style={{
        background: "#1a1a1a",
        "border-radius": "8px",
        padding: "16px",
        display: "flex",
        "flex-direction": "column",
        gap: "12px",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          "justify-content": "space-between",
          "align-items": "center",
        }}
      >
        <h3 style={{ margin: 0, "font-size": "16px" }}>{props.instance.name}</h3>
        <StatusBadge status={props.instance.status} />
      </div>

      {/* Details */}
      <div style={{ "font-size": "13px", color: "#888" }}>
        <div>
          <strong>Port:</strong> {props.instance.port}
        </div>
        <div style={{ "word-break": "break-all" }}>
          <strong>Config:</strong> {props.instance.config_path}
        </div>
        <Show when={props.instance.gpu !== null}>
          <div>
            <strong>GPU:</strong> {props.instance.gpu}
          </div>
        </Show>
        <Show when={props.instance.pid}>
          <div>
            <strong>PID:</strong> {props.instance.pid}
          </div>
        </Show>
        <Show when={props.instance.error_message}>
          <div style={{ color: "#ff4444" }}>
            <strong>Error:</strong> {props.instance.error_message}
          </div>
        </Show>
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: "8px", "margin-top": "auto" }}>
        <Show when={isActive()}>
          <button
            onClick={handleOpen}
            style={{
              flex: 1,
              padding: "8px 12px",
              background: "#2563eb",
              color: "white",
              border: "none",
              "border-radius": "4px",
              cursor: "pointer",
              "font-size": "13px",
            }}
          >
            Open
          </button>
          <button
            onClick={handleStop}
            style={{
              flex: 1,
              padding: "8px 12px",
              background: "#dc2626",
              color: "white",
              border: "none",
              "border-radius": "4px",
              cursor: "pointer",
              "font-size": "13px",
            }}
          >
            Stop
          </button>
        </Show>
        <Show when={!isActive()}>
          <button
            onClick={handleDelete}
            style={{
              flex: 1,
              padding: "8px 12px",
              background: "#3b3b3b",
              color: "#e0e0e0",
              border: "none",
              "border-radius": "4px",
              cursor: "pointer",
              "font-size": "13px",
            }}
          >
            Delete
          </button>
        </Show>
      </div>
    </div>
  );
};
