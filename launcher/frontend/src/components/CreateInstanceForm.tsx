/**
 * Form component for creating new viewer instances.
 */

import { Component, createSignal } from "solid-js";
import { instancesStore } from "../stores/instances";

interface CreateInstanceFormProps {
  onCreated?: () => void;
  onCancel?: () => void;
}

export const CreateInstanceForm: Component<CreateInstanceFormProps> = (props) => {
  const [configPath, setConfigPath] = createSignal("");
  const [name, setName] = createSignal("");
  const [port, setPort] = createSignal<string>("");
  const [gpu, setGpu] = createSignal<string>("");
  const [cacheSize, setCacheSize] = createSignal("100");
  const [viewOnly, setViewOnly] = createSignal(false);
  const [compact, setCompact] = createSignal(false);
  const [submitting, setSubmitting] = createSignal(false);

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    if (!configPath().trim()) return;

    setSubmitting(true);
    try {
      const result = await instancesStore.create({
        config_path: configPath().trim(),
        name: name().trim() || undefined,
        port: port().trim() ? parseInt(port()) : undefined,
        gpu: gpu().trim() ? parseInt(gpu()) : undefined,
        cache_size: parseInt(cacheSize()) || 100,
        view_only: viewOnly(),
        compact: compact(),
      });

      if (result) {
        // Reset form
        setConfigPath("");
        setName("");
        setPort("");
        setGpu("");
        setCacheSize("100");
        setViewOnly(false);
        setCompact(false);
        props.onCreated?.();
      }
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle = {
    width: "100%",
    padding: "10px 12px",
    background: "#2a2a2a",
    border: "1px solid #3b3b3b",
    "border-radius": "4px",
    color: "#e0e0e0",
    "font-size": "14px",
  };

  const labelStyle = {
    display: "block",
    "margin-bottom": "6px",
    "font-size": "13px",
    color: "#a0a0a0",
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        background: "#1a1a1a",
        "border-radius": "8px",
        padding: "20px",
        "margin-bottom": "24px",
      }}
    >
      <h3 style={{ margin: "0 0 16px 0", "font-size": "16px" }}>
        Launch New Viewer
      </h3>

      <div style={{ display: "grid", gap: "16px" }}>
        {/* Config Path */}
        <div>
          <label style={labelStyle}>Config Path (PLY folder or JSON) *</label>
          <input
            type="text"
            value={configPath()}
            onInput={(e) => setConfigPath(e.currentTarget.value)}
            placeholder="C:/path/to/ply/folder"
            style={inputStyle}
            required
          />
        </div>

        {/* Name and Port row */}
        <div style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "16px" }}>
          <div>
            <label style={labelStyle}>Name (optional)</label>
            <input
              type="text"
              value={name()}
              onInput={(e) => setName(e.currentTarget.value)}
              placeholder="My Viewer"
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>Port (auto if empty)</label>
            <input
              type="number"
              value={port()}
              onInput={(e) => setPort(e.currentTarget.value)}
              placeholder="6020"
              min="1024"
              max="65535"
              style={inputStyle}
            />
          </div>
        </div>

        {/* GPU and Cache Size row */}
        <div style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "16px" }}>
          <div>
            <label style={labelStyle}>GPU Device (optional)</label>
            <input
              type="number"
              value={gpu()}
              onInput={(e) => setGpu(e.currentTarget.value)}
              placeholder="0"
              min="0"
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>Cache Size</label>
            <input
              type="number"
              value={cacheSize()}
              onInput={(e) => setCacheSize(e.currentTarget.value)}
              min="1"
              max="1000"
              style={inputStyle}
            />
          </div>
        </div>

        {/* Checkboxes */}
        <div style={{ display: "flex", gap: "24px" }}>
          <label style={{ display: "flex", "align-items": "center", gap: "8px", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={viewOnly()}
              onChange={(e) => setViewOnly(e.currentTarget.checked)}
            />
            <span style={{ "font-size": "13px" }}>View Only</span>
          </label>
          <label style={{ display: "flex", "align-items": "center", gap: "8px", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={compact()}
              onChange={(e) => setCompact(e.currentTarget.checked)}
            />
            <span style={{ "font-size": "13px" }}>Compact UI</span>
          </label>
        </div>

        {/* Buttons */}
        <div style={{ display: "flex", gap: "12px", "justify-content": "flex-end" }}>
          <button
            type="button"
            onClick={props.onCancel}
            style={{
              padding: "10px 20px",
              background: "#3b3b3b",
              color: "#e0e0e0",
              border: "none",
              "border-radius": "4px",
              cursor: "pointer",
              "font-size": "14px",
            }}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting() || !configPath().trim()}
            style={{
              padding: "10px 20px",
              background: submitting() || !configPath().trim() ? "#2a2a2a" : "#2563eb",
              color: submitting() || !configPath().trim() ? "#666" : "white",
              border: "none",
              "border-radius": "4px",
              cursor: submitting() || !configPath().trim() ? "not-allowed" : "pointer",
              "font-size": "14px",
            }}
          >
            {submitting() ? "Launching..." : "Launch"}
          </button>
        </div>
      </div>
    </form>
  );
};
