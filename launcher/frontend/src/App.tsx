/**
 * Main application component.
 */

import { Component, createSignal, Show } from "solid-js";
import { instancesStore } from "./stores/instances";
import { InstanceList } from "./components/InstanceList";
import { CreateInstanceForm } from "./components/CreateInstanceForm";

const App: Component = () => {
  const [showForm, setShowForm] = createSignal(false);

  const handleCreated = () => {
    setShowForm(false);
    instancesStore.fetchAll();
  };

  return (
    <div
      style={{
        "max-width": "1200px",
        margin: "0 auto",
        padding: "24px",
      }}
    >
      {/* Header */}
      <header
        style={{
          display: "flex",
          "justify-content": "space-between",
          "align-items": "center",
          "margin-bottom": "24px",
        }}
      >
        <div>
          <h1 style={{ margin: 0, "font-size": "24px" }}>Viewer Launcher</h1>
          <p style={{ margin: "4px 0 0 0", color: "#888", "font-size": "14px" }}>
            Manage Gaussian Splatting viewer instances
          </p>
        </div>
        <button
          onClick={() => setShowForm(!showForm())}
          style={{
            padding: "10px 20px",
            background: showForm() ? "#3b3b3b" : "#2563eb",
            color: "white",
            border: "none",
            "border-radius": "6px",
            cursor: "pointer",
            "font-size": "14px",
            "font-weight": "500",
          }}
        >
          {showForm() ? "Cancel" : "+ New Instance"}
        </button>
      </header>

      {/* Error banner */}
      <Show when={instancesStore.error()}>
        <div
          style={{
            background: "#3d0000",
            color: "#ff4444",
            padding: "12px 16px",
            "border-radius": "6px",
            "margin-bottom": "16px",
            display: "flex",
            "justify-content": "space-between",
            "align-items": "center",
          }}
        >
          <span>{instancesStore.error()}</span>
          <button
            onClick={() => instancesStore.clearError()}
            style={{
              background: "transparent",
              border: "none",
              color: "#ff4444",
              cursor: "pointer",
              "font-size": "18px",
            }}
          >
            x
          </button>
        </div>
      </Show>

      {/* Create form */}
      <Show when={showForm()}>
        <CreateInstanceForm
          onCreated={handleCreated}
          onCancel={() => setShowForm(false)}
        />
      </Show>

      {/* Instance list */}
      <InstanceList />
    </div>
  );
};

export default App;
