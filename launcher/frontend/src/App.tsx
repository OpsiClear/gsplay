/**
 * Main application component.
 */

import { Component, onMount, onCleanup, Show } from "solid-js";
import { appStore } from "./stores/app";
import { Header } from "./components/Header";
import { LaunchSettings } from "./components/LaunchSettings";
import { FileBrowser } from "./components/FileBrowser";
import { Instances } from "./components/Instances";
import { Console } from "./components/Console";

const App: Component = () => {
  let stopPolling: (() => void) | undefined;

  onMount(async () => {
    await appStore.initialize();
    stopPolling = appStore.startPolling();
  });

  onCleanup(() => {
    stopPolling?.();
  });

  return (
    <>
      <Header />

      <main>
        {/* Error banner */}
        <Show when={appStore.error()}>
          <div class="error-banner">
            <span>{appStore.error()}</span>
            <button
              onClick={() => appStore.clearError()}
              style={{
                background: "transparent",
                border: "none",
                color: "#f87171",
                cursor: "pointer",
                "font-size": "16px",
                padding: "0 4px",
              }}
            >
              x
            </button>
          </div>
        </Show>

        {/* Launch Settings */}
        <LaunchSettings />

        {/* File Browser */}
        <FileBrowser />

        {/* Instances (with integrated stream preview) */}
        <Instances />

        {/* Console */}
        <Console />
      </main>
    </>
  );
};

export default App;
