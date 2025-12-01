/**
 * File browser component with breadcrumbs and file list.
 */

import { Component, For, Show } from "solid-js";
import { browseStore } from "../stores/app";
import { LaunchHistory } from "./LaunchHistory";

export const FileBrowser: Component = () => {
  const config = () => browseStore.config();
  const entries = () => browseStore.entries();
  const breadcrumbs = () => browseStore.breadcrumbs();
  const currentPath = () => browseStore.currentPath();

  const isAtRoot = () => !currentPath();

  const dirs = () => entries().filter((e) => e.is_directory);

  // Filter out root breadcrumb since we show it manually
  const filteredBreadcrumbs = () =>
    breadcrumbs().filter((c) => c.name.toLowerCase() !== "root" && c.path !== "");

  const getParentPath = () => {
    const parts = currentPath().split("/").filter((p) => p);
    return parts.slice(0, -1).join("/");
  };

  return (
    <Show when={config()?.enabled}>
      <div class="card">
        <div class="card-header">File Browser</div>
        <div class="card-body">
          {/* Launch History */}
          <LaunchHistory />

          {/* Breadcrumb */}
          <div class="path-bar">
            <div class="breadcrumb">
              <a onClick={() => browseStore.navigateTo("")}>root</a>
              <For each={filteredBreadcrumbs()}>
                {(crumb) => (
                  <>
                    <span class="sep">/</span>
                    <a onClick={() => browseStore.navigateTo(crumb.path)}>{crumb.name}</a>
                  </>
                )}
              </For>
            </div>
          </div>

          {/* File list */}
          <div class="card-inner">
            <Show when={dirs().length === 0 && isAtRoot()}>
              <div class="empty-state">Empty directory</div>
            </Show>

            {/* Parent directory */}
            <Show when={!isAtRoot()}>
              <div class="file-item" onClick={() => browseStore.navigateTo(getParentPath())}>
                <span class="file-icon">📁</span>
                <span class="file-name">..</span>
                <span class="file-meta"></span>
              </div>
            </Show>

            {/* Directories */}
            <For each={dirs()}>
              {(entry) => {
                const isEmpty = !entry.is_ply_folder && entry.subfolder_count === 0;
                const meta = entry.is_ply_folder
                  ? `${entry.ply_count} PLY, ${entry.total_size_mb.toFixed(1)} MB`
                  : entry.subfolder_count > 0
                  ? `${entry.subfolder_count} folder${entry.subfolder_count > 1 ? "s" : ""}`
                  : "empty";

                return (
                  <div
                    class={`file-item ${entry.is_ply_folder ? "ply-folder" : ""} ${isEmpty ? "disabled" : ""}`}
                    onClick={() => !isEmpty && browseStore.navigateTo(entry.path)}
                  >
                    <span class="file-icon">{entry.is_ply_folder ? "📦" : "📁"}</span>
                    <span class="file-name">{entry.name}</span>
                    <span class="file-meta">{meta}</span>
                    <Show when={entry.is_ply_folder}>
                      <div class="file-actions">
                        <button
                          class="btn-success btn-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            browseStore.launchFolder(entry);
                          }}
                        >
                          Launch
                        </button>
                      </div>
                    </Show>
                  </div>
                );
              }}
            </For>
          </div>
        </div>
      </div>
    </Show>
  );
};
