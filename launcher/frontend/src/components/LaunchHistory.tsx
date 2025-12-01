/**
 * Launch history component showing recent launches with quick relaunch.
 */

import { Component, For, Show } from "solid-js";
import { historyStore, browseStore, LaunchHistoryEntry } from "../stores/app";

// Format relative time (e.g., "2 min ago", "1 hour ago")
function formatRelativeTime(timestamp: string): string {
  const now = Date.now();
  const then = new Date(timestamp).getTime();
  const diffMs = now - then;

  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return "just now";
}

// Get folder name from path
function getFolderName(path: string): string {
  const parts = path.split("/").filter((p) => p);
  return parts[parts.length - 1] || path;
}

export const LaunchHistory: Component = () => {
  const history = () => historyStore.history();
  const expanded = () => historyStore.expanded();
  const hasHistory = () => history().length > 0;

  const handleNavigate = (entry: LaunchHistoryEntry) => {
    // Navigate to parent directory of the launch path
    const parts = entry.path.split("/").filter((p) => p);
    const parentPath = parts.slice(0, -1).join("/");
    browseStore.navigateTo(parentPath);
  };

  const handleRelaunch = async (entry: LaunchHistoryEntry, e: Event) => {
    e.stopPropagation();
    await historyStore.relaunchFromHistory(entry);
  };

  return (
    <Show when={hasHistory()}>
      <div class="history-section">
        <div class="history-header" onClick={() => historyStore.toggleExpanded()}>
          <span class="history-chevron">{expanded() ? "▼" : "▶"}</span>
          <span class="history-title">Recent Launches</span>
          <span class="history-count">{history().length}</span>
          <Show when={expanded()}>
            <button
              class="history-clear"
              onClick={(e) => {
                e.stopPropagation();
                historyStore.clearHistory();
              }}
            >
              Clear
            </button>
          </Show>
        </div>

        <Show when={expanded()}>
          <div class="history-list">
            <For each={history()}>
              {(entry) => (
                <div class="history-item">
                  <div class="history-item-main">
                    <span class="history-item-icon">📦</span>
                    <span class="history-item-name" title={entry.path}>
                      {getFolderName(entry.path)}
                    </span>
                    <span class="history-item-time">{formatRelativeTime(entry.timestamp)}</span>
                  </div>
                  <div class="history-item-meta">
                    <Show when={entry.gpu !== null}>
                      <span class="history-badge">GPU {entry.gpu}</span>
                    </Show>
                    <Show when={entry.compact}>
                      <span class="history-badge">compact</span>
                    </Show>
                    <Show when={entry.view_only}>
                      <span class="history-badge">view</span>
                    </Show>
                  </div>
                  <div class="history-item-actions">
                    <button
                      class="btn-secondary btn-xs"
                      onClick={(e) => { e.stopPropagation(); handleNavigate(entry); }}
                      title="Go to folder location"
                    >
                      Goto
                    </button>
                    <button
                      class="btn-success btn-xs"
                      onClick={(e) => handleRelaunch(entry, e)}
                      title="Relaunch with same settings"
                    >
                      Relaunch
                    </button>
                  </div>
                </div>
              )}
            </For>
          </div>
        </Show>
      </div>
    </Show>
  );
};
