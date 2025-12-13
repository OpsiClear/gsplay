/**
 * Unified Instances component with stream preview integration.
 */

import { Component, For, Show, createSignal, createEffect, onCleanup } from "solid-js";
import { instancesStore, browseStore, orphanedStore, streamStore } from "../stores/app";
import { Instance } from "../api/client";

const CopyIcon = () => (
  <svg viewBox="0 0 24 24">
    <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" />
  </svg>
);

const RecordIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor">
    <circle cx="12" cy="12" r="8" />
  </svg>
);

const StopRecordIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="1" />
  </svg>
);

function copyLink(url: string, event: MouseEvent): void {
  navigator.clipboard.writeText(url).then(() => {
    const btn = event.target as HTMLElement;
    if (btn) {
      const original = btn.innerHTML;
      btn.textContent = "Copied!";
      btn.style.background = "var(--success)";
      btn.style.color = "white";
      setTimeout(() => {
        btn.innerHTML = original;
        btn.style.background = "";
        btn.style.color = "";
      }, 1000);
    }
  });
}

// Recording state management per instance
const recordingState = new Map<string, {
  mediaRecorder: MediaRecorder;
  chunks: Blob[];
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
}>();

const [recordingInstances, setRecordingInstances] = createSignal<Set<string>>(new Set());

function startRecording(instanceId: string, instanceName: string): void {
  if (recordingState.has(instanceId)) return;

  const canvas = document.createElement("canvas");
  canvas.width = 1920;
  canvas.height = 1080;
  const ctx = canvas.getContext("2d")!;

  const stream = canvas.captureStream(30); // 30 fps
  const mediaRecorder = new MediaRecorder(stream, {
    mimeType: "video/webm;codecs=vp9",
    videoBitsPerSecond: 8000000, // 8 Mbps
  });

  const chunks: Blob[] = [];
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${instanceName}_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, "-")}.webm`;
    a.click();
    URL.revokeObjectURL(url);
    recordingState.delete(instanceId);
    setRecordingInstances((prev) => {
      const next = new Set(prev);
      next.delete(instanceId);
      return next;
    });
  };

  recordingState.set(instanceId, { mediaRecorder, chunks, canvas, ctx });
  mediaRecorder.start(100); // Collect data every 100ms

  setRecordingInstances((prev) => {
    const next = new Set(prev);
    next.add(instanceId);
    return next;
  });
}

function stopRecording(instanceId: string): void {
  const state = recordingState.get(instanceId);
  if (state) {
    state.mediaRecorder.stop();
  }
}

function drawFrameToRecording(instanceId: string, imageUrl: string): void {
  const state = recordingState.get(instanceId);
  if (!state) return;

  const img = new Image();
  img.onload = () => {
    const { canvas, ctx } = state;
    // Scale to fit canvas while maintaining aspect ratio
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const x = (canvas.width - img.width * scale) / 2;
    const y = (canvas.height - img.height * scale) / 2;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
  };
  img.src = imageUrl;
}

const InstanceCard: Component<{ instance: Instance }> = (props) => {
  const config = () => browseStore.config();
  const isStarting = () => ["pending", "starting"].includes(props.instance.status);
  const isReady = () => ["running", "orphaned"].includes(props.instance.status);
  const isActive = () => isStarting() || isReady();
  const isFailed = () => props.instance.status === "failed";
  const isSelected = () => instancesStore.selectedInstanceId() === props.instance.id;

  const openUrl = () =>
    config()?.external_url
      ? `${config()!.external_url}/v/${props.instance.id}/`
      : props.instance.url;

  const hasStream = () => isReady() && props.instance.encoded_stream_path;
  const streamPath = () => props.instance.encoded_stream_path;

  // Full stream viewer URL for opening in new window
  const streamViewerUrl = () => {
    const path = streamPath();
    if (!path) return null;
    // Backend returns full URL when external_url is configured, otherwise just the path
    if (path.startsWith("http")) return path;
    return `${window.location.origin}${path}`;
  };

  // Stream preview state
  const activeStreams = () => streamStore.activeStreams();
  const streamImages = () => streamStore.streamImages();
  const streamResolutions = () => streamStore.streamResolutions();
  const streamErrors = () => streamStore.streamErrors();

  const isStreamActive = () => activeStreams().has(props.instance.id);
  const imageUrl = () => streamImages().get(props.instance.id);
  const resolution = () => streamResolutions().get(props.instance.id);
  const hasError = () => streamErrors().has(props.instance.id);
  const isConnecting = () => isStreamActive() && !imageUrl() && !hasError();
  const isLive = () => isStreamActive() && imageUrl() && !hasError();
  const isRecording = () => recordingInstances().has(props.instance.id);

  // Draw frames to recording canvas when recording
  createEffect(() => {
    const url = imageUrl();
    if (url && isRecording()) {
      drawFrameToRecording(props.instance.id, url);
    }
  });

  // Cleanup recording on unmount
  onCleanup(() => {
    if (recordingState.has(props.instance.id)) {
      stopRecording(props.instance.id);
    }
  });

  return (
    <div
      class={`instance-card ${isSelected() ? "selected" : ""} ${isLive() ? "live" : ""} ${isRecording() ? "recording" : ""}`}
      onClick={() => instancesStore.selectInstance(props.instance.id)}
    >
      {/* Header: name + badge */}
      <div class="instance-row">
        <span class="instance-name">{props.instance.name}</span>
        <Show when={isRecording()}>
          <span class="rec-badge">REC</span>
        </Show>
        <span class={`badge badge-${props.instance.status}`}>{props.instance.status}</span>
      </div>

      {/* Meta: port, GPU, resolution, tags */}
      <div class="instance-row meta">
        <span>:{props.instance.port}</span>
        {props.instance.gpu !== null && <span>GPU {props.instance.gpu}</span>}
        <Show when={resolution()}><span>{resolution()}</span></Show>
        <Show when={props.instance.view_only}><span class="tag cyan">View</span></Show>
        <Show when={props.instance.compact}><span class="tag purple">Compact</span></Show>
      </div>

      {/* Custom IDs: viewer_id and stream_token */}
      <Show when={props.instance.viewer_id || props.instance.stream_token}>
        <div class="instance-row meta">
          <Show when={props.instance.viewer_id}>
            <span class="tag green" title="Viewer path">/v/{props.instance.viewer_id}/</span>
          </Show>
          <Show when={props.instance.stream_token}>
            <span class="tag orange" title="Stream path">/s/{props.instance.stream_token}/</span>
          </Show>
        </div>
      </Show>

      {/* Stream viewport (only when streaming is available) */}
      <Show when={hasStream()}>
        <div class="instance-viewport" onClick={(e) => { e.stopPropagation(); streamStore.toggle(props.instance.id); }}>
          <Show when={isLive()}>
            <img src={imageUrl()} alt={props.instance.name} />
            {/* Recording indicator overlay */}
            <Show when={isRecording()}>
              <div class="rec-indicator">● REC</div>
            </Show>
          </Show>
          <div class={`viewport-overlay ${isLive() ? "hidden" : ""}`}>
            <Show when={isConnecting()}>
              <div class="spinner"></div>
              <span>Connecting...</span>
            </Show>
            <Show when={hasError()}>
              <span class="error-text">Stream unavailable</span>
            </Show>
            <Show when={!isStreamActive() && !hasError()}>
              <span class="play-hint">▶ Preview</span>
            </Show>
          </div>
        </div>
      </Show>

      {/* Actions */}
      <div class="instance-row actions">
        {/* Starting state - show spinner and cancel button */}
        <Show when={isStarting()}>
          <span class="starting-indicator">
            <span class="spinner-small"></span>
            <span>Starting...</span>
          </span>
          <button class="btn-danger btn-sm" onClick={async (e) => {
            e.stopPropagation();
            await instancesStore.stopInstance(props.instance.id);
            instancesStore.deleteInstance(props.instance.id);
          }}>
            Cancel
          </button>
        </Show>
        {/* Ready state - show full controls */}
        <Show when={isReady()}>
          <span class="btn-group">
            <button class="btn-success btn-sm" onClick={(e) => { e.stopPropagation(); window.open(openUrl(), "_blank"); }}>Viewer</button>
            <button class="btn-success btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); copyLink(openUrl(), e); }} title="Copy viewer URL"><CopyIcon /></button>
          </span>
          <Show when={hasStream()}>
            <span class="btn-group">
              <button
                class="btn-secondary btn-sm"
                onClick={(e) => { e.stopPropagation(); window.open(streamViewerUrl()!, "_blank"); }}
                title="Open stream viewer in new window"
              >
                Stream
              </button>
              <button class="btn-secondary btn-sm btn-icon" onClick={(e) => { e.stopPropagation(); copyLink(streamViewerUrl()!, e); }} title="Copy stream URL"><CopyIcon /></button>
            </span>
            {/* Record button - show when stream is active */}
            <Show when={isStreamActive()}>
              <button
                class={`btn-sm ${isRecording() ? "btn-danger recording-btn" : "btn-secondary"}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (isRecording()) {
                    stopRecording(props.instance.id);
                  } else {
                    startRecording(props.instance.id, props.instance.name);
                  }
                }}
                title={isRecording() ? "Stop recording & download" : "Start recording"}
              >
                {isRecording() ? "⏹ Save" : "⏺ Rec"}
              </button>
            </Show>
          </Show>
          <button class="btn-danger btn-sm" onClick={async (e) => {
            e.stopPropagation();
            await instancesStore.stopInstance(props.instance.id);
            instancesStore.deleteInstance(props.instance.id);
          }}>
            Stop
          </button>
        </Show>
        <Show when={!isActive()}>
          <Show when={isFailed()}>
            <button
              class="btn-secondary btn-sm"
              onClick={async (e) => {
                e.stopPropagation();
                await instancesStore.deleteInstance(props.instance.id);
              }}
              title="Remove failed instance"
            >
              Delete
            </button>
          </Show>
        </Show>
      </div>
    </div>
  );
};

export const Instances: Component = () => {
  const instances = () => instancesStore.instances();
  const orphaned = () => orphanedStore.processes();
  const activeCount = () =>
    instances().filter((i) => ["running", "starting"].includes(i.status)).length;
  const streamingCount = () =>
    instances().filter((i) => i.encoded_stream_path && ["running", "starting", "orphaned"].includes(i.status)).length;

  return (
    <div class="card">
      <div class="card-header">
        <span>
          Instances{" "}
          <span style={{ "font-weight": "normal" }}>
            {activeCount()}/{instances().length}
            <Show when={streamingCount() > 0}>
              {" "}• {streamStore.activeStreams().size} streaming
            </Show>
          </span>
        </span>
        <span style={{ display: "flex", gap: "4px", "align-items": "center" }}>
          <Show when={streamingCount() > 0}>
            <button class="btn-secondary btn-sm" onClick={() => streamStore.startAll()}>All On</button>
            <button class="btn-secondary btn-sm" onClick={() => streamStore.stopAll()}>All Off</button>
          </Show>
          <button
            class="btn-secondary btn-sm"
            onClick={() => orphanedStore.scan()}
            disabled={orphanedStore.scanning()}
          >
            {orphanedStore.scanning() ? "..." : "Scan"}
          </button>
        </span>
      </div>
      <div class="card-body">
        <Show when={instances().length === 0}>
          <div class="empty-state">No instances</div>
        </Show>
        <Show when={instances().length > 0}>
          <div class="instances-grid">
            <For each={instances()}>{(instance) => <InstanceCard instance={instance} />}</For>
          </div>
        </Show>

        {/* Orphaned processes */}
        <Show when={orphaned().length > 0}>
          <div class="orphaned-section">
            <div class="orphaned-header">
              <span>Orphaned ({orphaned().length})</span>
              <button class="btn-danger btn-sm" onClick={() => orphanedStore.killAll()}>Kill All</button>
            </div>
            <For each={orphaned()}>
              {(p) => (
                <div class="orphaned-row">
                  <span class="orphan-info">
                    <span class="pid">PID {p.pid}</span>
                    <span>:{p.port || "?"}</span>
                    <span class="dim">{p.memory_mb.toFixed(0)}MB</span>
                  </span>
                  <button class="btn-danger btn-sm" onClick={() => orphanedStore.kill(p.pid)}>Kill</button>
                </div>
              )}
            </For>
          </div>
        </Show>
      </div>
    </div>
  );
};
