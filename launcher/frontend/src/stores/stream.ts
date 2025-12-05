/**
 * Stream preview store.
 */

import { createSignal } from "solid-js";
import { Instance } from "../api/client";
import { instancesStore } from "./instances";

const [activeStreams, setActiveStreams] = createSignal<Set<string>>(new Set());
const streamWebSockets = new Map<string, WebSocket>();
const [streamImages, setStreamImages] = createSignal<Map<string, string>>(new Map());
const [streamResolutions, setStreamResolutions] = createSignal<Map<string, string>>(new Map());
const [streamErrors, setStreamErrors] = createSignal<Set<string>>(new Set());
const [streamPanelCollapsed, setStreamPanelCollapsed] = createSignal(false);

function getStreamingInstances(): Instance[] {
  return instancesStore.instances().filter(
    (i) => ["running", "starting", "orphaned"].includes(i.status) && i.encoded_stream_path
  );
}

function startStreamPreview(instanceId: string): void {
  const instance = instancesStore.instances().find((i) => i.id === instanceId);
  if (!instance || !instance.encoded_stream_path) return;

  if (streamWebSockets.has(instanceId)) {
    streamWebSockets.get(instanceId)?.close();
    streamWebSockets.delete(instanceId);
  }

  setStreamErrors((prev) => { const next = new Set(prev); next.delete(instanceId); return next; });
  setActiveStreams((prev) => new Set([...prev, instanceId]));

  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = instance.encoded_stream_path.replace(/^https?:/, wsProtocol).replace(/\/$/, "") + "/ws";
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      const oldUrl = streamImages().get(instanceId);
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      setStreamImages((prev) => new Map(prev).set(instanceId, url));

      const img = new Image();
      img.onload = () => {
        if (img.naturalWidth && img.naturalHeight) {
          setStreamResolutions((prev) => new Map(prev).set(instanceId, `${img.naturalWidth}×${img.naturalHeight}`));
        }
        URL.revokeObjectURL(url);
      };
      const checkUrl = URL.createObjectURL(blob);
      img.src = checkUrl;
    }
  };

  ws.onerror = () => setStreamErrors((prev) => new Set([...prev, instanceId]));
  ws.onclose = () => streamWebSockets.delete(instanceId);
  streamWebSockets.set(instanceId, ws);
}

function stopStreamPreview(instanceId: string): void {
  setActiveStreams((prev) => { const next = new Set(prev); next.delete(instanceId); return next; });
  setStreamErrors((prev) => { const next = new Set(prev); next.delete(instanceId); return next; });

  streamWebSockets.get(instanceId)?.close();
  streamWebSockets.delete(instanceId);

  const url = streamImages().get(instanceId);
  if (url) {
    URL.revokeObjectURL(url);
    setStreamImages((prev) => { const next = new Map(prev); next.delete(instanceId); return next; });
  }
  setStreamResolutions((prev) => { const next = new Map(prev); next.delete(instanceId); return next; });
}

function toggleStreamPreview(instanceId: string): void {
  activeStreams().has(instanceId) ? stopStreamPreview(instanceId) : startStreamPreview(instanceId);
}

function retryStream(instanceId: string): void {
  stopStreamPreview(instanceId);
  setTimeout(() => startStreamPreview(instanceId), 100);
}

function startAllStreams(): void {
  getStreamingInstances().forEach((i) => { if (!activeStreams().has(i.id)) startStreamPreview(i.id); });
}

function stopAllStreams(): void {
  [...activeStreams()].forEach((id) => stopStreamPreview(id));
}

function toggleStreamPanel(): void {
  setStreamPanelCollapsed(!streamPanelCollapsed());
}

export const streamStore = {
  activeStreams,
  streamImages,
  streamResolutions,
  streamErrors,
  panelCollapsed: streamPanelCollapsed,
  getStreamingInstances,
  start: startStreamPreview,
  stop: stopStreamPreview,
  toggle: toggleStreamPreview,
  retry: retryStream,
  startAll: startAllStreams,
  stopAll: stopAllStreams,
  togglePanel: toggleStreamPanel,
};
