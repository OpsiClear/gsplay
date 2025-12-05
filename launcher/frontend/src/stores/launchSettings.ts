/**
 * Launch settings store.
 */

import { createSignal } from "solid-js";

const [instanceName, setInstanceName] = createSignal("");
const [port, setPort] = createSignal<number | null>(null);
const [compact, setCompact] = createSignal(true);
const [viewOnly, setViewOnly] = createSignal(true);
const [viewerId, setViewerId] = createSignal("");
const [streamToken, setStreamToken] = createSignal("");

export const launchSettingsStore = {
  instanceName,
  setInstanceName,
  port,
  setPort,
  compact,
  setCompact,
  viewOnly,
  setViewOnly,
  viewerId,
  setViewerId,
  streamToken,
  setStreamToken,
};
