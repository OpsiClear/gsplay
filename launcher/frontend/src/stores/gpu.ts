/**
 * GPU information store.
 */

import { createSignal } from "solid-js";
import { api, GpuInfo } from "../api/client";

const [gpus, setGpus] = createSignal<GpuInfo[]>([]);
const [driverVersion, setDriverVersion] = createSignal("");
const [cudaVersion, setCudaVersion] = createSignal("");
const [selectedGpu, setSelectedGpu] = createSignal(0);

async function fetchGpuInfo(): Promise<void> {
  try {
    const data = await api.gpu.getInfo();
    setGpus(data.gpus);
    setDriverVersion(data.driver_version);
    setCudaVersion(data.cuda_version);
  } catch (e) {
    console.error("Failed to fetch GPU info:", e);
  }
}

export const gpuStore = {
  gpus,
  driverVersion,
  cudaVersion,
  selectedGpu,
  setSelectedGpu,
  fetchGpuInfo,
};
