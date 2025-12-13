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
    // Preserve selected GPU if it's still valid in the new list
    const currentSelection = selectedGpu();
    const validIndices = data.gpus.map((g) => g.index);
    if (!validIndices.includes(currentSelection) && validIndices.length > 0) {
      // Selected GPU no longer exists, reset to first available
      setSelectedGpu(validIndices[0]);
    }
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
