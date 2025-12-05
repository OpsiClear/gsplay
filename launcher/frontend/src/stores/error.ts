/**
 * Global error notification utility.
 * Extracted to avoid circular dependencies between stores.
 */

import { createSignal } from "solid-js";

const [error, setError] = createSignal<string | null>(null);

export function showError(message: string): void {
  setError(message);
  setTimeout(() => setError(null), 5000);
}

export function clearError(): void {
  setError(null);
}

export { error };
