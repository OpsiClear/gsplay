import { defineConfig } from "npm:vite@^5.0.0";
import solid from "npm:vite-plugin-solid@^2.8.0";

export default defineConfig({
  plugins: [solid()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    target: "esnext",
  },
});
