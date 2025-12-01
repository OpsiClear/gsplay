import { defineConfig } from "npm:vite@5.4.19";
import solid from "npm:vite-plugin-solid@2.10.2";

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
