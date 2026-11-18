import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// host: true lets you also open the dev UI from a phone/another machine on the same network.
export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173 },
});
