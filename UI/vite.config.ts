import { fileURLToPath, URL } from "node:url"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig, loadEnv } from "vite"

export default defineConfig(({ mode }) => {
  const { VEKRYX_API_URL } = loadEnv(mode, process.cwd(), "VEKRYX_")
  const proxy = {
    "/v1": { target: VEKRYX_API_URL || "http://127.0.0.1:8080" },
  }

  return {
    plugins: [react(), tailwindcss()],
    resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
    server: { host: "127.0.0.1", proxy },
    preview: { host: "127.0.0.1", proxy },
  }
})
