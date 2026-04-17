import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

/** Shared dev + preview: API on :8000; production build under /app uses /app/api (rewrite to /api). */
const apiProxy: Record<string, object> = {
  '/app/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (p: string) => p.replace(/^\/app\/api/, '/api'),
  },
  '/api/ws': {
    target: 'ws://localhost:8000',
    ws: true,
    changeOrigin: true,
  },
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
  '/ws': {
    target: 'ws://localhost:8000',
    ws: true,
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [
    react(),
    // After build, copy index.html into sub-route folders so static CDNs can serve them directly.
    // This fixes 404 on /app when deployed to DigitalOcean / Netlify / any static host.
    {
      name: 'spa-static-route-fallback',
      closeBundle() {
        const dist = path.resolve(__dirname, 'dist')
        const indexHtml = path.join(dist, 'index.html')
        if (fs.existsSync(indexHtml)) {
          const spaRoutes = ['app']
          for (const route of spaRoutes) {
            const dir = path.join(dist, route)
            fs.mkdirSync(dir, { recursive: true })
            fs.copyFileSync(indexHtml, path.join(dir, 'index.html'))
          }
        }
      },
    },
  ],
  server: {
    port: 5173,
    strictPort: false,
    // Cursor / Simple Browser + port forwarding: listen on all interfaces
    host: true,
    open: '/app',
    proxy: apiProxy,
  },
  preview: {
    port: 4173,
    strictPort: false,
    host: true,
    open: '/app',
    proxy: apiProxy,
  },
})
