const { app, BrowserWindow } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

let mainWindow;
let backendProcess;

const BACKEND_URL = 'http://127.0.0.1:8000';
const BACKEND_HEALTH_PATH = '/api/sessions';

function isDevMode() {
  return process.env.NODE_ENV === 'development';
}

function backendHealthUrl() {
  return `${BACKEND_URL}${BACKEND_HEALTH_PATH}`;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function killProcessTree(pid) {
  if (!pid) {
    return;
  }

  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(pid), '/t', '/f'], { shell: true });
    return;
  }

  try {
    process.kill(-pid, 'SIGTERM');
  } catch (_err) {
    try {
      process.kill(pid, 'SIGTERM');
    } catch (_err2) {
      // No-op: process already exited.
    }
  }
}

function checkBackendReady(timeoutMs = 1200) {
  return new Promise((resolve) => {
    const req = http.get(backendHealthUrl(), (res) => {
      // Any non-5xx response indicates backend is reachable.
      resolve(res.statusCode && res.statusCode < 500);
      res.resume();
    });

    req.on('error', () => resolve(false));
    req.setTimeout(timeoutMs, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackendReady(maxAttempts = 30, delayMs = 500) {
  for (let i = 0; i < maxAttempts; i += 1) {
    const ready = await checkBackendReady();
    if (ready) {
      return true;
    }
    await wait(delayMs);
  }
  return false;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "Plodder",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const isDev = isDevMode();

  if (isDev) {
    // In dev mode, load the Vite dev server app route.
    mainWindow.loadURL('http://localhost:5173/app');
  } else {
    // In production, load the app route fallback generated during Vite build.
    mainWindow.loadFile(path.join(__dirname, '../dist/app/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startBackend() {
  if (process.env.PLODDER_SKIP_DESKTOP_BACKEND === '1') {
    console.log('[Desktop] Skipping backend startup (PLODDER_SKIP_DESKTOP_BACKEND=1).');
    return;
  }

  console.log('[Desktop] Starting Python backend...');

  // Navigate one directory up from frontend to the root directory.
  const rootDir = path.join(__dirname, '../../');

  backendProcess = spawn('poetry', ['run', 'plodder', 'serve', '--host', '127.0.0.1', '--port', '8000'], {
    cwd: rootDir,
    shell: true,
    detached: process.platform !== 'win32',
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend] ${data.toString().trim()}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[Backend Error] ${data.toString().trim()}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`[Desktop] Python backend exited with code ${code}`);
  });
}

app.whenReady().then(async () => {
  startBackend();
  createWindow();

  const ready = await waitForBackendReady();
  if (!ready && mainWindow) {
    const html = `
      <html>
        <body style="font-family: sans-serif; margin: 24px;">
          <h2>Backend is still starting...</h2>
          <p>Please wait a few seconds, then reload the app window.</p>
          <p>Health URL: ${backendHealthUrl()}</p>
        </body>
      </html>
    `;
    mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  // Ensure we kill the backend process tree when the app quits.
  if (backendProcess && backendProcess.pid) {
    console.log('[Desktop] Stopping Python backend...');
    killProcessTree(backendProcess.pid);
  }
});
