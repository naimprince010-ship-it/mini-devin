const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "Plodder",
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  const isDev = process.env.NODE_ENV === 'development';

  if (isDev) {
    // In dev mode, load the Vite dev server
    mainWindow.loadURL('http://localhost:5173');
    // Open dev tools automatically if desired
    // mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built static files
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startBackend() {
  console.log("Starting Python backend...");
  
  // Navigate one directory up from frontend to the root directory
  const rootDir = path.join(__dirname, '../../');
  
  // Determine command based on whether we use poetry or bare python
  // For simplicity, we use the standard uvicorn launch matching the existing batch script
  pythonProcess = spawn('poetry', ['run', 'uvicorn', 'mini_devin.api:app', '--host', '127.0.0.1', '--port', '8000'], {
    cwd: rootDir,
    shell: true,
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Backend]: ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Backend Error]: ${data.toString().trim()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python backend exited with code ${code}`);
  });
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

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
  // Ensure we kill the backend process when the app quits
  if (pythonProcess) {
    console.log("Killing Python backend...");
    pythonProcess.kill();
  }
});
