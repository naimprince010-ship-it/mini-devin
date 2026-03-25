@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set PYTHONIOENCODING=utf-8
title Mini-Devin — Starting...

echo.
echo  =====================================================
echo    Mini-Devin  ^|  AI Software Engineer
echo  =====================================================
echo.

REM ── Step 1: Check Python ────────────────────────────────
echo  [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python is not installed on your computer!
    echo.
    echo  Please download and install Python 3.10 or newer:
    echo     https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: During install, check "Add Python to PATH"
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo     Found: %%v

REM ── Step 2: Install / update dependencies ───────────────
echo.
echo  [2/4] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo     Installing dependencies (first time — please wait)...
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo.
        echo  [ERROR] Failed to install dependencies.
        echo  Please check your internet connection and try again.
        pause
        exit /b 1
    )
    echo     Dependencies installed successfully!
) else (
    echo     Dependencies OK!
)

REM ── Step 3: Load .env ───────────────────────────────────
echo.
echo  [3/4] Loading configuration...
if exist .env (
    for /f "usebackq eol=# tokens=1* delims==" %%a in (".env") do (
        if not "%%a"=="" set "%%a=%%b"
    )
    echo     Configuration loaded.
) else (
    echo     No .env file found — using defaults.
    echo.
    echo     TIP: Copy .env.example to .env and add your API key!
)

REM ── Step 4: Start server ─────────────────────────────────
echo.
echo  [4/4] Starting Mini-Devin server...
echo.
echo  =====================================================
echo    Opening in your browser at: http://localhost:8000
echo    Press Ctrl+C to stop the server
echo  =====================================================
echo.

REM Wait 2 seconds then open browser
ping -n 3 127.0.0.1 >nul 2>&1
start "" "http://localhost:8000"

REM Find and use poetry venv if available, otherwise use system python
set "USE_POETRY=0"
poetry env info >nul 2>&1
if not errorlevel 1 set "USE_POETRY=1"

if "!USE_POETRY!"=="1" (
    poetry run python -m uvicorn mini_devin.api.app:app --host 0.0.0.0 --port 8000
) else (
    python -m uvicorn mini_devin.api.app:app --host 0.0.0.0 --port 8000
)

REM If server crashes, pause so user can see error
if errorlevel 1 (
    echo.
    echo  [ERROR] Mini-Devin stopped unexpectedly.
    echo  Please check the error message above.
    pause
)
