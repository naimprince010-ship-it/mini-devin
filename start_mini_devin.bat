@echo off
REM Mini Devin Startup Script
REM This script launches Mini Devin using the installation on D: drive (to avoid C: drive space issues)

REM Fix encoding for Unicode support
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Load .env variables
if exist .env (
    for /f "eol=# tokens=1* delims==" %%a in (.env) do set "%%a=%%b"
)

SET VENV_PYTHON="D:\PoetryCache\virtualenvs\mini-devin-NCmu-bfV-py3.14\Scripts\python.exe"
SET MINI_DEVIN_CMD="D:\PoetryCache\virtualenvs\mini-devin-NCmu-bfV-py3.14\Scripts\mini-devin.cmd"

echo Starting Mini Devin...
echo Python: %VENV_PYTHON%

REM Example: Run a simple task
REM %MINI_DEVIN_CMD% run "echo Hello World" --dir .

REM Forward arguments if provided, otherwise show help
if "%~1"=="" (
    echo For Web Dashboard, run:
    echo %VENV_PYTHON% -m mini_devin.api.app
    echo.
    echo For CLI Help:
    %MINI_DEVIN_CMD% --help
) else (
    %MINI_DEVIN_CMD% %*
)

if errorlevel 1 pause
cmd /k
