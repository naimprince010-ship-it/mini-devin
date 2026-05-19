@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set PYTHONIOENCODING=utf-8
title Plodder Desktop App

echo.
echo  =====================================================
echo    Plodder Desktop App Starter
echo  =====================================================
echo.

echo  [1/3] Checking dependencies...
if not exist frontend\node_modules (
    echo  Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

if not exist .env (
    echo  Creating default .env file...
    copy .env.example .env >nul
)

echo.
echo  [2/3] Starting Desktop App (Electron + FastAPI)...
echo  Please wait a moment while the backend starts in the background...
echo.

cd frontend
call npm run dev:desktop

echo.
echo  [3/3] Desktop App closed. Cleaning up...
pause
