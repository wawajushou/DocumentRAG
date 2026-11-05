@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Please run the setup steps in USAGE.md first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
docrag %*

if "%~1"=="" pause
