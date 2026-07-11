@echo off
REM LoRA Dataset Studio - launch the UI (run setup.bat once first)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo [ERROR] .venv not found - run setup.bat first.
    exit /b 1
)
.venv\Scripts\python.exe app.py
