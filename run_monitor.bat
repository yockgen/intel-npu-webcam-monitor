@echo off
set "PROJECT_DIR=%~dp0"
set "PYTHONW=%PROJECT_DIR%.venv\Scripts\pythonw.exe"

if not exist "%PYTHONW%" (
    echo Virtual environment not found. Follow the setup steps in README.md first.
    exit /b 1
)

if exist "%PROJECT_DIR%.stop_monitor" del /q "%PROJECT_DIR%.stop_monitor"
start "" /D "%PROJECT_DIR%" "%PYTHONW%" "%PROJECT_DIR%npu.py"