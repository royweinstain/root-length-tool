@echo off
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".deps_installed" (
    echo First-time setup: installing dependencies...
    echo This may take a minute.
    echo.
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo Failed to install dependencies.
        echo Try running: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo. > .deps_installed
    echo.
    echo Setup complete!
    echo.
)

python run_gui.py
