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

REM Install dependencies on first run, or whenever requirements.txt changes.
REM .deps_installed holds a copy of the last-installed requirements; if it is
REM missing or differs from the current requirements.txt, we (re)install.
if not exist ".deps_installed" goto install
fc /b requirements.txt ".deps_installed" >nul 2>nul
if errorlevel 1 goto install
goto run

:install
echo Installing/updating dependencies...
echo This may take a minute.
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Failed to install dependencies.
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
copy /y requirements.txt ".deps_installed" >nul
echo.
echo Setup complete!
echo.

:run
python run_gui.py
