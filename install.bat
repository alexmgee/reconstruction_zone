@echo off
echo ============================================
echo  Reconstruction Zone -- Setup
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo.
    echo Install Python 3.10-3.12 from https://www.python.org/downloads/
    echo Make sure "Add Python to PATH" is checked during install.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Running installer...
echo.
.venv\Scripts\python setup_install.py
pause
