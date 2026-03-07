@echo off
REM Quick start script cho Windows
REM Chạy: scripts\quickstart.bat

setlocal enabledelayedexpansion

echo.
echo ========================================
echo  DENTAL CAVITY DETECTION SYSTEM
echo  Quick Start Script
echo ========================================
echo.

REM Change to root directory
cd /d "%~dp0\.."

REM Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not installed!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo      %PYVER% OK
echo.

REM Install dependencies
echo [2/5] Checking dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo      Installing packages...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)
echo      All packages OK
echo.

REM Generate synthetic data
echo [3/5] Generating synthetic dataset...
python scripts\train.py --gen-data >nul 2>&1
if errorlevel 1 (
    echo      WARNING: Could not generate synthetic data
    echo      You can do this manually: python scripts/train.py --gen-data
) else (
    echo      50 synthetic X-ray images created
)
echo.

REM Train UNet
echo [4/5] Training UNet model...
echo      (This may take 5-10 minutes on GPU, 30+ on CPU)
timeout /t 3 /nobreak >nul
python scripts\train.py --unet >nul 2>&1
if errorlevel 1 (
    echo      WARNING: Could not train UNet
    echo      Models may not be available for prediction
    echo      You can train manually: python scripts/train.py --unet
) else (
    echo      UNet model trained
)
echo.

REM Start Flask server
echo [5/5] Starting Flask server...
echo.
echo ========================================
echo  SERVER STARTING...
echo  Open: http://localhost:5000
echo  Press Ctrl+C to stop server
echo ========================================
echo.

cd app
python run.py

REM Server is running, script ends here when user stops it
exit /b 0
