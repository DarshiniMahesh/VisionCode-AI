@echo off
echo.
echo ╔══════════════════════════════════════╗
echo ║        NEXUS AI — Quick Start        ║
echo ╚══════════════════════════════════════╝
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause & exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip -q
pip install flask flask-cors groq Pillow numpy requests -q

if not exist models mkdir models

echo.
echo Starting NEXUS AI server...
echo Open http://localhost:5000 in your browser
echo.

python app.py
pause
