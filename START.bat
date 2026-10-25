@echo off
rem Launches the poly-chatterbox server. Extra arguments are passed to "poly-chatterbox serve".
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found at .venv. See README.md for setup.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
poly-chatterbox serve --port 8000 %*
pause
