@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Simple launcher for Web UI
REM Usage: run-web.bat [PORT]

set PORT=%~1
if "%PORT%"=="" set PORT=8000

REM Activate local venv if present
if exist .\pilar-venv\Scripts\activate (
  call .\pilar-venv\Scripts\activate
)

set URL=http://localhost:%PORT%
echo Starting Web UI on %URL%

REM Note: Do not auto-open a browser

python main.py --web --no-gui --host 0.0.0.0 --port %PORT%

endlocal
