@echo off
cd /d "%~dp0"
if not exist ".app.lock" goto launch
set /p OLDPID=<".app.lock"
tasklist /FI "PID eq %OLDPID%" 2>nul | find "%OLDPID%" >nul
if not errorlevel 1 (
    echo Already running.
    timeout /t 2 >nul
    exit /b 0
)
del ".app.lock"
:launch
start "" "%~dp0pythonw_speech2sub.exe" "%~dp0launcher.py"
timeout /t 2 >nul
