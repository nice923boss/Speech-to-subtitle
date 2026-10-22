@echo off
cd /d "%~dp0"
set LOCK=.app.lock

REM 1. Kill by image name (unique exe)
taskkill /f /im pythonw_speech2sub.exe >nul 2>&1

REM 2. Clean lockfile
if exist "%LOCK%" del "%LOCK%" >nul 2>&1

echo Done.
timeout /t 2 >nul
