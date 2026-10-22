@echo off
cd /d "%~dp0"
set PY_HOME=%LOCALAPPDATA%\Programs\Python\Python312

copy /y "%PY_HOME%\pythonw.exe" "pythonw_speech2sub.exe"
copy /y "%PY_HOME%\python312.dll" "python312.dll"
copy /y "%PY_HOME%\python3.dll" "python3.dll"
echo Relinked.
pause
