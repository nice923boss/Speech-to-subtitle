@echo off
cd /d "%~dp0"
echo ============================================
echo   語音轉字幕：安裝依賴
echo ============================================
echo.
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY312%" goto run
for /f "delims=" %%i in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PY312=%%i"
if exist "%PY312%" goto run

echo 找不到 Python 3.12。
where /q winget
if errorlevel 1 goto manual
choice /c YN /m "要用 winget 安裝 Python 3.12 嗎"
if errorlevel 2 goto manual
winget install -e --id Python.Python.3.12 --scope user
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY312%" goto run

:manual
echo 請到 https://www.python.org/downloads/ 安裝 Python 3.12，再執行本檔。
pause
exit /b 1

:run
"%PY312%" install_deps.py
pause
