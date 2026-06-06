@echo off
rem Fast Eddie's Video Splitter - launcher
rem Opens the GUI using the correct Python from the local environment.
rem Dependencies (onnxruntime, PySide6, ...) are installed under Python 3.11,
rem so a local .venv or Python 3.11 is preferred over the system default.
setlocal
cd /d "%~dp0"
set "GUI=fast_eddies_video_split_gui.py"
set "PYCMD="

rem 1) Local virtual environment, if present
if exist ".venv\Scripts\python.exe" set "PYCMD=.venv\Scripts\python.exe"

rem 2) Python 3.11 via the launcher (where the deps are installed)
if not defined PYCMD py -3.11 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.11"

rem 3) Any Python 3 via the launcher
if not defined PYCMD py -3 -c "import sys" >nul 2>&1 && set "PYCMD=py -3"

rem 4) Plain python on PATH
if not defined PYCMD set "PYCMD=python"

%PYCMD% "%GUI%" %*

if errorlevel 1 (
    echo.
    echo The app exited with an error. See messages above.
    pause
)
