@echo off
rem ============================================================
rem  Fast Eddie's Video Splitter - one-time setup
rem  Run this ONCE on a new machine:
rem    1) checks that Python is installed
rem    2) installs all required Python packages
rem    3) downloads the ONNX model
rem    4) checks that FFmpeg is available
rem  Afterwards, launch the app with RunSplitter.bat
rem ============================================================
setlocal
cd /d "%~dp0"

echo(
echo === Fast Eddie's Video Splitter - setup ===
echo(

rem ---- 1) Locate a usable Python --------------------------------
set "PYCMD="
if exist ".venv\Scripts\python.exe" set "PYCMD=.venv\Scripts\python.exe"
if not defined PYCMD py -3.11 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.11"
if not defined PYCMD py -3 -c "import sys" >nul 2>&1 && set "PYCMD=py -3"
if not defined PYCMD python -c "import sys" >nul 2>&1 && set "PYCMD=python"

if not defined PYCMD (
    echo [ERROR] Python was not found on this machine.
    echo(
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo During install, tick "Add python.exe to PATH", then run setup.bat again.
    echo(
    pause
    exit /b 1
)

echo [1/4] Using Python: %PYCMD%
%PYCMD% --version
echo(

rem ---- 2) Install Python packages -------------------------------
echo [2/4] Installing required packages from requirements.txt ...
%PYCMD% -m pip install --upgrade pip
%PYCMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo(
    echo [ERROR] Package installation failed. See messages above.
    pause
    exit /b 1
)
echo(

rem ---- 3) Download the ONNX model -------------------------------
echo [3/4] Downloading / verifying the ONNX model ...
%PYCMD% -c "import video_splitter_ai as v; p=v.default_model_path(); v.ensure_model_downloaded(p, v.MODEL_URL_DEFAULT, v.MODEL_SHA256_DEFAULT, print); print('      Model ready at', p)"
if errorlevel 1 (
    echo(
    echo [ERROR] Model download failed. Check your internet connection and retry.
    pause
    exit /b 1
)
echo(

rem ---- 4) Check FFmpeg ------------------------------------------
echo [4/4] Checking for FFmpeg ...
set "FFOK="
if exist "ffmpeg.exe" if exist "ffprobe.exe" set "FFOK=1"
if not defined FFOK where ffmpeg >nul 2>&1 && where ffprobe >nul 2>&1 && set "FFOK=1"
if defined FFOK (
    echo       FFmpeg found.
) else (
    echo       [WARNING] ffmpeg.exe / ffprobe.exe were not found.
    echo       Place ffmpeg.exe and ffprobe.exe in this folder, OR install FFmpeg
    echo       and add it to your PATH. The app cannot split video without them.
)
echo(

echo === Setup complete. ===
echo Launch the app by running RunSplitter.bat
echo(
pause
