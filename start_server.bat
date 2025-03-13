@echo off
REM Batch script to start the SoundWatch server
REM This script is designed to run on Windows

echo Starting SoundWatch Server...

REM Change to the server directory
cd /d "%~dp0"
echo Changed to server directory

REM Check for existing server processes
echo Checking for existing server processes...
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo Python processes are running. Do you want to stop them? (y/n)
    set /p confirmation=
    if /I "%confirmation%"=="y" (
        taskkill /F /IM python.exe
        echo Stopped existing Python processes
    )
)

REM Activate virtual environment
echo Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Configure environment for TensorFlow
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Install required packages
echo Checking for required packages...

REM Check for flask-socketio
python -c "try: import flask_socketio; print('flask_socketio installed'); exit(0)\nexcept ImportError: print('flask_socketio NOT installed'); exit(1)"
if %ERRORLEVEL% NEQ 0 (
    echo Installing flask-socketio...
    pip install flask-socketio==5.1.1 python-socketio==5.4.0 python-engineio==4.2.1
)

REM Check for google-cloud-speech
python -c "try: import google.cloud.speech; print('google-cloud-speech installed'); exit(0)\nexcept ImportError: print('google-cloud-speech NOT installed'); exit(1)"
if %ERRORLEVEL% NEQ 0 (
    echo Installing google-cloud-speech...
    pip install google-cloud-speech
)

REM Check for other required packages
python -c "try: import numpy, tensorflow, torch, transformers; print('Core packages installed'); exit(0)\nexcept ImportError as e: print(f'Missing package: {e}'); exit(1)"
if %ERRORLEVEL% NEQ 0 (
    echo Installing core packages...
    pip install -r requirements.txt
)

REM Download the model if it doesn't exist
echo Checking for model file...
python download_model.py

if %ERRORLEVEL% NEQ 0 (
    echo Failed to download model file. Please check the error messages above.
    pause
    exit /b 1
)

REM Start the server
echo Starting the SoundWatch server...

REM Set environment variables for the server
set APPLY_SPEECH_BIAS_CORRECTION=True
set SPEECH_BIAS_CORRECTION=0.3

REM Start the server
python server.py --port 8080

if %ERRORLEVEL% NEQ 0 (
    echo Server failed to start with exit code %ERRORLEVEL%
    echo Please check the error messages above for more information.
    pause
    exit /b 1
)

pause 