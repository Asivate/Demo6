# PowerShell script to start the SoundWatch server
# This script is designed to run on Windows with PowerShell

# Set execution policy for this script only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Function to check if a process is running
function Is-ProcessRunning {
    param (
        [string]$ProcessName
    )
    
    $process = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    return ($null -ne $process)
}

# Function to activate virtual environment
function Activate-VirtualEnv {
    param (
        [string]$VenvPath
    )
    
    if (Test-Path "$VenvPath\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..."
        & "$VenvPath\Scripts\Activate.ps1"
        return $true
    } else {
        Write-Host "Virtual environment not found at $VenvPath"
        return $false
    }
}

# Function to install required packages
function Install-RequiredPackages {
    Write-Host "Checking for required packages..."
    
    # Check for flask-socketio
    python -c "try: import flask_socketio; print('flask_socketio installed'); exit(0)`nexcept ImportError: print('flask_socketio NOT installed'); exit(1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing flask-socketio..."
        pip install flask-socketio==5.1.1 python-socketio==5.4.0 python-engineio==4.2.1
    }
    
    # Check for google-cloud-speech
    python -c "try: import google.cloud.speech; print('google-cloud-speech installed'); exit(0)`nexcept ImportError: print('google-cloud-speech NOT installed'); exit(1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing google-cloud-speech..."
        pip install google-cloud-speech
    }
    
    # Check for other required packages
    python -c "try: import numpy, tensorflow, torch, transformers; print('Core packages installed'); exit(0)`nexcept ImportError as e: print(f'Missing package: {e}'); exit(1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing core packages..."
        pip install -r requirements.txt
    }
}

# Main script
Write-Host "Starting SoundWatch Server..."

# Change to the server directory
$serverDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $serverDir
Write-Host "Changed to server directory"

# Check for existing server processes
Write-Host "Checking for existing server processes..."
if (Is-ProcessRunning "python") {
    $confirmation = Read-Host "Python processes are running. Do you want to stop them? (y/n)"
    if ($confirmation -eq 'y') {
        Stop-Process -Name "python" -Force
        Write-Host "Stopped existing Python processes"
    }
}

# Activate virtual environment
$venvPath = ".\venv"
if (-not (Activate-VirtualEnv $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
    Activate-VirtualEnv $venvPath
}

# Configure environment for TensorFlow
$env:TF_CPP_MIN_LOG_LEVEL = "2"  # Reduce TensorFlow logging
$env:PYTHONPATH = "$serverDir;$env:PYTHONPATH"  # Add server directory to Python path

# Install required packages
Install-RequiredPackages

# Download the model if it doesn't exist
Write-Host "Checking for model file..."
python download_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to download model file. Please check the error messages above."
    exit 1
}

# Start the server
Write-Host "Starting the SoundWatch server..."
try {
    # Set environment variables for the server
    $env:APPLY_SPEECH_BIAS_CORRECTION = "True"  # Enable speech bias correction
    $env:SPEECH_BIAS_CORRECTION = "0.3"  # Set speech bias correction amount
    
    # Start the server
    python server.py --port 8080
} catch {
    Write-Host "Server failed to start with error: $_"
    Write-Host "Please check the error messages above for more information."
    exit 1
} 