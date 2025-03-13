# SoundWatch Server Starter Script for Windows
# Optimized for 1-second, 16KHz audio processing as per SoundWatch research

# Set execution policy for this process only
$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Function to check if a process is running
function Is-ProcessRunning {
    param ($ProcessName)
    return (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue).Count -gt 0
}

# Function to activate the virtual environment
function Activate-VirtualEnv {
    if (Test-Path "venv") {
        Write-Host "Activating virtual environment..." -ForegroundColor Green
        & .\venv\Scripts\Activate.ps1
    } else {
        Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        & .\venv\Scripts\Activate.ps1
    }
}

# Function to install required packages
function Install-RequiredPackages {
    Write-Host "Checking required packages..." -ForegroundColor Green
    
    # Core dependencies for SoundWatch
    $requiredPackages = @(
        "flask-socketio",
        "numpy",
        "tensorflow",
        "scipy",
        "torch",
        "transformers",
        "wget",
        "matplotlib"
    )
    
    # Audio and speech processing packages
    $speechPackages = @(
        "google-cloud-speech",
        "soundfile"
    )
    
    # Install core dependencies
    foreach ($package in $requiredPackages) {
        Write-Host "Checking $package..." -ForegroundColor Cyan
        python -c "import importlib.util; print('OK' if importlib.util.find_spec('$package') else 'MISSING')" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing $package..." -ForegroundColor Yellow
            pip install $package
        }
    }
    
    # Try to install speech packages, but don't fail if they're not available
    foreach ($package in $speechPackages) {
        Write-Host "Checking optional package $package..." -ForegroundColor Cyan
        python -c "import importlib.util; print('OK' if importlib.util.find_spec('$package') else 'MISSING')" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing optional package $package..." -ForegroundColor Yellow
            pip install $package -q
        }
    }
}

# Main script
Write-Host "Starting SoundWatch Server (Optimized for 1-second, 16KHz audio processing)" -ForegroundColor Magenta
Write-Host "=====================================================================" -ForegroundColor Magenta

# Check if python is already running
if (Is-ProcessRunning -ProcessName "python") {
    Write-Host "WARNING: Python processes already running. These might interfere with the server." -ForegroundColor Yellow
    $choice = Read-Host "Do you want to stop these processes before continuing? (y/n)"
    if ($choice -eq "y") {
        Write-Host "Stopping Python processes..." -ForegroundColor Yellow
        Stop-Process -Name "python" -Force
    }
}

# Activate virtual environment
Activate-VirtualEnv

# Set TensorFlow logging level
$env:TF_CPP_MIN_LOG_LEVEL = "2"  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR

# Set speech bias correction to true
$env:SPEECH_BIAS = "1"

# Configure environment for SoundWatch as per research specifications
Write-Host "Configuring environment for SoundWatch 1-second, 16KHz audio model..." -ForegroundColor Green
$env:USE_AST_MODEL = "0"  # Disable AST model, use VGG model as per research

# Install required packages
Install-RequiredPackages

# Start the server
Write-Host "Starting SoundWatch server..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server"
Write-Host "=====================================================================" -ForegroundColor Magenta

# Run the server with optimized settings
python server.py --port 8080 