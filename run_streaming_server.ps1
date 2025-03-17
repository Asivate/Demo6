# PowerShell script to set up and run the SoundWatch server with streaming speech recognition
# This script checks for Google Cloud credentials and runs the server with appropriate settings

# Display welcome message
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  SoundWatch Server - Streaming Speech Recognition   " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Google Cloud credentials file exists
$credentialsFile = Join-Path $env:USERPROFILE "asivate-452914-9778a9b91269.json"
$foundCredentials = $false

if (Test-Path $credentialsFile) {
    Write-Host "✅ Google Cloud credentials file found at: $credentialsFile" -ForegroundColor Green
    # Set environment variable for Google Cloud credentials
    $env:GOOGLE_APPLICATION_CREDENTIALS = $credentialsFile
    $foundCredentials = $true
}
else {
    Write-Host "❌ Google Cloud credentials file not found at: $credentialsFile" -ForegroundColor Yellow
    
    # Try to find any .json files in the home directory
    $jsonFiles = Get-ChildItem -Path $env:USERPROFILE -Filter "*.json"
    
    if ($jsonFiles.Count -gt 0) {
        Write-Host "Found these .json files in your home directory:" -ForegroundColor Yellow
        
        foreach ($file in $jsonFiles) {
            # Check if file content looks like a service account key
            $content = Get-Content -Path $file.FullName -Raw
            if ($content -match '"type": "service_account"') {
                Write-Host "✅ Found potential service account key: $($file.FullName)" -ForegroundColor Green
                $env:GOOGLE_APPLICATION_CREDENTIALS = $file.FullName
                $foundCredentials = $true
                break
            }
            else {
                Write-Host "   - $($file.Name) (not a service account key)" -ForegroundColor Gray
            }
        }
    }
}

# If still no credentials, provide instructions
if (-not $foundCredentials) {
    Write-Host "❌ No Google Cloud service account credentials found!" -ForegroundColor Red
    Write-Host "Please place your credentials file in your home directory: $env:USERPROFILE" -ForegroundColor Yellow
    Write-Host "Or set the GOOGLE_APPLICATION_CREDENTIALS environment variable manually:" -ForegroundColor Yellow
    Write-Host '$env:GOOGLE_APPLICATION_CREDENTIALS = "path\to\your\credentials.json"' -ForegroundColor Yellow
    
    # Ask if user wants to continue anyway
    $continue = Read-Host "Do you want to continue without credentials? (y/n)"
    if ($continue -ne "y") {
        Write-Host "Exiting..." -ForegroundColor Red
        exit 1
    }
}

# Check if we're in the server directory
$currentDir = Get-Location
$serverPyPath = Join-Path $currentDir "server.py"

if (-not (Test-Path $serverPyPath)) {
    # Try to find the server directory
    if (Test-Path (Join-Path $currentDir "server")) {
        Set-Location (Join-Path $currentDir "server")
    }
    elseif (Test-Path (Join-Path $currentDir ".." "server")) {
        Set-Location (Join-Path $currentDir ".." "server")
    }
    else {
        Write-Host "❌ Could not find the server directory. Please run this script from the SoundWatch root directory or the server directory." -ForegroundColor Red
        exit 1
    }
}

# Display the current directory
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan

# Check if virtual environment exists and activate it
if (Test-Path "venv") {
    Write-Host "✅ Virtual environment found, activating..." -ForegroundColor Green
    & .\venv\Scripts\Activate.ps1
}
else {
    Write-Host "ℹ️ No virtual environment found, using system Python" -ForegroundColor Yellow
}

# Run the server with debug flag
Write-Host "Starting SoundWatch server with streaming recognition..." -ForegroundColor Green
try {
    # Try with python command first
    python server.py --debug
}
catch {
    # If python command fails, try with python3
    Write-Host "Trying with python3 command..." -ForegroundColor Yellow
    python3 server.py --debug
}

# Keep console window open if there's an error
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 