# PowerShell script to set Google Cloud credentials and start the server
# Usage: ./start_with_credentials.ps1 -CredentialsPath "C:\path\to\your\credentials.json" [-UseGoogleSpeech] [-Port 5000] [-Debug]

param (
    [Parameter(Mandatory=$true)]
    [string]$CredentialsPath,
    
    [Parameter(Mandatory=$false)]
    [switch]$UseGoogleSpeech,
    
    [Parameter(Mandatory=$false)]
    [int]$Port = 5000,
    
    [Parameter(Mandatory=$false)]
    [switch]$Debug
)

# Check if credentials file exists
if (-not (Test-Path $CredentialsPath)) {
    Write-Host "Error: Credentials file not found at $CredentialsPath" -ForegroundColor Red
    exit 1
}

# Set the Google Cloud credentials environment variable
$env:GOOGLE_APPLICATION_CREDENTIALS = $CredentialsPath
Write-Host "Set GOOGLE_APPLICATION_CREDENTIALS to: $CredentialsPath" -ForegroundColor Green

# Build the command to start the server
$pythonCmd = "python server.py"

if ($Port -ne 5000) {
    $pythonCmd += " --port $Port"
}

if ($UseGoogleSpeech) {
    $pythonCmd += " --use-google-speech"
    Write-Host "Using Google Cloud Speech-to-Text for transcription" -ForegroundColor Cyan
} else {
    Write-Host "Using Whisper for transcription (default)" -ForegroundColor Cyan
}

if ($Debug) {
    $pythonCmd += " --debug"
    Write-Host "Debug mode enabled" -ForegroundColor Yellow
}

# Print server information
$ip = (Get-NetIPAddress | Where-Object {$_.AddressFamily -eq "IPv4" -and $_.PrefixOrigin -ne "WellKnown"} | Select-Object -First 1).IPAddress
Write-Host "Starting server..." -ForegroundColor Green
Write-Host "Server URL: http://$($ip):$Port" -ForegroundColor Cyan
Write-Host "WebSocket URL: ws://$($ip):$Port" -ForegroundColor Cyan

# Run the command
Write-Host "Executing: $pythonCmd" -ForegroundColor Blue
Invoke-Expression $pythonCmd 