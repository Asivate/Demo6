# PowerShell script to restart the SoundWatch server

Write-Host "==== Restarting SoundWatch Server ====" -ForegroundColor Cyan
Write-Host "Stopping any existing Python processes..." -ForegroundColor Yellow

# Try to stop existing Python processes gracefully
try {
    $pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($pythonProcesses) {
        $pythonProcesses | ForEach-Object { 
            Write-Host "Stopping process with ID: $($_.Id)" -ForegroundColor Yellow
            Stop-Process -Id $_.Id -Force 
        }
        Start-Sleep -Seconds 2
    } else {
        Write-Host "No Python processes found." -ForegroundColor Green
    }
} catch {
    Write-Host "Error stopping Python processes: $_" -ForegroundColor Red
}

# Check if Google Cloud credentials environment variable is set
if (-not $env:GOOGLE_APPLICATION_CREDENTIALS) {
    # Look for credentials file in user's home directory
    $credentialsFile = Join-Path $env:USERPROFILE "asivate-452914-9778a9b91269.json"
    
    if (Test-Path $credentialsFile) {
        Write-Host "Setting Google Cloud credentials from: $credentialsFile" -ForegroundColor Green
        $env:GOOGLE_APPLICATION_CREDENTIALS = $credentialsFile
    } else {
        Write-Host "Warning: Google Cloud credentials file not found." -ForegroundColor Yellow
        Write-Host "Speech recognition may not work properly." -ForegroundColor Yellow
    }
}

# Start the server
Write-Host "Starting SoundWatch server with debug mode..." -ForegroundColor Green
python server.py --debug

# Keep the window open if there was an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Server exited with code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Press any key to close this window..." 
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 