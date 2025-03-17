#!/bin/bash
# Script to restart the SoundWatch server with fixed Google Speech streaming implementation

echo "========================================================"
echo "Restarting SoundWatch server with streaming transcription fixes"
echo "========================================================"

# Check for Google Cloud credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set."
    CREDS_FILE="$HOME/asivate-452914-9778a9b91269.json"
    if [ -f "$CREDS_FILE" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
        echo "Set GOOGLE_APPLICATION_CREDENTIALS to $CREDS_FILE"
    else
        echo "ERROR: Could not find Google Cloud credentials file."
        echo "Speech recognition will not work without credentials."
        echo "Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable"
        echo "or place the credentials file in your home directory."
        exit 1
    fi
fi

# Kill any existing server process
if pgrep -f "python.*server.py" > /dev/null; then
    echo "Stopping existing server processes..."
    pkill -f "python.*server.py"
    sleep 2
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment at venv/"
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating virtual environment at ../venv/"
    source ../venv/bin/activate
fi

# Check if server.py exists in current directory
if [ ! -f "server.py" ]; then
    echo "Error: server.py not found in current directory"
    exit 1
fi

# Start the server
echo "Starting server with debug flag..."
python server.py --debug

# Check if server started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start server. Check logs for details."
    exit 1
fi 