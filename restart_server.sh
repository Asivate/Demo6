#!/bin/bash
# Script to restart the SoundWatch server on Debian

echo "Restarting SoundWatch server..."

# Check if we're in the server directory
if [ ! -f "server.py" ]; then
    echo "Error: server.py not found in current directory."
    echo "Please run this script from the server directory."
    exit 1
fi

# Check for Google Cloud credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set."
    CREDS_FILE="$HOME/asivate-452914-9778a9b91269.json"
    if [ -f "$CREDS_FILE" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
        echo "Set GOOGLE_APPLICATION_CREDENTIALS to $CREDS_FILE"
    else
        echo "Could not find Google Cloud credentials file."
        echo "Speech recognition might not work correctly."
    fi
fi

# Check for virtual environment and activate if present
if [ -d "venv" ]; then
    echo "Found virtual environment, activating..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Found virtual environment in parent directory, activating..."
    source ../venv/bin/activate
else
    echo "No virtual environment found, using system Python."
fi

# Start the server with debug flag
echo "Starting server with debug flag..."
python server.py --debug

# Check exit code
if [ $? -ne 0 ]; then
    echo "Error starting server."
    exit 1
fi 