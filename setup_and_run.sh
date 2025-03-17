#!/bin/bash

# Script to set up Google Cloud credentials and run the SoundWatch server

echo "=== SoundWatch Server Setup and Run ==="
echo ""

# Set up credentials environment variable
CREDENTIALS_FILE="$HOME/asivate-452914-9778a9b91269.json"

if [ -f "$CREDENTIALS_FILE" ]; then
    echo "✅ Found credentials file at: $CREDENTIALS_FILE"
    export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_FILE"
    echo "✅ Set GOOGLE_APPLICATION_CREDENTIALS environment variable"
else
    echo "❌ Credentials file not found at: $CREDENTIALS_FILE"
    echo "Searching for credentials file..."
    
    # Look for credentials file in common locations
    for file in $(find $HOME -name "*.json" -type f 2>/dev/null); do
        grep -q "service_account" "$file"
        if [ $? -eq 0 ]; then
            echo "✅ Found potential credentials file at: $file"
            export GOOGLE_APPLICATION_CREDENTIALS="$file"
            echo "✅ Set GOOGLE_APPLICATION_CREDENTIALS to: $file"
            break
        fi
    done
fi

# Verify the environment variable is set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ Failed to set GOOGLE_APPLICATION_CREDENTIALS"
    echo "Please set it manually with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json"
else
    echo "GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
fi

echo ""
echo "=== Running SoundWatch Server ==="
echo ""

# Change to the server directory and run the server
cd "$(dirname "$0")/server" || cd server
python server.py --debug

# If Python command fails, try with python3
if [ $? -ne 0 ]; then
    echo "Trying with python3..."
    python3 server.py --debug
fi 