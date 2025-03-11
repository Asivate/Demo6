#!/bin/bash

# Script to set Google Cloud credentials and start the server
# Usage: ./start_with_credentials.sh [-c /path/to/credentials.json] [-g] [-p PORT] [-d]
#   -c: Path to credentials file (default: /home/hirwa0250/asivate-452914-5c12101797af.json)
#   -g: Use Google Speech-to-Text instead of Whisper
#   -p: Port number (default: 5000)
#   -d: Enable debug mode

# Default values
CREDENTIALS_PATH="/home/hirwa0250/asivate-452914-5c12101797af.json"
USE_GOOGLE_SPEECH=0
PORT=5000
DEBUG=0

# Parse command line arguments
while getopts "c:gp:d" opt; do
  case $opt in
    c)
      CREDENTIALS_PATH="$OPTARG"
      ;;
    g)
      USE_GOOGLE_SPEECH=1
      ;;
    p)
      PORT="$OPTARG"
      ;;
    d)
      DEBUG=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if credentials file exists
if [ ! -f "$CREDENTIALS_PATH" ]; then
    echo -e "\e[31mError: Credentials file not found at $CREDENTIALS_PATH\e[0m"
    exit 1
fi

# Set the Google Cloud credentials environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_PATH"
echo -e "\e[32mSet GOOGLE_APPLICATION_CREDENTIALS to: $CREDENTIALS_PATH\e[0m"

# Build the command to start the server
PYTHON_CMD="python3 server.py"

if [ "$PORT" != "5000" ]; then
    PYTHON_CMD="$PYTHON_CMD --port $PORT"
fi

if [ "$USE_GOOGLE_SPEECH" -eq 1 ]; then
    PYTHON_CMD="$PYTHON_CMD --use-google-speech"
    echo -e "\e[36mUsing Google Cloud Speech-to-Text for transcription\e[0m"
else
    echo -e "\e[36mUsing Whisper for transcription (default)\e[0m"
fi

if [ "$DEBUG" -eq 1 ]; then
    PYTHON_CMD="$PYTHON_CMD --debug"
    echo -e "\e[33mDebug mode enabled\e[0m"
fi

# Print server information
IP=$(hostname -I | awk '{print $1}')
echo -e "\e[32mStarting server...\e[0m"
echo -e "\e[36mServer URL: http://$IP:$PORT\e[0m"
echo -e "\e[36mWebSocket URL: ws://$IP:$PORT\e[0m"

# Make the script executable when pulled on Linux
chmod +x "$(dirname "$0")/start_with_credentials.sh"

# Run the command
echo -e "\e[34mExecuting: $PYTHON_CMD\e[0m"
$PYTHON_CMD 