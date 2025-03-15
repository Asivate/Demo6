#!/bin/bash
# Simple script to set up the environment for Google Cloud Speech-to-Text
# on your Debian virtual machine

# Set fixed paths for your VM environment
CREDENTIALS_PATH="/home/hirwa0250/asivate-452914-9778a9b91269.json"
SERVER_DIR="/home/hirwa0250/Sonarity-server"

# Check if credentials file exists
if [ ! -f "$CREDENTIALS_PATH" ]; then
    echo -e "\e[31mError: Credentials file not found at $CREDENTIALS_PATH\e[0m"
    echo -e "\e[33mMake sure the file exists in your home directory\e[0m"
    exit 1
fi

# Set the Google Cloud credentials environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_PATH"
echo -e "\e[32mSet GOOGLE_APPLICATION_CREDENTIALS to: $CREDENTIALS_PATH\e[0m"

# Make sure we're in the server directory
cd "$SERVER_DIR" || { 
    echo -e "\e[31mError: Could not change to server directory: $SERVER_DIR\e[0m"
    exit 1
}

# Start the server with Google Speech-to-Text enabled
echo -e "\e[32mStarting server with Google Cloud Speech-to-Text...\e[0m"
python3 server.py --use-google-speech

# Note: You can save this in your ~/.bashrc to set the environment variable automatically:
# echo 'export GOOGLE_APPLICATION_CREDENTIALS="/home/hirwa0250/asivate-452914-9778a9b91269.json"' >> ~/.bashrc 