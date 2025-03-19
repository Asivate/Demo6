#!/bin/bash

# Script to set up Google Cloud credentials for speech and sentiment analysis

echo "Setting up Google Cloud credentials for SoundWatch..."

# Check if environment variable is already set
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS is already set to: $GOOGLE_APPLICATION_CREDENTIALS"
    
    # Check if the file exists and is readable
    if [ -r "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "Credentials file is readable. Using existing credentials."
        exit 0
    else
        echo "Warning: Existing credentials file is not readable or doesn't exist!"
        echo "Will attempt to use local credentials instead."
    fi
fi

# Check if credential file exists in the current directory
if [ -f "google_cloud_credentials.json" ]; then
    echo "Credentials file found: google_cloud_credentials.json"
else
    echo "Local credentials file 'google_cloud_credentials.json' not found!"
    echo "Please place your Google Cloud credentials JSON file in this directory and name it 'google_cloud_credentials.json'"
    echo "or ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is correctly set."
    exit 1
fi

# Export credentials environment variable if we're using the local file
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/google_cloud_credentials.json
echo "Exported GOOGLE_APPLICATION_CREDENTIALS to: $GOOGLE_APPLICATION_CREDENTIALS"

# Check that the file is readable
if [ -r "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Credentials file is readable."
else
    echo "ERROR: Credentials file is not readable!"
    exit 1
fi

echo ""
echo "Setup complete! To use these credentials in your current shell, run:"
echo "source setup_google_cloud.sh"
echo ""
echo "To enable the APIs for your Google Cloud project, visit:"
echo "https://console.cloud.google.com/apis/library"
echo "and enable the following APIs:"
echo "- Cloud Speech-to-Text API"
echo "- Cloud Natural Language API" 