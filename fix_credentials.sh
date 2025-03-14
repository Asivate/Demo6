#!/bin/bash
# This script fixes the Google Cloud credentials for the SoundWatch server
# It sets the environment variable to point to the credentials file

# Update the path to the correct location of your credentials file
export GOOGLE_APPLICATION_CREDENTIALS="/home/hirwa0250/asivate-452914-5c12101797af.json"

# Print confirmation
echo "Google Cloud credentials set to: $GOOGLE_APPLICATION_CREDENTIALS"

# Verify the file exists
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Credentials file exists. ✓"
else
    echo "ERROR: Credentials file not found at $GOOGLE_APPLICATION_CREDENTIALS"
    exit 1
fi
