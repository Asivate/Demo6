#!/bin/bash
# This script helps diagnose and fix Google Cloud Speech API permission issues

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if credentials file exists
CREDENTIALS_FILE="/home/hirwa0250/asivate-452914-5c12101797af.json"

echo -e "${YELLOW}Checking Google Cloud Speech API permissions...${NC}"

# 1. Check if credentials file exists
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo -e "${RED}Error: Credentials file not found at $CREDENTIALS_FILE${NC}"
    echo "Please make sure the file exists and is accessible."
    exit 1
fi

echo -e "${GREEN}✓ Credentials file exists${NC}"

# 2. Set the environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_FILE"
echo -e "${GREEN}✓ Set GOOGLE_APPLICATION_CREDENTIALS environment variable${NC}"

# 3. Run a simple test to check API access
echo -e "${YELLOW}Testing Speech API access...${NC}"
python3 - << EOF
import os
import sys
try:
    from google.cloud import speech
    client = speech.SpeechClient()
    print("Successfully created Speech client")
    
    # Try a simple API call to verify permissions
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    audio = speech.RecognitionAudio(content=b"")  # Empty audio for permission test
    
    try:
        response = client.recognize(config=config, audio=audio)
        print("API call successful - permissions are correct")
        sys.exit(0)
    except Exception as e:
        if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(e):
            print("Error: Insufficient permissions for Speech API")
            print("The service account doesn't have the required scopes")
            sys.exit(2)
        else:
            print(f"Error calling Speech API: {e}")
            sys.exit(1)
except ImportError as e:
    print(f"Error importing Speech API: {e}")
    sys.exit(1)
EOF

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Google Cloud Speech API permissions are correctly configured${NC}"
elif [ $EXIT_CODE -eq 2 ]; then
    echo -e "${RED}× The service account lacks necessary permissions${NC}"
    echo -e "${YELLOW}To fix this issue:${NC}"
    echo "1. Go to the Google Cloud Console: https://console.cloud.google.com/"
    echo "2. Navigate to IAM & Admin > Service Accounts"
    echo "3. Find your service account and edit its permissions"
    echo "4. Add the 'Cloud Speech-to-Text API User' role"
    echo "   or ensure it has the 'https://www.googleapis.com/auth/cloud-platform' scope"
    echo ""
    echo "Alternatively, you can create a new service account with the proper permissions:"
    echo "1. Go to Google Cloud Console > APIs & Services > Credentials"
    echo "2. Create a new service account with the 'Cloud Speech-to-Text API User' role"
    echo "3. Download the JSON key file"
    echo "4. Update the GOOGLE_APPLICATION_CREDENTIALS path to point to the new file"
else
    echo -e "${RED}× Error testing Google Cloud Speech API${NC}"
    echo "Please check the error message above for details."
fi

exit $EXIT_CODE 