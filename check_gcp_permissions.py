#!/usr/bin/env python3
"""
Verify Google Cloud Speech API permissions and scopes.

This script checks if the current Google Cloud credentials have the necessary
permissions to use the Speech-to-Text API. It will report any issues and
suggest solutions.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# ANSI color codes for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def print_colored(color, message):
    """Print a colored message to the terminal."""
    print(f"{color}{message}{NC}")

def check_environment_variable():
    """Check if GOOGLE_APPLICATION_CREDENTIALS environment variable is set."""
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        print_colored(RED, "❌ GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        print("Set it to the path of your service account key JSON file:")
        print("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        return None
    
    print_colored(GREEN, f"✓ GOOGLE_APPLICATION_CREDENTIALS is set to: {credentials_path}")
    
    # Check if the file exists
    if not Path(credentials_path).is_file():
        print_colored(RED, f"❌ Credentials file not found at: {credentials_path}")
        return None
    
    print_colored(GREEN, "✓ Credentials file exists")
    return credentials_path

def check_credentials_format(credentials_path):
    """Check if the credentials file has the correct format."""
    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        
        # Check for required fields
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 
                          'client_email', 'client_id', 'auth_uri', 'token_uri']
        
        missing_fields = [field for field in required_fields if field not in credentials]
        
        if missing_fields:
            print_colored(RED, f"❌ Credentials file is missing required fields: {', '.join(missing_fields)}")
            return None
        
        print_colored(GREEN, "✓ Credentials file has the correct format")
        return credentials
    except json.JSONDecodeError:
        print_colored(RED, "❌ Credentials file is not valid JSON")
        return None
    except Exception as e:
        print_colored(RED, f"❌ Error reading credentials file: {str(e)}")
        return None

def check_speech_api():
    """Check if the Speech API is enabled and accessible."""
    try:
        print_colored(YELLOW, "Testing Speech API access...")
        
        # Try to import the speech module
        try:
            from google.cloud import speech
            print_colored(GREEN, "✓ Successfully imported google.cloud.speech")
        except ImportError as e:
            print_colored(RED, f"❌ Failed to import google.cloud.speech: {str(e)}")
            print("Try running: pip install google-cloud-speech")
            return False
        
        # Try to create a client
        try:
            client = speech.SpeechClient()
            print_colored(GREEN, "✓ Successfully created SpeechClient")
        except Exception as e:
            print_colored(RED, f"❌ Failed to create SpeechClient: {str(e)}")
            return False
        
        # Try a simple API call to verify permissions
        try:
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )
            audio = speech.RecognitionAudio(content=b"")  # Empty audio for testing
            
            print("Making test API call... (this may take a few seconds)")
            response = client.recognize(config=config, audio=audio)
            print_colored(GREEN, "✓ API call successful - permissions are correct")
            return True
        except Exception as e:
            error_str = str(e)
            print_colored(RED, f"❌ Error calling Speech API: {error_str}")
            
            if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in error_str:
                print_colored(YELLOW, "\nPERMISSION ERROR: The service account doesn't have required scopes")
                print("\nTo fix this issue:")
                print("1. Go to the Google Cloud Console: https://console.cloud.google.com/")
                print("2. Navigate to IAM & Admin > Service Accounts")
                print("3. Find your service account and edit its permissions")
                print("4. Add the 'Cloud Speech-to-Text API User' role")
                print("5. Make sure the Speech-to-Text API is enabled in APIs & Services")
                print("\nAlternatively, create a new service account with proper permissions:")
                print("1. Go to Google Cloud Console > IAM & Admin > Service Accounts")
                print("2. Create a new service account")
                print("3. Grant it the 'Cloud Speech-to-Text API User' role")
                print("4. Create a new key (JSON) for this service account")
                print("5. Update your GOOGLE_APPLICATION_CREDENTIALS to point to this new file")
            return False
    except Exception as e:
        print_colored(RED, f"❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main function to check Google Cloud Speech API setup."""
    print_colored(YELLOW, "=== Google Cloud Speech API Permission Check ===")
    
    # Step 1: Check environment variable
    credentials_path = check_environment_variable()
    if not credentials_path:
        return False
    
    # Step 2: Check credentials format
    credentials = check_credentials_format(credentials_path)
    if not credentials:
        return False
    
    # Step 3: Check Speech API
    return check_speech_api()

if __name__ == "__main__":
    success = main()
    if success:
        print_colored(GREEN, "\n✓ Your Google Cloud Speech API setup is correctly configured!")
        sys.exit(0)
    else:
        print_colored(RED, "\n❌ There are issues with your Google Cloud Speech API setup.")
        print("Please fix the issues mentioned above before starting the server.")
        sys.exit(1) 