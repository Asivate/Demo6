#!/usr/bin/env python
"""
Fix Google Cloud Speech API dependency issues.
This script installs the required packages for Google Cloud Speech API.
"""

import os
import sys
import subprocess
import platform

def main():
    """Install Google Cloud Speech API dependencies."""
    print("Fixing Google Cloud Speech API dependencies...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("Warning: Not running in a virtual environment. It's recommended to run this script in a virtual environment.")
    
    # Install Google Cloud Speech API
    print("Installing google-cloud-speech...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-speech"])
        print("Successfully installed google-cloud-speech")
    except subprocess.CalledProcessError:
        print("Failed to install google-cloud-speech. Please install it manually.")
        return False
    
    # Create a test file to check if the import works
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_google_speech.py")
    with open(test_file, "w") as f:
        f.write("""
try:
    from google.cloud import speech
    print("Successfully imported google.cloud.speech")
    exit(0)
except ImportError as e:
    print(f"Failed to import google.cloud.speech: {e}")
    exit(1)
""")
    
    # Test the import
    print("Testing Google Cloud Speech API import...")
    try:
        result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
        if result.returncode == 0:
            print("Google Cloud Speech API is working correctly!")
            os.remove(test_file)
            return True
        else:
            print(f"Error testing Google Cloud Speech API: {result.stdout}\n{result.stderr}")
            os.remove(test_file)
            return False
    except Exception as e:
        print(f"Error running test: {e}")
        if os.path.exists(test_file):
            os.remove(test_file)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Google Cloud Speech API dependencies fixed successfully!")
        sys.exit(0)
    else:
        print("Failed to fix Google Cloud Speech API dependencies.")
        print("The server will fall back to using Whisper for speech recognition.")
        sys.exit(1) 