#!/usr/bin/env python3
"""
Script to restart the SoundWatch server.
"""
import os
import sys
import subprocess
import time

def main():
    """Restart the SoundWatch server."""
    print("Restarting SoundWatch server...")
    
    # Ensure we're in the server directory
    if not os.path.exists('server.py'):
        print("Error: server.py not found in current directory.")
        print("Please run this script from the server directory.")
        return 1
    
    # Check for Google Cloud credentials
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        creds_file = os.path.expanduser('~/asivate-452914-9778a9b91269.json')
        if os.path.exists(creds_file):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
            print(f"Set GOOGLE_APPLICATION_CREDENTIALS to {creds_file}")
        else:
            print("Could not find Google Cloud credentials file.")
            print("Speech recognition might not work correctly.")
    
    # Start the server with debug flag
    try:
        print("Starting server with debug flag...")
        subprocess.run([sys.executable, 'server.py', '--debug'], check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        return 0

if __name__ == '__main__':
    sys.exit(main()) 