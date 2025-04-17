#!/bin/bash
# This script sets up the environment and runs the server with Google Cloud credentials

# Check if credentials file is provided as argument
if [ "$#" -eq 1 ]; then
    CREDENTIALS_FILE="$1"
    echo "Using provided credentials file: $CREDENTIALS_FILE"
else
    # Look for credentials in common locations
    echo "No credentials file provided as argument, looking for credentials file..."
    
    # Check in current directory
    if [ -f "google-credentials.json" ]; then
        CREDENTIALS_FILE="google-credentials.json"
        echo "Found credentials file in current directory: $CREDENTIALS_FILE"
    elif [ -f "$HOME/google-credentials.json" ]; then
        CREDENTIALS_FILE="$HOME/google-credentials.json"
        echo "Found credentials file in home directory: $CREDENTIALS_FILE"
    else
        echo "No credentials file found. Please provide path to credentials as argument:"
        echo "Usage: ./run_server.sh /path/to/your-credentials.json"
        echo "You can still continue without credentials, but sentiment analysis will be disabled."
        read -p "Continue without credentials? (y/n): " CONTINUE
        if [[ $CONTINUE != "y" && $CONTINUE != "Y" ]]; then
            exit 1
        fi
        CREDENTIALS_FILE=""
    fi
fi

# Set Google Cloud credentials environment variable if we have credentials
if [ ! -z "$CREDENTIALS_FILE" ]; then
    # Check if file exists and is readable
    if [ -f "$CREDENTIALS_FILE" ] && [ -r "$CREDENTIALS_FILE" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_FILE"
        echo "GOOGLE_APPLICATION_CREDENTIALS set to $CREDENTIALS_FILE"
    else
        echo "Warning: Credentials file $CREDENTIALS_FILE does not exist or is not readable."
        echo "Sentiment analysis will be disabled."
    fi
fi

# Check Python version (need 3.7+ for TensorFlow compatibility)
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 7 ]; then
    echo "Warning: Python version $PYTHON_VERSION detected."
    echo "This server works best with Python 3.7 or newer."
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ $CONTINUE != "y" && $CONTINUE != "Y" ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install python3-venv package."
        echo "On Ubuntu: sudo apt install python3-venv"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if not already installed
if [ ! -f ".requirements_installed" ]; then
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Also ensure we have these important packages
    pip install google-cloud-speech==2.0.1 google-cloud-language==2.0.0
    
    # Mark requirements as installed
    touch .requirements_installed
    echo "Requirements installed successfully."
else
    echo "Requirements already installed."
fi

# Run the server
echo "Starting server..."
echo "Sentiment analysis status: "
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "ENABLED - using credentials from $GOOGLE_APPLICATION_CREDENTIALS"
else 
    echo "DISABLED - no credentials provided"
fi

python server.py

# Deactivate virtual environment on exit
deactivate 