#!/bin/bash

# SoundWatch Server Setup Script
# This script creates a Python 3.6.4 virtual environment and installs required dependencies

echo "==== SoundWatch Server Setup ===="
echo "This script will set up a Python 3.6.4 virtual environment and install dependencies."

# Check if Python 3.6.4 is available
if ! python3.6 --version 2>&1 | grep -q "Python 3.6.4"; then
    echo "Python 3.6.4 is required but not found."
    echo "Please make sure Python 3.6.4 is installed and available in your PATH."
    exit 1
fi

# Create a directory for virtual environments if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment directory..."
    mkdir -p "$VENV_DIR"
fi

# Create a Python 3.6.4 virtual environment
SOUNDWATCH_VENV="$VENV_DIR/soundwatch_env"
echo "Creating Python 3.6.4 virtual environment at $SOUNDWATCH_VENV..."
python3.6 -m venv "$SOUNDWATCH_VENV"

# Activate the virtual environment
echo "Activating virtual environment..."
source "$SOUNDWATCH_VENV/bin/activate"

# Update pip to the latest version
echo "Updating pip..."
pip install --upgrade pip

# Install the dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."

# Check if models directory exists, create if not
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Install specific versions for compatibility with Python 3.6.4
echo "Installing required packages..."
pip install Flask==1.0.2
pip install Flask-Login==0.4.1
pip install Flask-Session==0.3.1
pip install Flask_SocketIO
pip install itsdangerous==1.1.0
pip install Jinja2==2.10
pip install MarkupSafe==1.1.0
pip install python-engineio
pip install python-socketio
pip install six==1.11.0
pip install Werkzeug==0.14.1
pip install gunicorn
pip install eventlet
pip install numpy==1.14.1
pip install tensorflow==1.5.0
pip install keras==2.1.6
pip install wget

echo ""
echo "==== Setup Complete ===="
echo "To activate the virtual environment, run:"
echo "source $SOUNDWATCH_VENV/bin/activate"
echo ""
echo "To start the server, run:"
echo "python server.py"
echo ""
echo "Or for more options:"
echo "python start_server.py --type [main|e2e|timer]" 