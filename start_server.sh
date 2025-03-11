#!/bin/bash

# Exit on error
set -e

# Configuration
PORT=8080
DEBUG=false
VENV_DIR="venv"
PYTHON_VERSION="3.8.16"
USE_GOOGLE_SPEECH=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --use-google-speech)
      USE_GOOGLE_SPEECH=true
      shift
      ;;
    --help)
      echo "Usage: ./start_server.sh [--port=PORT] [--debug] [--use-google-speech]"
      echo ""
      echo "Options:"
      echo "  --port=PORT         Specify the port to run the server on (default: 8080)"
      echo "  --debug             Run in debug mode"
      echo "  --use-google-speech Use Google Cloud Speech-to-Text instead of Whisper"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Function to install system dependencies
install_system_dependencies() {
    echo "Installing system dependencies..."
    
    # First, remove any existing Python 3.8 installation
    if [ -d "/usr/local/lib/python3.8" ]; then
        echo "Removing existing Python 3.8 installation..."
        sudo rm -rf /usr/local/lib/python3.8
        sudo rm -f /usr/local/bin/python3.8
        sudo rm -f /usr/local/bin/pip3.8
    fi

    # Update package list
    echo "Updating package list..."
    sudo apt-get update

    # Install build dependencies
    echo "Installing build dependencies..."
    sudo apt-get install -y \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        libbz2-dev \
        ffmpeg \
        portaudio19-dev \
        libsndfile1-dev

    # Download Python source
    echo "Downloading Python ${PYTHON_VERSION}..."
    cd /tmp
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz
    tar -xf Python-${PYTHON_VERSION}.tar.xz
    cd Python-${PYTHON_VERSION}

    # Configure and build Python
    echo "Configuring Python build..."
    ./configure \
        --enable-optimizations \
        --prefix=/usr/local \
        --with-ssl \
        --with-openssl=/usr \
        --with-system-ffi \
        LDFLAGS="-Wl,-rpath=/usr/local/lib"

    # Build Python (using 4 cores)
    echo "Building Python (this may take a while)..."
    make -j 4

    # Install Python
    echo "Installing Python..."
    sudo make altinstall

    # Cleanup
    cd /tmp
    rm -rf Python-${PYTHON_VERSION}*

    # Create symlinks
    echo "Creating symlinks..."
    sudo ln -sf /usr/local/bin/python3.8 /usr/local/bin/python3
    sudo ln -sf /usr/local/bin/pip3.8 /usr/local/bin/pip3

    # Install pip
    echo "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    /usr/local/bin/python3.8 get-pip.py --no-warn-script-location
    rm get-pip.py

    # Verify installation
    echo "Verifying Python installation..."
    /usr/local/bin/python3.8 --version
    /usr/local/bin/python3.8 -c "import ssl; print('SSL support available')"
}

# Check if Python 3.8 is installed with SSL support
if ! command -v python3.8 &> /dev/null || ! python3.8 -c "import ssl" &> /dev/null; then
    echo "Python 3.8 with SSL support not found. Installing from source..."
    install_system_dependencies
fi

# Remove existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create new virtual environment with Python 3.8
echo "Creating virtual environment with Python 3.8..."
/usr/local/bin/python3.8 -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify Python version and SSL support
echo "Verifying Python installation..."
python -c "import ssl; print('SSL support available')" || {
    echo "SSL support not available. Installation failed."
    exit 1
}

CURRENT_PYTHON_VERSION=$(python --version)
echo "Using Python version: $CURRENT_PYTHON_VERSION"

# Upgrade pip to latest version
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies with increased timeout
echo "Installing Python dependencies..."
python -m pip install --default-timeout=100 -r requirements.txt

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Check if Google Cloud credentials file exists
if [ "$USE_GOOGLE_SPEECH" = true ]; then
  if [ ! -f "asivate-452914-3c56106e7a07.json" ]; then
    echo "Error: Google Cloud credentials file 'asivate-452914-3c56106e7a07.json' not found."
    echo "Please place the credentials file in the server directory."
    exit 1
  fi
  echo "Using Google Cloud Speech-to-Text for speech recognition"
fi

# Run the server
echo "Starting Sonarity server on port $PORT"
if [ "$DEBUG" = true ]; then
  if [ "$USE_GOOGLE_SPEECH" = true ]; then
    python server.py --port "$PORT" --debug --use-google-speech
  else
    python server.py --port "$PORT" --debug
  fi
else
  if [ "$USE_GOOGLE_SPEECH" = true ]; then
    python server.py --port "$PORT" --use-google-speech
  else
    python server.py --port "$PORT"
  fi
fi 