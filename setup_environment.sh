#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}SoundWatch Server Environment Setup Script${NC}"
echo -e "${YELLOW}This script will set up a virtual environment and install all dependencies for Python 3.11.2${NC}"

# Check if Python 3.11 is installed
if command -v python3.11 &>/dev/null; then
    echo -e "${GREEN}Python 3.11 found.${NC}"
    PYTHON_CMD="python3.11"
elif command -v python3 &>/dev/null; then
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    if [[ $PYTHON_VERSION == 3.11* ]]; then
        echo -e "${GREEN}Python $PYTHON_VERSION found.${NC}"
        PYTHON_CMD="python3"
    else
        echo -e "${YELLOW}Python $PYTHON_VERSION found, but Python 3.11 is recommended.${NC}"
        echo -e "${YELLOW}Will proceed with $PYTHON_VERSION, but some packages might not be compatible.${NC}"
        PYTHON_CMD="python3"
    fi
else
    echo -e "${RED}Error: Python 3.11 not found. Please install Python 3.11.${NC}"
    echo "You can install it with: sudo apt-get update && sudo apt-get install python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

# Create directory for models if it doesn't exist
if [ ! -d "models" ]; then
    echo -e "${GREEN}Creating models directory...${NC}"
    mkdir -p models
fi

# Create a virtual environment
VENV_NAME=".venv"
echo -e "${GREEN}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv $VENV_NAME

# Activate the virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source $VENV_NAME/bin/activate

# Update pip
echo -e "${GREEN}Updating pip...${NC}"
pip install --upgrade pip

# Create updated requirements file
echo -e "${GREEN}Creating updated requirements.txt with Python 3.11 compatible versions...${NC}"

cat > requirements_updated.txt << EOF
Flask>=2.0.0,<3.0.0
Flask-Login>=0.6.0
Flask-Session>=0.4.0
Flask_SocketIO>=5.3.0
itsdangerous>=2.0.0
Jinja2>=3.0.0
MarkupSafe>=2.0.0
python-engineio>=4.3.0
python-socketio>=5.7.0
six>=1.16.0
Werkzeug>=2.0.0
gunicorn>=20.1.0
eventlet>=0.33.0
numpy>=1.23.0
tensorflow>=2.9.0
keras>=2.9.0
wget>=3.2
pygame>=2.1.0
pillow>=9.0.0
h5py>=3.7.0
EOF

echo -e "${YELLOW}Notice: The following significant changes were made:${NC}"
echo -e "  - TensorFlow updated from 1.5.0 to >=2.9.0 (required for Python 3.11)"
echo -e "  - Keras updated from 2.1.6 to >=2.9.0 (compatible with TensorFlow 2.x)"
echo -e "  - NumPy updated from 1.14.1 to >=1.23.0 (required for Python 3.11)"
echo -e "  - Most Flask dependencies updated to latest compatible versions"
echo -e "  - Added h5py for model loading compatibility"

echo -e "${GREEN}Installing dependencies...${NC}"
pip install -r requirements_updated.txt

echo -e "${YELLOW}Note: TensorFlow API has changed significantly between 1.x and 2.x versions.${NC}"
echo -e "${YELLOW}The following code modifications might be needed in your server files:${NC}"
echo -e "  1. Replace 'tf.get_default_graph()' with '@tf.function' decorators or tf.compat.v1 API"
echo -e "  2. Update model loading and prediction syntax for Keras/TensorFlow 2.x"
echo -e "  3. Update NumPy 'fromstring' method to 'frombuffer' (fromstring is deprecated)"

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}To activate this environment, run:${NC}"
echo -e "source $VENV_NAME/bin/activate"

echo -e "${GREEN}To start the server, run:${NC}"
echo -e "python start_server.py --type main"

echo -e "${YELLOW}If you encounter any errors related to TensorFlow API changes, you may need to update the server code.${NC}"
echo -e "${YELLOW}Consider creating a compatibility wrapper script that adapts between TensorFlow 1.x and 2.x APIs.${NC}" 