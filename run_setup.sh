#!/bin/bash

# Master setup script for SoundWatch Server
# This script performs all necessary steps to set up the environment and update code for Python 3.11.2

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}SoundWatch Server Complete Setup Script${NC}"
echo -e "${GREEN}==============================================${NC}"
echo -e "${YELLOW}This script will:${NC}"
echo -e "${YELLOW}1. Set up a virtual environment with Python 3.11${NC}"
echo -e "${YELLOW}2. Install all required dependencies${NC}"
echo -e "${YELLOW}3. Update server code for TensorFlow 2.x compatibility${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""

# Check if script has execute permissions
if [ ! -x "./setup_environment.sh" ]; then
    echo -e "${YELLOW}Setting execute permissions on scripts...${NC}"
    chmod +x setup_environment.sh
    chmod +x update_server.py
    chmod +x start_server.py
fi

# Step 1: Setup environment
echo -e "${GREEN}Step 1: Setting up Python environment...${NC}"
./setup_environment.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}Environment setup failed. Please check the errors above.${NC}"
    exit 1
fi

# Step 2: Activate the virtual environment
echo -e "${GREEN}Step 2: Activating virtual environment...${NC}"
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}Virtual environment activated!${NC}"
else
    echo -e "${RED}Virtual environment not found. Please run setup_environment.sh first.${NC}"
    exit 1
fi

# Step 3: Update server code
echo -e "${GREEN}Step 3: Updating server code for TensorFlow 2.x compatibility...${NC}"
python update_server.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Server code update failed. Please check the errors above.${NC}"
    echo -e "${YELLOW}You may need to manually update some code files.${NC}"
else
    echo -e "${GREEN}Server code updated successfully!${NC}"
fi

echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}==============================================${NC}"
echo -e "${YELLOW}To start the server:${NC}"
echo -e "1. Activate the virtual environment if not already activated:"
echo -e "   ${GREEN}source .venv/bin/activate${NC}"
echo -e "2. Run one of the following commands:"
echo -e "   ${GREEN}python start_server.py --type main${NC}   (Main server)"
echo -e "   ${GREEN}python start_server.py --type e2e${NC}    (End-to-end latency server)"
echo -e "   ${GREEN}python start_server.py --type timer${NC}  (Model timer server)"
echo -e "${GREEN}==============================================${NC}"
echo -e "${YELLOW}Note: The server will listen on 0.0.0.0:8080 as configured.${NC}"
echo -e "${YELLOW}Make sure your firewall allows connections to port 8080.${NC}"
echo -e "${GREEN}==============================================${NC}" 