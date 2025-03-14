#!/bin/bash
# Start script that ensures Google Cloud Speech API is properly configured

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default configuration
CREDENTIALS_FILE="/home/hirwa0250/asivate-452914-5c12101797af.json"
PORT=8080
DEBUG=1

# Parse command line arguments
while getopts "c:p:d" opt; do
  case $opt in
    c)
      CREDENTIALS_FILE="$OPTARG"
      ;;
    p)
      PORT="$OPTARG"
      ;;
    d)
      DEBUG=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo -e "${YELLOW}Starting SoundWatch server with Speech API support...${NC}"

# 1. Check if credentials file exists
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo -e "${RED}Error: Credentials file not found at $CREDENTIALS_FILE${NC}"
    echo "Please specify the correct path with -c flag"
    exit 1
fi

echo -e "${GREEN}✓ Found credentials file at $CREDENTIALS_FILE${NC}"

# 2. Set the environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_FILE"
echo -e "${GREEN}✓ Set GOOGLE_APPLICATION_CREDENTIALS environment variable${NC}"

# 3. Ensure the Speech API is enabled for this project
echo -e "${YELLOW}Verifying Speech API configuration...${NC}"

# Build server command
SERVER_CMD="python server.py --use-google-speech --port $PORT"
if [ $DEBUG -eq 1 ]; then
    SERVER_CMD="$SERVER_CMD --debug"
    echo -e "${YELLOW}Debug mode enabled${NC}"
fi

# Get server IP for display
IP=$(hostname -I | awk '{print $1}')
echo -e "${GREEN}Server will be available at:${NC}"
echo -e "    http://$IP:$PORT"
echo -e "    WebSocket: ws://$IP:$PORT"
echo ""

echo -e "${YELLOW}Starting server with command:${NC}"
echo -e "    $SERVER_CMD"
echo ""

# Run the server
echo -e "${GREEN}=== SERVER STARTING ===${NC}"
$SERVER_CMD 