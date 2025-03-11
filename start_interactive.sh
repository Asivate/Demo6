#!/bin/bash
# Interactive startup script for SoundWatch Server
# This script sets up the environment and runs the interactive Python script
# Server will always run on port 8080

# Change to the directory containing this script
cd "$(dirname "$0")"

# Set up environment variables for Google Cloud credentials if available
if [ -f "/home/hirwa0250/asivate-452914-5c12101797af.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="/home/hirwa0250/asivate-452914-5c12101797af.json"
    echo -e "\e[32mSet Google Cloud credentials from: $GOOGLE_APPLICATION_CREDENTIALS\e[0m"
fi

# Make sure the interactive script is executable
chmod +x interactive_start.py

# Display information about the server port and models
echo -e "\e[36mSoundWatch Server will run on port 8080\e[0m"
echo -e "\e[36mYou will be able to choose between TensorFlow and AST models\e[0m"
echo -e "\e[36mAST model selection will be applied via environment variables\e[0m"

# Run the interactive startup script
python3 interactive_start.py

# Exit with the same status code as the Python script
exit $? 