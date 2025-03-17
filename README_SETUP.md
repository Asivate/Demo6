# SoundWatch Server Setup Guide

This guide will help you set up and run the SoundWatch server on your Debian machine using Python 3.6.4.

## Prerequisites

- Debian Linux system
- Python 3.6.4 installed
- Access to terminal/command line
- Internet connection to download packages

## Setup Instructions

### 1. Make the setup script executable

```bash
chmod +x setup_venv.sh
```

### 2. Run the setup script

This will create a Python 3.6.4 virtual environment and install all required dependencies:

```bash
./setup_venv.sh
```

The script will:
- Verify Python 3.6.4 is available
- Create a virtual environment
- Install all required packages with specific versions for compatibility

### 3. Activate the virtual environment

After setup is complete, activate the virtual environment:

```bash
source venv/soundwatch_env/bin/activate
```

You should see `(soundwatch_env)` at the beginning of your command prompt, indicating that the virtual environment is active.

## Running the Server

### Starting the Main Server

To start the main SoundWatch server:

```bash
python server.py
```

This will start the server on IP address 0.0.0.0 (all interfaces) and port 8080.

### Alternative Server Options

You can also use the `start_server.py` script to easily choose which server to start:

```bash
# For the main audio classification server
python start_server.py --type main

# For the end-to-end latency measurement server
python start_server.py --type e2e

# For the model timer server
python start_server.py --type timer
```

## Verifying the Server

Once the server is running, you can verify it works by:

1. Opening a web browser and navigating to `http://your-server-ip:8080/`
2. You should see the server's debug interface
3. You can test sending a message through the interface to verify the Socket.IO connection

## Server Types

- **Main Server (server.py)**: The primary server that handles audio classification
- **End-to-End Latency Server (e2eServer.py)**: Measures full round-trip performance
- **Model Timer Server (modelTimerServer.py)**: Records model prediction timing data

## Troubleshooting

If you encounter issues:

1. **Verify Python Version**: Make sure you have Python 3.6.4
    ```
    python3.6 --version
    ```

2. **Check Package Installation**: Verify all packages were installed
    ```
    pip list
    ```

3. **Network Issues**: Ensure port 8080 is open and accessible

4. **Model Download**: If the model doesn't download automatically, you may need to manually download it from the link in the code

## Notes

- The server is configured to run on port 8080, which matches the client configuration
- The server will listen on all interfaces (0.0.0.0) to be accessible from other devices
- Make sure your firewall allows connections to port 8080 