"""
Run Optimized SoundWatch Server

This script runs the SoundWatch server with optimized speech recognition settings.
It includes all the improvements to Whisper transcription and sentiment analysis.
"""
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('soundwatch_server.log')
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're in a virtual environment and dependencies are installed"""
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        logger.warning("Not running in a virtual environment! It's recommended to use a venv.")
        choice = input("Do you want to continue anyway? (y/n): ")
        if choice.lower() != 'y':
            logger.info("Exiting. Please create and activate a virtual environment.")
            sys.exit(0)
    
    # Check for required dependencies
    try:
        import torch
        import transformers
        import flask
        import scipy
        import numpy
        logger.info("Required dependencies are installed.")
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.info("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check if CUDA is available for GPU acceleration
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU acceleration available: {gpu_name}")
        else:
            logger.info("Running in CPU mode (no CUDA GPU detected)")
    except Exception as e:
        logger.warning(f"Could not check CUDA availability: {e}")

def print_server_info():
    """Print information about the server and its improvements"""
    print("\n" + "="*70)
    print("SOUNDWATCH OPTIMIZED SERVER".center(70))
    print("="*70)
    print("\nStarting server with enhanced Whisper transcription:")
    print(" ✓ Improved audio preprocessing for cleaner input")
    print(" ✓ Voice activity detection to avoid processing silence")
    print(" ✓ Audio buffering for better context in transcription")
    print(" ✓ Hallucination prevention with advanced filtering")
    print(" ✓ Optimized Whisper model with appropriate parameters")
    print(" ✓ Enhanced sentiment analysis with emotion categories")
    print("\nAccess the web interface at:")
    print(" → http://localhost:8080 (local)")
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f" → http://{local_ip}:8080 (internal network)")
        # Add external IP information
        print(f" → http://34.16.101.179:8080 (external - Internet)")
    except:
        pass
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")

def run_server():
    """Run the SoundWatch server"""
    try:
        # Import the server module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        from server import app, socketio
        
        # Start the server
        print_server_info()
        socketio.run(app, debug=False, host='0.0.0.0', port=8080)
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check environment
    check_environment()
    
    # Run server
    run_server() 