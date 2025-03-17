#!/usr/bin/env python3.6

"""
SoundWatch Server Installation Checker
This script verifies that all required components are properly installed and configured.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    print("\n--- Checking Python Version ---")
    python_version = sys.version_info
    print(f"Using Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor != 6:
        print("WARNING: This application was designed for Python 3.6.x")
        print(f"Current Python version is {python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    return True

def check_required_modules():
    print("\n--- Checking Required Modules ---")
    required_modules = [
        "flask", "flask_socketio", "keras", "tensorflow", 
        "numpy", "wget", "eventlet", "gunicorn"
    ]
    
    all_installed = True
    for module_name in required_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ Found {module_name} {getattr(module, '__version__', 'unknown version')}")
        except ImportError:
            print(f"✗ Missing module: {module_name}")
            all_installed = False
    
    return all_installed

def check_tensorflow_keras_compatibility():
    print("\n--- Checking TensorFlow/Keras Compatibility ---")
    try:
        import tensorflow as tf
        import keras
        
        tf_version = tf.__version__
        keras_version = keras.__version__
        
        print(f"TensorFlow version: {tf_version}")
        print(f"Keras version: {keras_version}")
        
        # For this specific project, we're expecting TensorFlow 1.5.0 and Keras 2.1.6
        if tf_version != '1.5.0':
            print(f"WARNING: Expected TensorFlow 1.5.0, but found {tf_version}")
        
        if keras_version != '2.1.6':
            print(f"WARNING: Expected Keras 2.1.6, but found {keras_version}")
        
        return True
    except Exception as e:
        print(f"Error checking TensorFlow/Keras: {e}")
        return False

def check_model_files():
    print("\n--- Checking Model Files ---")
    model_file = Path("models/example_model.hdf5")
    
    if model_file.is_file():
        print(f"✓ Found model file: {model_file}")
        print(f"  Size: {model_file.stat().st_size / (1024*1024):.2f} MB")
        return True
    else:
        print(f"✗ Missing model file: {model_file}")
        print("  The model will be downloaded automatically when running the server.")
        return False

def check_server_configuration():
    print("\n--- Checking Server Configuration ---")
    server_files = ["server.py", "e2eServer.py", "modelTimerServer.py"]
    
    for file_name in server_files:
        file_path = Path(file_name)
        if file_path.is_file():
            print(f"✓ Found server file: {file_name}")
            
            # Check if the server is configured to listen on 0.0.0.0:8080
            with open(file_path, 'r') as f:
                content = f.read()
                if "host='0.0.0.0'" in content and "port=8080" in content:
                    print(f"  ✓ {file_name} configured to listen on 0.0.0.0:8080")
                else:
                    print(f"  ✗ {file_name} may not be properly configured to listen on 0.0.0.0:8080")
        else:
            print(f"✗ Missing server file: {file_name}")
    
    return True

def check_network_access():
    print("\n--- Checking Network Access ---")
    try:
        # Check if the system hostname resolves
        hostname = subprocess.check_output("hostname", shell=True).decode().strip()
        print(f"System hostname: {hostname}")
        
        # Check if port 8080 is already in use
        result = subprocess.run("netstat -tuln | grep :8080", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("⚠ Warning: Port 8080 is already in use. The server may not start correctly.")
            print("  " + result.stdout.decode().strip())
        else:
            print("✓ Port 8080 is available")
        
        return True
    except Exception as e:
        print(f"Error checking network: {e}")
        return False

def main():
    print("=== SoundWatch Server Installation Checker ===")
    
    checks = [
        check_python_version,
        check_required_modules,
        check_tensorflow_keras_compatibility,
        check_model_files,
        check_server_configuration,
        check_network_access
    ]
    
    results = [check() for check in checks]
    
    print("\n=== Summary ===")
    if all(results):
        print("All checks passed! The server should be ready to run.")
    else:
        print("Some checks failed. Please address the issues before running the server.")
    
    print("\nTo start the server, activate the virtual environment and run:")
    print("python server.py  # For the main server")
    print("python start_server.py --type [main|e2e|timer]  # For alternative server types")

if __name__ == "__main__":
    main() 