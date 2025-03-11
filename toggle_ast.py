#!/usr/bin/env python3
"""
Toggle script for switching between AST and TensorFlow models.
This script sets the USE_AST_MODEL environment variable and restarts the server.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_current_setting():
    """Get the current AST model setting"""
    current_setting = os.environ.get('USE_AST_MODEL', '1')
    return current_setting == '1'

def toggle_ast_setting(enable=None):
    """Toggle or set the AST model setting"""
    if enable is None:
        # Toggle the current setting
        current_setting = get_current_setting()
        new_setting = '0' if current_setting else '1'
    else:
        # Set to the specified value
        new_setting = '1' if enable else '0'
    
    return new_setting

def restart_server(use_ast):
    """Restart the server with the new AST setting"""
    # Get the path to the server directory
    server_dir = Path(__file__).resolve().parent
    
    # Check if the server script exists
    server_script = server_dir / "server.py"
    if not server_script.exists():
        logger.error(f"Server script not found at {server_script}")
        return False
    
    # Build the command to run
    env = os.environ.copy()
    env["USE_AST_MODEL"] = use_ast
    
    # Notify the user
    logger.info(f"Setting USE_AST_MODEL={use_ast}")
    logger.info(f"AST model is now {'ENABLED' if use_ast == '1' else 'DISABLED'}")
    
    # Ask for confirmation
    if input("Restart the server now? (y/n): ").lower() == 'y':
        logger.info("Restarting server...")
        
        # Platform-specific command
        if sys.platform.startswith('win'):
            # For Windows
            cmd = ['python', str(server_script)]
        else:
            # For Linux/Mac
            cmd = ['python3', str(server_script)]
        
        try:
            # Start the server in a new process
            subprocess.Popen(cmd, env=env, cwd=str(server_dir))
            logger.info("Server started with new settings")
            return True
        except Exception as e:
            logger.error(f"Failed to restart server: {e}")
            return False
    else:
        logger.info("Server not restarted. New setting will take effect on next manual start.")
        return True

def main():
    """Main function"""
    # Parse command-line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ('on', 'enable', '1', 'true', 'yes'):
            new_setting = '1'
        elif arg in ('off', 'disable', '0', 'false', 'no'):
            new_setting = '0'
        else:
            print(f"Invalid argument: {arg}")
            print("Usage: python toggle_ast.py [on|off]")
            return 1
    else:
        # Toggle the current setting
        current_setting = get_current_setting()
        new_setting = toggle_ast_setting()
        print(f"Current AST setting: {'ENABLED' if current_setting else 'DISABLED'}")
        print(f"New AST setting: {'ENABLED' if new_setting == '1' else 'DISABLED'}")
    
    # Restart the server with the new setting
    restart_server(new_setting)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 