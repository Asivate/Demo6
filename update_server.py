#!/usr/bin/env python

"""
Server Code Update Script

This script updates the SoundWatch server code to be compatible with TensorFlow 2.x
and Python 3.11. It makes the following changes:

1. Adds imports for the TensorFlow compatibility layer
2. Updates NumPy fromstring calls to use the compatibility function
3. Updates model loading to use the compatibility function
4. Adds basic logging for debugging purposes
"""

import os
import re
import sys
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("update_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Files to update
SERVER_FILES = [
    'server.py',
    'e2eServer.py',
    'modelTimerServer.py'
]

def backup_file(filename):
    """Create a backup of the file"""
    backup_name = f"{filename}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(filename, backup_name)
    logging.info(f"Created backup of {filename} as {backup_name}")
    return backup_name

def update_imports(content):
    """Add imports for the TensorFlow compatibility layer"""
    # Check if already imported
    if "import tensorflow_compatibility" in content:
        return content
    
    # Find the tensorflow import line
    tf_import_pattern = r'import tensorflow as tf'
    if re.search(tf_import_pattern, content):
        # Add our compatibility import after the TensorFlow import
        new_import = (
            "import tensorflow as tf\n"
            "# Import compatibility layer for TensorFlow 2.x\n"
            "import tensorflow_compatibility\n"
            "from tensorflow_compatibility import graph, numpy_compat_fromstring, load_model_compat, with_graph_context"
        )
        content = re.sub(tf_import_pattern, new_import, content)
    else:
        # If no TensorFlow import found, add at the beginning with a warning
        logging.warning(f"Could not find 'import tensorflow as tf' pattern. Adding imports at the top.")
        content = (
            "import tensorflow as tf\n"
            "# Import compatibility layer for TensorFlow 2.x\n"
            "import tensorflow_compatibility\n"
            "from tensorflow_compatibility import graph, numpy_compat_fromstring, load_model_compat, with_graph_context\n\n"
        ) + content
    
    return content

def update_model_loading(content):
    """Update model loading to use compatibility function"""
    # Replace model loading
    model_load_pattern = r'model = load_model\(([^)]+)\)'
    if re.search(model_load_pattern, content):
        content = re.sub(
            model_load_pattern,
            r'model = load_model_compat(\1)',
            content
        )
        logging.info("Updated model loading to use compatibility function")
    else:
        logging.warning("Could not find model loading pattern")
    
    return content

def update_graph_references(content):
    """Update graph references to use compatibility variable"""
    content = re.sub(
        r'graph = tf\.get_default_graph\(\)',
        r'# Use compatibility graph\ngraph = tensorflow_compatibility.graph',
        content
    )
    logging.info("Updated graph references")
    
    return content

def update_numpy_fromstring(content):
    """Update numpy fromstring calls to use compatibility function"""
    # Find and replace np.fromstring calls
    fromstring_pattern = r'np\.fromstring\(([^,]+),\s*dtype=([^,]+)(?:,\s*sep=([^)]+))?\)'
    
    def replace_fromstring(match):
        # Get the arguments
        data = match.group(1).strip()
        dtype = match.group(2).strip()
        sep = match.group(3) if match.group(3) else "''"
        
        return f"numpy_compat_fromstring({data}, dtype={dtype}, sep={sep})"
    
    if re.search(fromstring_pattern, content):
        content = re.sub(fromstring_pattern, replace_fromstring, content)
        logging.info("Updated numpy fromstring calls to use compatibility function")
    else:
        logging.warning("Could not find np.fromstring pattern")
    
    return content

def update_with_graph_context(content):
    """Update graph context usages with the decorator pattern where needed"""
    
    # Find functions that use the graph context
    graph_context_pattern = r'with graph\.as_default\(\):'
    
    if re.search(graph_context_pattern, content):
        logging.info("Found graph context usage, but not updating functions with decorators yet")
        logging.info("Please consider manually decorating your prediction functions with @with_graph_context")
    
    return content

def update_file(filename):
    """Update a single file"""
    logging.info(f"Updating {filename}...")
    
    try:
        # Create backup
        backup_name = backup_file(filename)
        
        # Read file content
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply updates
        content = update_imports(content)
        content = update_model_loading(content)
        content = update_graph_references(content)
        content = update_numpy_fromstring(content)
        content = update_with_graph_context(content)
        
        # Add warning comment about manual review
        warning_comment = """
# WARNING: This file has been automatically updated for TensorFlow 2.x compatibility.
# Please review the changes and test thoroughly before deploying to production.
# If you encounter issues, you may need to manually adjust the code or refer to
# the tensorflow_compatibility.py module.
"""
        content = warning_comment + content
        
        # Write updated content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logging.info(f"Successfully updated {filename}")
        return True
    except Exception as e:
        logging.error(f"Error updating {filename}: {e}")
        # Try to restore from backup
        try:
            if 'backup_name' in locals():
                shutil.copy2(backup_name, filename)
                logging.info(f"Restored {filename} from backup")
        except Exception as restore_error:
            logging.error(f"Error restoring from backup: {restore_error}")
        return False

def main():
    """Main function"""
    logging.info("Starting server code update process...")
    
    success_count = 0
    for filename in SERVER_FILES:
        if os.path.exists(filename):
            if update_file(filename):
                success_count += 1
        else:
            logging.warning(f"File {filename} not found, skipping")
    
    if success_count == len(SERVER_FILES):
        logging.info("All files updated successfully.")
        logging.info("Please review the changes and test the server before deploying.")
    else:
        logging.warning(f"Updated {success_count}/{len(SERVER_FILES)} files. Some files might need manual updates.")
    
    logging.info("Update process completed.")

if __name__ == "__main__":
    main() 