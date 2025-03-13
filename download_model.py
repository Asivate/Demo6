#!/usr/bin/env python
"""
Download the model file if it doesn't exist
"""

import os
import sys
import urllib.request
import argparse

def download_file(url, destination):
    """
    Download a file from a URL to a destination
    
    Args:
        url: URL to download from
        destination: Path to save the file to
    """
    print(f"Downloading {url} to {destination}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download the file with progress reporting
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            s = f"\rDownloading: {percent:.1f}% ({read_so_far} / {total_size} bytes)"
            sys.stdout.write(s)
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloaded {read_so_far} bytes")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def check_model_file(model_path, model_url=None):
    """
    Check if the model file exists and download it if it doesn't
    
    Args:
        model_path: Path to the model file
        model_url: URL to download the model from
    
    Returns:
        bool: True if the model file exists or was downloaded successfully
    """
    if os.path.exists(model_path):
        print(f"Model file found at {model_path}")
        return True
    
    if model_url:
        print(f"Model file not found at {model_path}")
        return download_file(model_url, model_path)
    else:
        print(f"Model file not found at {model_path} and no URL provided for download")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download model file if it doesn\'t exist')
    parser.add_argument('--model-path', default='models/example_model.hdf5',
                        help='Path to the model file')
    parser.add_argument('--model-url', 
                        default='https://github.com/makeabilitylab/SoundWatch/raw/master/server/models/example_model.hdf5',
                        help='URL to download the model from')
    args = parser.parse_args()
    
    success = check_model_file(args.model_path, args.model_url)
    
    if success:
        print("Model file is ready to use")
        sys.exit(0)
    else:
        print("Failed to download model file")
        sys.exit(1)

if __name__ == "__main__":
    main() 