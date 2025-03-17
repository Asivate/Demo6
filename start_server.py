#!/usr/bin/env python

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start SoundWatch server')
    parser.add_argument('--type', choices=['main', 'e2e', 'timer'], default='main',
                        help='Type of server to start: main (default), e2e (end-to-end latency), or timer (model timer)')
    
    args = parser.parse_args()
    
    if args.type == 'main':
        print("Starting main server...")
        os.system("python server.py")
    elif args.type == 'e2e':
        print("Starting end-to-end latency server...")
        os.system("python e2eServer.py")
    elif args.type == 'timer':
        print("Starting model timer server...")
        os.system("python modelTimerServer.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 