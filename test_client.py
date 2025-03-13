#!/usr/bin/env python
"""
Test client for SoundWatch server
This script connects to the SoundWatch server and listens for sound predictions
"""

import socketio
import time
import argparse
import os
import numpy as np
import json
from datetime import datetime

# Create a Socket.IO client
sio = socketio.Client()

# Keep track of received predictions
predictions = []

@sio.event
def connect():
    print(f"Connected to server at {datetime.now().strftime('%H:%M:%S')}")
    print("Waiting for sound predictions...")

@sio.event
def disconnect():
    print(f"Disconnected from server at {datetime.now().strftime('%H:%M:%S')}")

@sio.event
def sound_prediction(data):
    """Handle sound prediction events from the server"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"\n[{timestamp}] Received sound prediction:")
    
    # Extract and print the prediction details
    label = data.get('label', 'Unknown')
    confidence = data.get('confidence', 0.0)
    db_level = data.get('db_level', 0.0)
    
    print(f"  Sound: {label}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Decibel Level: {db_level:.2f} dB")
    
    # Print additional details if available
    if 'details' in data and data['details']:
        print(f"  Details: {data['details']}")
    
    # Store the prediction
    predictions.append({
        'timestamp': timestamp,
        'label': label,
        'confidence': confidence,
        'db_level': db_level,
        'details': data.get('details', '')
    })

def send_test_audio(duration=5):
    """
    Generate and send test audio data to the server
    
    Args:
        duration: Duration of the test audio in seconds
    """
    print(f"Sending {duration} seconds of test audio...")
    
    # Sample rate (samples per second)
    sample_rate = 16000
    
    # Generate random audio data (white noise)
    num_samples = int(duration * sample_rate)
    audio_data = np.random.normal(0, 0.1, num_samples).astype(np.float32)
    
    # Convert to list for JSON serialization
    audio_list = audio_data.tolist()
    
    # Send the audio data to the server
    sio.emit('audio_data', {
        'audio': audio_list,
        'sample_rate': sample_rate
    })
    
    print(f"Sent {len(audio_list)} audio samples to the server")

def main():
    parser = argparse.ArgumentParser(description='Test client for SoundWatch server')
    parser.add_argument('--server', default='http://localhost:8080', 
                        help='Server URL (default: http://localhost:8080)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Test duration in seconds (default: 10)')
    parser.add_argument('--send-audio', action='store_true',
                        help='Send test audio data to the server')
    args = parser.parse_args()
    
    try:
        # Connect to the server
        print(f"Connecting to server at {args.server}...")
        sio.connect(args.server)
        
        # Send test audio if requested
        if args.send_audio:
            time.sleep(1)  # Wait for connection to stabilize
            send_test_audio(5)  # Send 5 seconds of test audio
        
        # Wait for the specified duration
        print(f"Listening for predictions for {args.duration} seconds...")
        time.sleep(args.duration)
        
        # Print summary
        print("\n--- Test Summary ---")
        print(f"Connected to: {args.server}")
        print(f"Test duration: {args.duration} seconds")
        print(f"Predictions received: {len(predictions)}")
        
        if predictions:
            print("\nPrediction Summary:")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. [{pred['timestamp']}] {pred['label']} ({pred['confidence']:.4f})")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Disconnect from the server
        if sio.connected:
            sio.disconnect()
        print("Test completed")

if __name__ == "__main__":
    main() 