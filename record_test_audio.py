#!/usr/bin/env python3
"""
Simple audio recording tool to help test the sentiment analysis feature.
This script records audio from the microphone and saves it to a file.
"""
import os
import sys
import time
import argparse
import numpy as np
import wave

# Check if PyAudio is installed
try:
    import pyaudio
except ImportError:
    print("PyAudio not installed. Installing now...")
    
    # Different install methods depending on platform
    import platform
    system = platform.system()
    
    if system == "Linux":
        print("Installing system dependencies...")
        try:
            if os.system("apt-get -h > /dev/null 2>&1") == 0:
                os.system("sudo apt-get update && sudo apt-get install -y python3-pyaudio portaudio19-dev")
            elif os.system("dnf -h > /dev/null 2>&1") == 0:
                os.system("sudo dnf install -y python3-pyaudio portaudio-devel")
            else:
                print("Could not determine package manager. Please install PyAudio manually.")
                sys.exit(1)
        except Exception as e:
            print(f"Failed to install system dependencies: {e}")
            print("Please install the required packages manually:")
            print("  Ubuntu/Debian: sudo apt-get install python3-pyaudio portaudio19-dev")
            print("  Fedora: sudo dnf install python3-pyaudio portaudio-devel")
            sys.exit(1)
    
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
        import pyaudio
    except Exception as e:
        print(f"Failed to install PyAudio: {e}")
        print("Please install PyAudio manually.")
        sys.exit(1)

def record_audio(filename="test_audio.wav", duration=5, sample_rate=16000, channels=1):
    """Record audio from the microphone and save it to a file."""
    chunk = 1024
    audio_format = pyaudio.paInt16
    
    p = pyaudio.PyAudio()
    
    # List available input devices
    print("\nAvailable audio input devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            print(f"  Device {i}: {dev_info['name']}")
    
    # Get default input device
    default_device = p.get_default_input_device_info()
    print(f"\nUsing default input device: {default_device['name']} (device {default_device['index']})")
    
    # Open audio stream
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print(f"\n🎤 Recording {duration} seconds of audio...")
    print("Please speak clearly into the microphone.")
    
    # Countdown before recording
    for i in range(3, 0, -1):
        print(f"Recording starts in {i}...")
        time.sleep(1)
    
    print("🔴 Recording started!")
    
    frames = []
    
    # Calculate total number of chunks to read based on duration
    total_chunks = int(sample_rate / chunk * duration)
    
    # Record audio in chunks
    for i in range(total_chunks):
        data = stream.read(chunk)
        frames.append(data)
        
        # Print progress
        progress = (i + 1) / total_chunks * 100
        sys.stdout.write(f"\rProgress: {progress:.1f}% ")
        sys.stdout.flush()
    
    print("\n✅ Recording complete!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio to a WAV file
    print(f"💾 Saving audio to {filename}...")
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"✅ Audio saved to {filename}")
    return filename

def play_audio(filename):
    """Play the recorded audio file."""
    if not os.path.exists(filename):
        print(f"❌ File {filename} does not exist.")
        return
    
    try:
        p = pyaudio.PyAudio()
        
        # Open the WAV file
        wf = wave.open(filename, 'rb')
        
        # Open a stream to play the audio
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        
        print(f"🔊 Playing audio: {filename}...")
        
        # Read and play the audio in chunks
        chunk = 1024
        data = wf.readframes(chunk)
        
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        
        print("✅ Playback complete!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()
        
    except Exception as e:
        print(f"❌ Error playing audio: {e}")

def main():
    parser = argparse.ArgumentParser(description='Record audio for sentiment analysis testing.')
    parser.add_argument('--filename', type=str, default='test_audio.wav',
                        help='Output filename (default: test_audio.wav)')
    parser.add_argument('--duration', type=int, default=5,
                        help='Recording duration in seconds (default: 5)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of audio channels (default: 1)')
    parser.add_argument('--play', action='store_true',
                        help='Play the recorded audio after recording')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎤 SoundWatch Audio Recorder")
    print("=" * 60)
    print(f"This tool records audio for testing the sentiment analysis feature.")
    print(f"The audio will be saved to: {args.filename}")
    print(f"Duration: {args.duration} seconds")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Channels: {args.channels}")
    print("=" * 60)
    
    # Ask for confirmation before recording
    user_input = input("Press Enter to start recording or Ctrl+C to cancel: ")
    
    try:
        # Record audio
        filename = record_audio(args.filename, args.duration, args.sample_rate, args.channels)
        
        # Play the audio if requested
        if args.play:
            play_audio(filename)
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print(f"1. Run the sentiment analysis test: python test_sentiment.py")
        print(f"2. Test with the full server: python server.py")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n❌ Recording cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 