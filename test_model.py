#!/usr/bin/env python
"""
Test the sound classification model directly with sample audio
"""

import os
import sys
import numpy as np
import argparse
import tensorflow as tf
import json
from datetime import datetime

# Set TensorFlow to use compatibility mode with TF 1.x code
tf.compat.v1.disable_eager_execution()

def load_model(model_path):
    """
    Load the TensorFlow model from the specified path
    
    Args:
        model_path: Path to the model file
    
    Returns:
        The loaded TensorFlow model
    """
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run download_model.py first to download the model file")
        sys.exit(1)
    
    try:
        # Create a TensorFlow session
        sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(sess)
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_test_audio(duration=5, sample_rate=16000, frequency=440):
    """
    Generate a test audio signal (sine wave)
    
    Args:
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of the sine wave in Hz
    
    Returns:
        numpy array of audio samples
    """
    print(f"Generating {duration}s test audio at {sample_rate}Hz...")
    
    # Generate time points
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sine wave
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise
    audio += 0.01 * np.random.normal(size=audio.shape)
    
    return audio.astype(np.float32)

def load_labels():
    """
    Load the sound labels
    
    Returns:
        List of sound labels
    """
    try:
        # Try to load from homesounds.py if available
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from homesounds import HOMESOUNDS
        return HOMESOUNDS
    except ImportError:
        # Fallback to a basic list of common sounds
        return [
            "Speech", "Dog", "Cat", "Alarm", "Dishes", "Faucet", "Knock", 
            "Blender", "Vacuum", "Washing Machine", "Microwave", "Doorbell"
        ]

def make_prediction(model, audio, sample_rate=16000):
    """
    Make a prediction using the model
    
    Args:
        model: The TensorFlow model
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz
    
    Returns:
        Dictionary with prediction results
    """
    print("Making prediction...")
    
    # Reshape audio for the model (add batch and channel dimensions)
    audio_reshaped = audio.reshape(1, -1, 1)
    
    # Make prediction
    prediction = model.predict(audio_reshaped)
    
    # Get the top prediction
    top_idx = np.argmax(prediction[0])
    top_confidence = float(prediction[0][top_idx])
    
    # Get all predictions with confidence
    labels = load_labels()
    all_predictions = []
    
    for i, conf in enumerate(prediction[0]):
        if i < len(labels):
            all_predictions.append({
                "label": labels[i],
                "confidence": float(conf)
            })
    
    # Sort by confidence
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Return results
    return {
        "top_prediction": {
            "label": labels[top_idx] if top_idx < len(labels) else f"Unknown-{top_idx}",
            "confidence": top_confidence
        },
        "all_predictions": all_predictions[:5]  # Return top 5 predictions
    }

def main():
    parser = argparse.ArgumentParser(description='Test the sound classification model')
    parser.add_argument('--model-path', default='models/example_model.hdf5',
                        help='Path to the model file')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Duration of test audio in seconds')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate in Hz')
    parser.add_argument('--frequency', type=float, default=440.0,
                        help='Frequency of test tone in Hz')
    parser.add_argument('--speech-bias', type=float, default=0.3,
                        help='Speech bias correction factor')
    args = parser.parse_args()
    
    try:
        # Load the model
        model = load_model(args.model_path)
        
        # Generate test audio
        audio = generate_test_audio(args.duration, args.sample_rate, args.frequency)
        
        # Make prediction
        results = make_prediction(model, audio, args.sample_rate)
        
        # Print results
        print("\n--- Prediction Results ---")
        print(f"Top prediction: {results['top_prediction']['label']} "
              f"({results['top_prediction']['confidence']:.4f})")
        
        print("\nTop 5 predictions:")
        for i, pred in enumerate(results['all_predictions'], 1):
            print(f"{i}. {pred['label']}: {pred['confidence']:.4f}")
        
        # Apply speech bias correction if the top prediction is Speech
        if results['top_prediction']['label'] == "Speech" and args.speech_bias > 0:
            print("\n--- After Speech Bias Correction ---")
            
            # Find the Speech prediction
            speech_idx = next((i for i, p in enumerate(results['all_predictions']) 
                              if p['label'] == "Speech"), None)
            
            if speech_idx is not None:
                # Apply bias correction
                corrected_confidence = max(0, results['all_predictions'][speech_idx]['confidence'] - args.speech_bias)
                print(f"Speech confidence: {results['all_predictions'][speech_idx]['confidence']:.4f} -> {corrected_confidence:.4f}")
                
                # Create a copy of predictions with corrected confidence
                corrected_predictions = results['all_predictions'].copy()
                corrected_predictions[speech_idx]['confidence'] = corrected_confidence
                
                # Sort by confidence again
                corrected_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Print corrected top 5
                print("\nCorrected top 5 predictions:")
                for i, pred in enumerate(corrected_predictions, 1):
                    print(f"{i}. {pred['label']}: {pred['confidence']:.4f}")
                
                # Check if top prediction changed
                if corrected_predictions[0]['label'] != results['top_prediction']['label']:
                    print(f"\nTop prediction changed from {results['top_prediction']['label']} "
                          f"to {corrected_predictions[0]['label']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 