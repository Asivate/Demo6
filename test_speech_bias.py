#!/usr/bin/env python
"""
Test script to verify speech bias correction in SoundWatch.

This script simulates audio input and checks if the speech bias correction
is working properly by comparing predictions with and without the correction.
"""

import numpy as np
import tensorflow as tf
import os
import sys
import time
import traceback

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
import homesounds
from vggish_input import waveform_to_examples

# Configure TensorFlow to use compatibility mode with TF 1.x code
tf.compat.v1.disable_eager_execution()

# Create a TensorFlow session
tf_graph = tf.compat.v1.Graph()
tf_session = tf.compat.v1.Session(graph=tf_graph)

# Model settings
MODEL_PATH = "models/example_model.hdf5"

# Load the model
def load_model():
    """Load the TensorFlow model."""
    print("Loading TensorFlow model...")
    model_filename = os.path.abspath(MODEL_PATH)
    
    with tf_graph.as_default():
        with tf_session.as_default():
            # Load model
            model = tf.keras.models.load_model(model_filename, compile=False)
            
            # Create a custom predict function that uses the session directly
            def custom_predict(x):
                # Get input and output tensors
                input_tensor = model.inputs[0]
                output_tensor = model.outputs[0]
                # Run prediction in the session
                return tf_session.run(output_tensor, feed_dict={input_tensor: x})
            
            # Replace the model's predict function with our custom one
            model.predict = custom_predict
            
            # Test it
            dummy_input = np.zeros((1, 96, 64, 1))
            _ = model.predict(dummy_input)
            print("Model loaded successfully")
            
            return model

# Function to make predictions with and without speech bias correction
def test_predictions(model, audio_data, sample_rate=16000, speech_bias_correction=0.3):
    """
    Test predictions with and without speech bias correction.
    
    Args:
        model: The TensorFlow model
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio data
        speech_bias_correction: Amount to subtract from speech confidence
        
    Returns:
        Tuple of (original_predictions, corrected_predictions)
    """
    # Convert waveform to examples (spectrogram features)
    x = waveform_to_examples(audio_data, sample_rate)
    
    if x.shape[0] == 0:
        print("Error: Empty audio features")
        return None, None
    
    # Use the first frame
    if x.shape[0] > 1:
        x_input = x[0].reshape(1, 96, 64, 1)
    else:
        x_input = x.reshape(1, 96, 64, 1)
    
    # Make prediction without bias correction
    with tf_graph.as_default():
        with tf_session.as_default():
            original_predictions = model.predict(x_input)
    
    # Make a copy of the predictions
    corrected_predictions = original_predictions.copy()
    
    # Apply speech bias correction
    speech_idx = homesounds.labels.get('speech', 4)  # Default to 4 if not found
    if speech_idx < len(corrected_predictions[0]):
        # Apply correction factor to reduce speech confidence
        corrected_predictions[0][speech_idx] -= speech_bias_correction
        # Ensure it doesn't go below 0
        corrected_predictions[0][speech_idx] = max(0.0, corrected_predictions[0][speech_idx])
    
    return original_predictions, corrected_predictions

# Function to print predictions
def print_predictions(predictions, label_list, title="Predictions"):
    """Print predictions with labels."""
    print(f"===== {title} =====")
    for idx, pred in enumerate(predictions[0]):
        if idx < len(label_list):
            print(f"{label_list[idx]}: {pred:.6f}")
    print("=" * (len(title) + 12))

# Generate synthetic audio data for testing
def generate_test_audio(duration=1.0, sample_rate=16000):
    """Generate synthetic audio data for testing."""
    # Generate white noise
    audio_data = np.random.normal(0, 0.1, int(duration * sample_rate))
    return audio_data

# Main function
def main():
    """Main function to test speech bias correction."""
    try:
        # Load the model
        model = load_model()
        
        # Generate test audio
        print("Generating test audio...")
        audio_data = generate_test_audio()
        
        # Test predictions
        print("Testing predictions...")
        original_predictions, corrected_predictions = test_predictions(model, audio_data)
        
        if original_predictions is not None and corrected_predictions is not None:
            # Print original predictions
            print_predictions(original_predictions, homesounds.everything, "Original Predictions")
            
            # Print corrected predictions
            print_predictions(corrected_predictions, homesounds.everything, "Corrected Predictions")
            
            # Get the top prediction for each
            original_top_idx = np.argmax(original_predictions[0])
            corrected_top_idx = np.argmax(corrected_predictions[0])
            
            original_top_label = homesounds.everything[original_top_idx] if original_top_idx < len(homesounds.everything) else "unknown"
            corrected_top_label = homesounds.everything[corrected_top_idx] if corrected_top_idx < len(homesounds.everything) else "unknown"
            
            print(f"\nTop original prediction: {original_top_label} ({original_predictions[0][original_top_idx]:.4f})")
            print(f"Top corrected prediction: {corrected_top_label} ({corrected_predictions[0][corrected_top_idx]:.4f})")
            
            # Check if the top prediction changed
            if original_top_idx != corrected_top_idx:
                print("\nSpeech bias correction changed the top prediction!")
            else:
                print("\nSpeech bias correction did not change the top prediction.")
            
            # Check speech confidence specifically
            speech_idx = homesounds.labels.get('speech', 4)
            if speech_idx < len(original_predictions[0]):
                original_speech_conf = original_predictions[0][speech_idx]
                corrected_speech_conf = corrected_predictions[0][speech_idx]
                print(f"\nSpeech confidence: {original_speech_conf:.4f} -> {corrected_speech_conf:.4f}")
                print(f"Reduction: {original_speech_conf - corrected_speech_conf:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        # Close the TensorFlow session
        tf_session.close()

if __name__ == "__main__":
    main() 