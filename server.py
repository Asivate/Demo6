from threading import Lock
from flask import Flask, render_template, session, request, copy_current_request_context, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect

# Add NumPy patch for TensorFlow compatibility
import numpy as np
# Apply patch for NumPy 1.20+ compatibility with older TensorFlow
if not hasattr(np, 'typeDict'):
    np.typeDict = {}
    for name in dir(np):
        if isinstance(getattr(np, name), type):
            np.typeDict[name] = getattr(np, name)
if not hasattr(np, 'object'):
    np.object = object

import tensorflow as tf
from tensorflow import keras
import homesounds
from pathlib import Path
import time
import argparse
import wget
import traceback
from helpers import dbFS
import os
import socket
import torch
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from scipy import signal
import threading
import queue
from transformers import pipeline
import logging
from functools import wraps
import gc
import sys
import json

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG level for maximum information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('soundwatch_server.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)

# Import Google Speech API directly - we'll only use this and not Vosk
try:
    from google_speech import GoogleSpeechToText, transcribe_with_google
    GOOGLE_SPEECH_AVAILABLE = True
    logger.info("Google Cloud Speech API is available")
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    logger.warning("Google Cloud Speech API not available. Install with: pip install google-cloud-speech")

# Import continuous sentiment analyzer
from continuous_sentiment_analysis import initialize_sentiment_analyzer, get_sentiment_analyzer
try:
    from sentiment_analyzer import analyze_sentiment
    SENTIMENT_ANALYZER_AVAILABLE = True
    logger.info("Sentiment analyzer module is available")
except ImportError:
    SENTIMENT_ANALYZER_AVAILABLE = False
    logger.warning("Sentiment analyzer module not available. Some functionality may be limited.")

# Thread-safe model access
model_lock = threading.RLock()
models = {}  # Dictionary to store loaded models
prediction_count = 0  # Counter for predictions

# Global variables for conversation history
MAX_CONVERSATION_HISTORY = 1000  # Maximum number of entries to keep
conversation_history = []  # List to store conversation history
transcript_history = []  # List to store transcription history

# Global variables for sentiment analysis
ENABLE_SENTIMENT_ANALYSIS = True  # Set to False to disable sentiment analysis
SPEECH_BIAS_CORRECTION = 0.15  # Correction factor for speech detection
APPLY_SPEECH_BIAS_CORRECTION = True  # Whether to apply speech bias correction
PREDICTION_THRES = 0.5  # Threshold for sound prediction confidence
DBLEVEL_THRES = 45  # Threshold for dB level to consider sound significant

# Audio buffers for continuous processing
AUDIO_BUFFER_SIZE = 5  # Buffer size in seconds
audio_buffer = []  # Buffer to store audio data for continuous processing
audio_buffer_lock = threading.RLock()  # Lock for thread-safe access to audio buffer

# Enhanced audio buffer for classification
classification_buffer = []  # Buffer specifically for sound classification
classification_buffer_lock = threading.RLock()
MIN_SAMPLES_FOR_CLASSIFICATION = 16000  # Minimum samples needed for classification

# Performance timer decorator
def performance_timer(func):
    """Decorator to log execution time for functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {(end_time - start_time)*1000:.2f} ms")
        return result
    return wrapper

# Create Flask application and SocketIO instance
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Setup sentiment analysis service
sentiment_analyzer = initialize_sentiment_analyzer(socketio, sample_rate=16000)

# Set up a continuous speech analysis thread
class ContinuousSpeechAnalysisThread(threading.Thread):
    def __init__(self, socketio):
        threading.Thread.__init__(self, daemon=True)
        self.socketio = socketio
        self.running = True
        self.audio_queue = queue.Queue()
        self.transcript = ""
        self.last_sentiment_time = time.time()
        self.sentiment_interval = 2.0  # Analyze sentiment every 2 seconds
        self.sentiment_pipeline = None
        
        # Initialize the sentiment pipeline
        if ENABLE_SENTIMENT_ANALYSIS:
            try:
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                logger.info("Sentiment analysis pipeline initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize sentiment analysis pipeline: {e}")
                logger.error(traceback.format_exc())
                self.sentiment_pipeline = None
    
    def add_audio_data(self, audio_data, sample_rate):
        """Add audio data to the processing queue"""
        if self.running and audio_data is not None and len(audio_data) > 0:
            logger.debug(f"Adding audio data to queue, length: {len(audio_data)}")
            self.audio_queue.put((audio_data, sample_rate))
        else:
            logger.warning(f"Not adding audio data to queue: running={self.running}, audio_data_length={len(audio_data) if audio_data is not None else 'None'}")
    
    def run(self):
        """Main thread loop for continuous speech analysis"""
        logger.info("Starting continuous speech analysis thread")
        
        while self.running:
            try:
                # Process audio data from the queue if available
                try:
                    audio_data, sample_rate = self.audio_queue.get(timeout=0.5)
                    logger.debug(f"Got audio data from queue, length: {len(audio_data)}")
                    self.process_audio(audio_data, sample_rate)
                except queue.Empty:
                    # No data in queue, continue to next iteration
                    continue
                
                # Check if it's time to analyze sentiment
                current_time = time.time()
                if current_time - self.last_sentiment_time >= self.sentiment_interval and self.transcript:
                    logger.info(f"Time to analyze sentiment, transcript length: {len(self.transcript)}")
                    self.analyze_sentiment()
                    self.last_sentiment_time = current_time
            except Exception as e:
                logger.error(f"Error in continuous speech analysis thread: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)  # Avoid tight loop on errors
        
        logger.info("Continuous speech analysis thread stopped")
    
    def process_audio(self, audio_data, sample_rate):
        """Process audio data for transcription"""
        if not GOOGLE_SPEECH_AVAILABLE:
            logger.warning("Google Speech API not available, skipping transcription")
            return
            
        try:
            # Transcribe the audio using Google Speech API
            logger.info(f"Transcribing audio data of length {len(audio_data)} with Google Speech API")
            transcript = transcribe_with_google(audio_data, sample_rate)
            
            if transcript and isinstance(transcript, str) and len(transcript.strip()) > 0:
                logger.info(f"Transcription result: '{transcript}'")
                # Update the current transcript
                self.transcript += " " + transcript
                
                # Add to transcript history with timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                entry = {
                    "timestamp": timestamp,
                    "text": transcript,
                    "sentiment": None  # Will be filled in by sentiment analysis
                }
                transcript_history.append(entry)
                
                # Limit history size
                while len(transcript_history) > MAX_CONVERSATION_HISTORY:
                    transcript_history.pop(0)
                
                # Emit transcript update to clients
                logger.info(f"Emitting transcript_update to clients: {transcript}")
                self.socketio.emit('transcript_update', {
                    'transcript': transcript,
                    'timestamp': timestamp
                })
                
                logger.info(f"Transcribed: {transcript}")
            else:
                logger.info("No valid transcription result returned (empty or not a string)")
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            logger.error(traceback.format_exc())
    
    def analyze_sentiment(self):
        """Analyze sentiment of the current transcript"""
        if not self.transcript:
            logger.warning("No transcript to analyze for sentiment")
            return
            
        if not self.sentiment_pipeline:
            logger.warning("Sentiment pipeline not initialized, skipping sentiment analysis")
            return
            
        try:
            # Use the dedicated sentiment analysis function
            logger.info(f"Analyzing sentiment for transcript: '{self.transcript}'")
            sentiment_result = analyze_text_sentiment(self.transcript)
            
            if sentiment_result:
                logger.info(f"Sentiment analysis result: {sentiment_result}")
                
                # Update the most recent transcript entry with sentiment
                if transcript_history and sentiment_result:
                    transcript_history[-1]["sentiment"] = sentiment_result
                    logger.info(f"Updated transcript history with sentiment: {sentiment_result}")
                
                # Emit sentiment notification to clients
                sentiment_notification = {
                    'text': self.transcript,
                    'sentiment': sentiment_result,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                logger.info(f"Emitting sentiment_notification to clients: {sentiment_notification}")
                self.socketio.emit('sentiment_notification', sentiment_notification)
                
                # Clear transcript after analysis for fresh start
                self.transcript = ""
                logger.info("Cleared transcript after sentiment analysis")
            else:
                logger.warning("No sentiment result returned from analysis")
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            logger.error(traceback.format_exc())
    
    def stop(self):
        """Stop the continuous analysis thread"""
        logger.info("Stopping continuous speech analysis thread")
        self.running = False

# Global instance of the continuous speech analysis thread
continuous_speech_thread = None

def start_continuous_speech_analysis(socketio):
    """Start the continuous speech analysis thread"""
    global continuous_speech_thread
    
    if continuous_speech_thread is None:
        continuous_speech_thread = ContinuousSpeechAnalysisThread(socketio)
        continuous_speech_thread.start()
        logger.info("Continuous speech analysis thread started")
    else:
        logger.warning("Continuous speech analysis thread already running")

def stop_continuous_speech_analysis():
    """Stop the continuous speech analysis thread"""
    global continuous_speech_thread
    
    if continuous_speech_thread is not None:
        continuous_speech_thread.stop()
        continuous_speech_thread.join(timeout=5)
        continuous_speech_thread = None
        logger.info("Continuous speech analysis thread stopped")

# Function to add audio data to the continuous speech analysis thread
def add_audio_for_analysis(audio_data, sample_rate):
    """Add audio data to the continuous speech analysis thread"""
    global continuous_speech_thread
    
    if continuous_speech_thread is not None:
        continuous_speech_thread.add_audio_data(audio_data, sample_rate)

# Function to get local and public IP addresses
def get_ip_addresses():
    """Get the local and public IP addresses."""
    local_ips = []
    public_ip = None
    
    try:
        # Get local IP addresses
        hostname = socket.gethostname()
        local_ips = socket.gethostbyname_ex(hostname)[2]
        
        # Get public IP address
        try:
            import requests
            public_ip = requests.get('https://api.ipify.org', timeout=2).text
        except Exception:
            # Fallback method
            public_ip = None
    except Exception as e:
        logger.error(f"Error getting IP addresses: {e}")
    
    return local_ips, public_ip

# Function to load models (TensorFlow or AST)
def load_models():
    """Load all required models"""
    with model_lock:
        try:
            # Load sound classification model - note we now catch the return value
            sound_model_loaded = load_sound_classification_model()
            
            # Initialize sentiment analysis model
            initialize_sentiment_analyzer(socketio, sample_rate=16000)
            
            if sound_model_loaded:
                logger.info("All models loaded successfully")
                return True
            else:
                logger.warning("Sound classification model could not be loaded, but continuing with other functionality")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            return False

def load_sound_classification_model():
    """Load the sound classification model"""
    global models
    
    try:
        # Check if model already loaded
        if 'sound_model' in models and models['sound_model'] is not None:
            logger.info("Sound classification model already loaded")
            return True
            
        logger.info("Loading sound classification model...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Set model paths
        model_dir = Path('models')
        model_path = model_dir / 'homesounds_model.h5'
        
        # Download model if it doesn't exist
        if not model_path.exists():
            logger.info("Downloading sound classification model...")
            try:
                download_sound_classification_model(model_path)
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                return False
        
        # Check if model exists now
        if not model_path.exists():
            logger.error("Model file still doesn't exist after download attempt")
            return False
            
        # Load the model
        model = keras.models.load_model(str(model_path))
        models['sound_model'] = model
        
        logger.info("Sound classification model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading sound classification model: {e}")
        logger.error(traceback.format_exc())
        return False

def download_sound_classification_model(model_path):
    """Download the sound classification model"""
    try:
        # First check if example_model.hdf5 exists and use that
        example_model_path = Path('models') / 'example_model.hdf5'
        if example_model_path.exists():
            logger.info(f"Using existing example model: {example_model_path}")
            # Copy or link the example model to the expected location
            import shutil
            shutil.copy(str(example_model_path), str(model_path))
            logger.info(f"Copied example model to {model_path}")
            return True
        
        # Look for any .h5 or .hdf5 files in the models directory
        model_dir = Path('models')
        found_models = list(model_dir.glob('*.h5')) + list(model_dir.glob('*.hdf5'))
        
        if found_models:
            # Use the first found model
            found_model = found_models[0]
            logger.info(f"Using existing model file: {found_model}")
            # Copy to the expected location
            import shutil
            shutil.copy(str(found_model), str(model_path))
            logger.info(f"Copied found model to {model_path}")
            return True
        
        # URLs to try in order
        model_urls = [
            "https://github.com/SmartWatchProject/SoundWatch/raw/master/models/homesounds_model.h5",
            "https://github.com/makeabilitylab/SoundWatch/raw/master/server/models/example_model.hdf5",
            "https://github.com/makeabilitylab/SoundWatch/raw/master/models/homesounds_model.h5"
        ]
        
        # Try each URL until one works
        for model_url in model_urls:
            try:
                logger.info(f"Attempting to download model from {model_url}")
                wget.download(model_url, str(model_path))
                logger.info(f"Model downloaded to {model_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to download from {model_url}: {e}")
                continue
        
        # If we reach here, all URLs failed - try creating a simple empty model for testing
        logger.warning("All download attempts failed, creating a simple test model")
        try:
            logger.info("Creating a minimal test model for debugging (won't provide real predictions)")
            from tensorflow import keras
            import numpy as np
            
            # Create a simple model matching the expected input/output
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(96, 64, 1)),
                keras.layers.Flatten(),
                keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            
            # Save the model
            model.save(str(model_path))
            logger.info(f"Created and saved test model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create test model: {e}")
            return False
    except Exception as e:
        logger.error(f"Error downloading sound classification model: {e}")
        logger.error(traceback.format_exc())
        return False

# Handle audio data
@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data"""
    try:
        # Log the received audio data event
        logger.info(f"Received audio_data from client, data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        
        # Extract audio data - client sends it with key 'data', not 'audio_data'
        if 'data' in data:
            audio_data = None
            
            # Check if data is base64-encoded string or a raw list
            if isinstance(data['data'], str):
                # Handle base64-encoded string
                try:
                    logger.info("Processing audio data as base64-encoded string")
                    audio_bytes = base64.b64decode(data['data'])
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Error decoding base64 audio data: {e}")
                    logger.error(traceback.format_exc())
                    return
            elif isinstance(data['data'], list):
                # Handle list directly - convert to numpy array
                try:
                    logger.info(f"Processing audio data as list (length: {len(data['data'])})")
                    audio_data = np.array(data['data'], dtype=np.float32)
                except Exception as e:
                    logger.error(f"Error converting list to numpy array: {e}")
                    logger.error(traceback.format_exc())
                    return
            else:
                logger.error(f"Unsupported data type: {type(data['data'])}")
                return
            
            # Check if audio data was successfully extracted
            if audio_data is None:
                logger.warning("Failed to extract audio data")
                return
            
            logger.info(f"Audio data extracted, shape: {audio_data.shape}, dtype: {audio_data.dtype}, min: {np.min(audio_data) if len(audio_data) > 0 else 'N/A'}, max: {np.max(audio_data) if len(audio_data) > 0 else 'N/A'}")
            
            # Check if audio data is empty or contains only zeros
            if len(audio_data) == 0:
                logger.warning("Audio data is empty")
                return
            if np.all(audio_data == 0):
                logger.warning("Audio data contains only zeros")
                return
            
            # Get sample rate from data or use default
            sample_rate = data.get('sample_rate', 16000)
            logger.debug(f"Audio sample rate: {sample_rate} Hz")
            
            # Get dB level from data - client sends it with key 'db'
            db_level = None
            if 'db' in data:
                try:
                    db_level = float(data['db'])
                    logger.debug(f"Audio dB level from client: {db_level}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid dB level format: {data.get('db')}")
            else:
                # Calculate dB level if not provided
                if len(audio_data) > 0:
                    db_level = dbFS(audio_data)
                    logger.debug(f"Calculated dB level: {db_level}")
                else:
                    logger.warning("Cannot calculate dB level for empty audio data")
                    return
            
            # Check if audio is too quiet for sentiment analysis
            if db_level is not None and db_level < DBLEVEL_THRES:
                logger.info(f"Audio too quiet (dB {db_level} < threshold {DBLEVEL_THRES}), skipping processing")
                return
        
            # Add audio data to continuous speech analysis thread (always process for sentiment analysis)
            if GOOGLE_SPEECH_AVAILABLE and ENABLE_SENTIMENT_ANALYSIS:
                logger.info(f"Adding audio data to speech analysis thread (length: {len(audio_data)})")
                add_audio_for_analysis(audio_data, sample_rate)
            else:
                logger.debug("Not adding audio to speech analysis thread: Google Speech API or sentiment analysis is disabled")
            
            # Process audio for sound classification (buffer until we have enough)
            with classification_buffer_lock:
                # Add new audio to the buffer
                classification_buffer.extend(audio_data)
                buffer_length = len(classification_buffer)
                
                logger.info(f"Classification buffer now has {buffer_length} samples (need {MIN_SAMPLES_FOR_CLASSIFICATION})")
                
                # If we have enough samples, process for classification
                if buffer_length >= MIN_SAMPLES_FOR_CLASSIFICATION:
                    # Take the latest MIN_SAMPLES_FOR_CLASSIFICATION samples
                    classification_data = np.array(classification_buffer[-MIN_SAMPLES_FOR_CLASSIFICATION:], dtype=np.float32)
                    logger.info(f"Processing {MIN_SAMPLES_FOR_CLASSIFICATION} samples for sound classification")
                    
                    # Process the data for sound classification
                    process_sound_classification(classification_data, sample_rate, db_level)
                    
                    # Clear the buffer (we could optionally keep some overlap)
                    classification_buffer.clear()
                    logger.info("Cleared classification buffer after processing")
        else:
            logger.warning("No 'data' field in received audio data message")
                
    except Exception as e:
        logger.error(f"Error handling audio data: {e}")
        logger.error(traceback.format_exc())

def process_sound_classification(audio_data, sample_rate, db_level=None):
    """Process audio data for sound classification"""
    global prediction_count
    
    try:
        # Calculate dB level if not provided
        if db_level is None:
            db_level = dbFS(audio_data)
            logger.info(f"Calculated dB level for sound classification: {db_level}")
        
        # Skip processing if audio is too quiet
        if db_level < DBLEVEL_THRES:
            logger.info(f"Audio too quiet for sound classification (dB {db_level} < threshold {DBLEVEL_THRES}), skipping processing")
            return
        
        # Perform sound classification with thread-safe model access
        with model_lock:
            # Ensure model is loaded
            if 'sound_model' not in models:
                logger.info("Sound model not loaded yet, loading models...")
                load_models()
                
            sound_model = models.get('sound_model')
            if sound_model is None:
                logger.error("Sound classification model not loaded")
                return
        
            # Log audio data statistics for debugging
            logger.info(f"Processing audio data: shape={audio_data.shape}, min={np.min(audio_data)}, max={np.max(audio_data)}, mean={np.mean(audio_data)}")
            
            # Prepare audio for model input (assuming 1-second window)
            logger.info(f"Computing features from audio data of length {len(audio_data)}")
            try:
                audio_features = homesounds.compute_features(audio_data, sample_rate)
                logger.info(f"Computed audio features with shape: {audio_features.shape}")
                
                # Add debug visualization and save
                try:
                    # Generate a filename based on timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_file = f"debug_features_{timestamp}.npy"
                    
                    # Save features to file for debugging
                    np.save(os.path.join('models', debug_file), audio_features)
                    logger.info(f"Saved debug features to {debug_file}")
                except Exception as viz_error:
                    logger.warning(f"Could not save debug features: {viz_error}")
            except Exception as e:
                logger.error(f"Error computing audio features: {e}")
                logger.error(traceback.format_exc())
                return
            
            # Ensure correct shape for model input
            if audio_features.shape != (1, 96, 64, 1):
                logger.warning(f"Reshaping audio features from {audio_features.shape} to (1, 96, 64, 1)")
                try:
                    audio_features = np.reshape(audio_features, (1, 96, 64, 1))
                except Exception as e:
                    logger.error(f"Error reshaping audio features: {e}")
                    logger.error(traceback.format_exc())
                    return
            
            # Make prediction
            logger.info("Making prediction with sound model...")
            try:
                prediction = sound_model.predict(audio_features, verbose=0)[0]
                logger.info(f"Prediction shape: {prediction.shape}, sum: {np.sum(prediction):.4f}")
                
                # Debug: print full prediction array
                logger.debug(f"Full prediction array: {prediction}")
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                logger.error(traceback.format_exc())
                return
                
            prediction_count += 1
            
            # Get class with highest probability
            max_idx = np.argmax(prediction)
            max_prob = prediction[max_idx]
            
            logger.info(f"Prediction result: max_idx={max_idx}, max_prob={max_prob:.4f}")
            
            # Skip if confidence is too low
            if max_prob < PREDICTION_THRES:
                logger.info(f"Prediction confidence {max_prob:.4f} below threshold {PREDICTION_THRES}, skipping notification")
                return
                
            # Get corresponding class name
            class_names = homesounds.get_class_names()
            if max_idx < len(class_names):
                sound_class = class_names[max_idx]
            else:
                logger.error(f"Invalid class index: {max_idx}, max index: {len(class_names)-1}")
                return
                
            # Format prediction data
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
            predicted_sound = {
                'sound': sound_class,
                'confidence': f"{max_prob:.2f}",
                'db_level': f"{db_level:.2f}",
                'time': formatted_time,
                'record_time': formatted_time
            }
            
            # Log prediction
            logger.info(f"Sound prediction {prediction_count}: {sound_class} ({max_prob:.2f}), dB: {db_level:.2f}")
            
            # Emit notification to clients
            logger.info(f"Emitting sound_notification to clients: {predicted_sound}")
            socketio.emit('sound_notification', predicted_sound)
            
            # Add to conversation history
            add_to_conversation_history(predicted_sound)
        
    except Exception as e:
        logger.error(f"Error processing sound classification: {e}")
        logger.error(traceback.format_exc())

@socketio.on('audio_feature_data')
def handle_audio_feature_data(data):
    """Handle incoming audio feature data from phone/watch client"""
    global prediction_count
    
    try:
        # Log received data for debugging
        logger.info(f"Received audio_feature_data from client, data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        
        # Check if data field is present
        if 'data' not in data:
            logger.warning("Missing 'data' field in audio feature data")
            return
        
        feature_data = None
        
        # Convert data to numpy array, handling different possible formats
        try:
            # Check if data is already a list
            if isinstance(data['data'], list):
                logger.info(f"Processing feature data as list (length: {len(data['data'])})")
                feature_data = np.array(data['data'], dtype=np.float32)
            elif isinstance(data['data'], str):
                # Try parsing as string (older client format)
                logger.info(f"Processing feature data as string")
                data_str = str(data['data'])
                data_str = data_str.strip('[]')  # Remove brackets if present
                feature_data = np.fromstring(data_str, dtype=np.float32, sep=',')
            else:
                # Try to convert directly if it's some other type
                logger.info(f"Attempting to convert feature data of type {type(data['data'])}")
                feature_data = np.array(data['data'], dtype=np.float32)
            
            logger.info(f"Converted feature data shape: {feature_data.shape}")
        except Exception as e:
            logger.error(f"Error converting feature data: {e}")
            logger.error(f"Data type: {type(data['data'])}")
            logger.error(f"Data preview: {str(data['data'])[:100]}..." if isinstance(data['data'], (str, list)) else "Not displayable")
            return
        
        # Reshape if necessary
        if len(feature_data.shape) == 1 or feature_data.shape != (1, 96, 64, 1):
            logger.info(f"Reshaping feature data from {feature_data.shape} to (1, 96, 64, 1)")
            try:
                # For flat array, reshape to expected dimensions
                if len(feature_data.shape) == 1:
                    if feature_data.size == 96*64:
                        feature_data = feature_data.reshape(1, 96, 64, 1)
                    else:
                        logger.warning(f"Unexpected feature data size: {feature_data.size}, expected {96*64}")
                        return
                else:
                    feature_data = np.reshape(feature_data, (1, 96, 64, 1))
            except Exception as e:
                logger.error(f"Error reshaping feature data: {e}")
                logger.error(traceback.format_exc())
                return
                
        # Get dB level if available
        db_level = None
        if 'db_level' in data:
            try:
                db_level = float(data['db_level'])
                logger.info(f"DB level from client: {db_level}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid dB level format: {data.get('db_level')}")
                db_level = -100  # Default low value
        elif 'db' in data:
            try:
                db_level = float(data['db'])
                logger.info(f"DB level from client: {db_level}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid dB level format: {data.get('db')}")
                db_level = -100  # Default low value
                
        # Process prediction with thread-safe model access
        with model_lock:
            # Ensure model is loaded
            if 'sound_model' not in models:
                logger.info("Sound model not loaded, loading now...")
                load_models()
                
            sound_model = models.get('sound_model')
            if sound_model is None:
                logger.error("Sound classification model not loaded")
                return
                
            # Make prediction
            logger.info("Making prediction with sound model...")
            try:
                prediction = sound_model.predict(feature_data, verbose=0)[0]
                prediction_count += 1
                
                logger.info(f"Prediction shape: {prediction.shape}, sum: {np.sum(prediction):.4f}")
                logger.debug(f"Full prediction array: {prediction}")
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                logger.error(traceback.format_exc())
                return
            
            # Get class with highest probability
            max_idx = np.argmax(prediction)
            max_prob = prediction[max_idx]
            
            logger.info(f"Prediction result: max_idx={max_idx}, max_prob={max_prob:.4f}")
            
            # Skip if confidence is too low
            if max_prob < PREDICTION_THRES:
                logger.info(f"Prediction confidence {max_prob:.4f} below threshold {PREDICTION_THRES}, skipping")
                return
                
            # Get corresponding class name
            class_names = homesounds.get_class_names()
            if max_idx < len(class_names):
                sound_class = class_names[max_idx]
            else:
                logger.error(f"Invalid class index: {max_idx}, max index: {len(class_names)-1}")
                return
                
            # Format prediction data
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
            predicted_sound = {
                'sound': sound_class,
                'confidence': f"{max_prob:.2f}",
                'db_level': f"{db_level:.2f}" if db_level is not None else "N/A",
                'time': formatted_time,
                'record_time': formatted_time
            }
            
            # Log prediction
            logger.info(f"Sound prediction {prediction_count}: {sound_class} ({max_prob:.2f}), dB: {db_level}")
            
            # Emit notification to clients
            logger.info(f"Emitting sound_notification to clients: {predicted_sound}")
            socketio.emit('sound_notification', predicted_sound)
            
            # Add to conversation history
            add_to_conversation_history(predicted_sound)
            
    except Exception as e:
        logger.error(f"Error handling audio feature data: {e}")
        logger.error(traceback.format_exc())

def add_to_conversation_history(entry):
    """Add an entry to the conversation history"""
    global conversation_history
    
    conversation_history.append(entry)
    
    # Limit history size
    while len(conversation_history) > MAX_CONVERSATION_HISTORY:
        conversation_history.pop(0)

@socketio.on('get_conversation_history')
def handle_get_conversation_history():
    """Send conversation history to client"""
    try:
        return {'history': conversation_history}
    except Exception as e:
        logger.error(f"Error handling get_conversation_history: {e}")
        return {'history': [], 'error': str(e)}

@socketio.on('get_transcript_history')
def handle_get_transcript_history():
    """Send transcript history to client"""
    try:
        return {'history': transcript_history}
    except Exception as e:
        logger.error(f"Error handling get_transcript_history: {e}")
        return {'history': [], 'error': str(e)}

@app.route('/')
def index():
    """Serve the main index page"""
    return render_template('index.html')

@app.route('/conversation-history')
def conversation_history():
    """Serve the conversation history page"""
    return send_from_directory('static', 'conversation-history.html')

@app.route('/api/conversation-history')
def api_conversation_history():
    """Return the conversation history as JSON"""
    return jsonify({'history': conversation_history})

@app.route('/api/transcript-history')
def api_transcript_history():
    """Return the transcript history as JSON"""
    return jsonify({'history': transcript_history})

@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'ok',
        'server_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'sound_model_loaded': 'sound_model' in models,
        'google_speech_available': GOOGLE_SPEECH_AVAILABLE,
        'sentiment_analysis_enabled': ENABLE_SENTIMENT_ANALYSIS
    })

def analyze_text_sentiment(text):
    """
    Analyze the sentiment of a text.
    
    Args:
        text: The text to analyze
        
    Returns:
        dict: Sentiment analysis result with sentiment information,
              or None if analysis failed
    """
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for sentiment analysis")
        return None
    
    try:
        # First try to use the dedicated sentiment_analyzer module if available
        if SENTIMENT_ANALYZER_AVAILABLE:
            logger.info(f"Using sentiment_analyzer module to analyze: '{text}'")
            try:
                result = analyze_sentiment(text)
                if result:
                    logger.info(f"Sentiment analysis result from sentiment_analyzer: {result}")
                    return result
                logger.warning("sentiment_analyzer returned None, falling back to pipeline")
            except Exception as e:
                logger.error(f"Error using sentiment_analyzer: {e}")
                logger.error(traceback.format_exc())
                # Fall back to pipeline approach
        
        # Fall back to the pipeline approach
        logger.info(f"Using pipeline to analyze sentiment for: '{text}'")
        # Get or initialize the sentiment pipeline
        with model_lock:
            sentiment_pipeline = None
            try:
                # See if we already have one in a thread
                if continuous_speech_thread and hasattr(continuous_speech_thread, 'sentiment_pipeline'):
                    sentiment_pipeline = continuous_speech_thread.sentiment_pipeline
                    
                # If not, try to initialize one
                if sentiment_pipeline is None:
                    sentiment_pipeline = pipeline("sentiment-analysis")
                    logger.info("Initialized a new sentiment pipeline")
                    
                # Analyze sentiment
                if sentiment_pipeline:
                    start_time = time.time()
                    pipeline_result = sentiment_pipeline(text)[0]
                    end_time = time.time()
                    
                    logger.info(f"Sentiment analysis completed in {(end_time - start_time)*1000:.2f} ms")
                    
                    # Map sentiment to positive/negative/neutral
                    sentiment = pipeline_result['label'].lower()
                    confidence = pipeline_result['score']
                    
                    # Map POSITIVE/NEGATIVE to positive/negative
                    if sentiment == 'positive' or sentiment == 'POSITIVE':
                        category = 'Happy'
                        sentiment = 'positive'
                    elif sentiment == 'negative' or sentiment == 'NEGATIVE':
                        category = 'Unpleasant'
                        sentiment = 'negative'
                    else:
                        category = 'Neutral'
                        sentiment = 'neutral'
                    
                    # Add emoji based on sentiment
                    emoji = "😊" if sentiment == 'positive' else "😐" if sentiment == 'neutral' else "😟"
                    
                    result = {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'emoji': emoji,
                        'category': category,
                        'original_emotion': pipeline_result['label']
                    }
                    
                    logger.info(f"Pipeline sentiment analysis result: {result}")
                    return result
                else:
                    logger.error("No sentiment pipeline available")
                    return None
            except Exception as e:
                logger.error(f"Error in pipeline sentiment analysis: {e}")
                logger.error(traceback.format_exc())
                return None
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == '__main__':
    # Parse command-line arguments for port configuration
    parser = argparse.ArgumentParser(description='SoundWatch Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run server in debug mode')
    parser.add_argument('--use-google-speech', action='store_true', default=True, help='Use Google Speech-to-Text API')
    args = parser.parse_args()
    
    # Check Google Speech API availability
    if args.use_google_speech:
        if GOOGLE_SPEECH_AVAILABLE:
            logger.info("Google Speech API is enabled and available")
        else:
            logger.warning("Google Speech API was requested but is not available - speech transcription will be disabled")
            logger.warning("Install with: pip install google-cloud-speech")
    
    # Check for sentiment analysis availability
    if ENABLE_SENTIMENT_ANALYSIS:
        try:
            from transformers import pipeline
            logger.info("Sentiment analysis is enabled and HuggingFace Transformers is available")
        except ImportError:
            logger.warning("Sentiment analysis is enabled but HuggingFace Transformers is not available")
            logger.warning("Install with: pip install transformers torch")
    
    # Load models
    logger.info("Loading required models...")
    if load_models():
        logger.info("Models loaded successfully")
    else:
        logger.error("Failed to load models. Starting server anyway...")
    
    # Start continuous speech analysis thread
    logger.info("Starting continuous speech analysis thread...")
    try:
        start_continuous_speech_analysis(socketio)
        logger.info("Continuous speech analysis thread started successfully")
    except Exception as e:
        logger.error(f"Failed to start continuous speech analysis thread: {e}")
        logger.error(traceback.format_exc())
    
    # Get IP addresses
    local_ips, public_ip = get_ip_addresses()
    
    # Print server information
    logger.info("="*50)
    logger.info(" SoundWatch Server with Parallel Sound and Sentiment Analysis")
    logger.info("="*50)
    logger.info(f"Server running on port {args.port}")
    
    if local_ips:
        logger.info("Local IPs: " + ", ".join(local_ips))
    if public_ip:
        logger.info(f"Public IP: {public_ip}")
    
    logger.info("Access the web interface at:")
    logger.info(f" → http://localhost:{args.port} (local)")
    
    if local_ips:
        logger.info(f" → http://{local_ips[0]}:{args.port} (internal network)")
    
    if public_ip:
        logger.info(f" → http://{public_ip}:{args.port} (external - Internet)")
    
    logger.info("="*50)
    
    # Start the server
    socketio.run(app, host='0.0.0.0', port=args.port, debug=args.debug)