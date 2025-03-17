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
import urllib3
from urllib3.util.retry import Retry
import requests
from requests.adapters import HTTPAdapter

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
    from google_speech import GoogleSpeechToText, transcribe_with_google, streaming_transcribe_with_google, AudioStream, close_streaming_transcription
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

##################################################
# Global variables for sound analysis
##################################################
# Detection settings - these are now defined in homesounds.py
# PREDICTION_THRES = 0.2  # Threshold to consider a sound as detected
# DBLEVEL_THRES = -30     # Minimum dB level to consider audio for sound analysis

# Sound-specific thresholds (overrides PREDICTION_THRES for specific sounds)
# Now defined in homesounds.py as sound_specific_thresholds

# Classification buffer for continuous processing
classification_buffer = []
classification_buffer_lock = threading.RLock()  # Use RLock for consistency
TARGET_SR = 16000  # Target sample rate (16kHz)
MAX_CLASSIFICATION_BUFFER_SIZE = TARGET_SR * 5  # 5 seconds max buffer

# Store the most recent audio data for debugging
last_audio_data = None
last_timestamp = None
last_audio_lock = threading.RLock()  # Use RLock for consistency

# Model variables
models = {}
model_lock = threading.RLock()  # Use RLock for consistency
prediction_count = 0
sound_preds = None  # Global variable to store sound prediction results

# Global variables for conversation history
MAX_CONVERSATION_HISTORY = 1000  # Maximum number of entries to keep
conversation_history = []  # List to store conversation history
transcript_history = []  # List to store transcription history

# Global variables for sentiment analysis
ENABLE_SENTIMENT_ANALYSIS = True  # Set to False to disable sentiment analysis
SPEECH_BIAS_CORRECTION = 0.15  # Correction factor for speech detection
APPLY_SPEECH_BIAS_CORRECTION = True  # Whether to apply speech bias correction

# Audio buffers for continuous processing
AUDIO_BUFFER_SIZE = 5  # Buffer size in seconds
audio_buffer = []  # Buffer to store audio data for continuous processing
audio_buffer_lock = threading.RLock()  # Lock for thread-safe access to audio buffer

# Enhanced audio buffer for classification
MIN_SAMPLES_FOR_CLASSIFICATION = 16000  # Minimum samples needed for classification

# New buffer for speech transcription
transcription_buffer = []  # Buffer specifically for speech transcription
transcription_buffer_lock = threading.RLock()
MIN_SAMPLES_FOR_TRANSCRIPTION = 15000  # Minimum samples needed for transcription

# At top with other constants
SOUND_SPECIFIC_THRESHOLDS = homesounds.sound_specific_thresholds  # Centralized reference

# Increase connection pool size to avoid "Connection pool is full" warnings
# Configure larger connection pools for urllib3
urllib3.connectionpool.DEFAULT_POOLSIZE = 20  # Default is 10
urllib3.connectionpool.DEFAULT_RETRIES = 3   # Default is 0

# Configure larger connection pools for requests library
session = requests.Session()
adapter = HTTPAdapter(pool_connections=20,   # Default is 10
                     pool_maxsize=20,       # Default is 10 
                     max_retries=Retry(total=3, backoff_factor=0.5))
session.mount('http://', adapter)
session.mount('https://', adapter)

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
    """Thread for continuous sentiment analysis of speech"""
    def __init__(self, socketio, nlp_pipeline=None):
        """Initialize the thread"""
        threading.Thread.__init__(self)
        self.daemon = True
        self.running = True
        self.socketio = socketio
        self.queue = queue.Queue()
        self.transcript = ""
        self.streaming_thread = None
        self.streaming_active = False
        
        # Set up sentiment analysis
        if ENABLE_SENTIMENT_ANALYSIS:
            try:
                from transformers import pipeline
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                logger.info("Successfully loaded sentiment analysis pipeline")
            except Exception as e:
                logger.error(f"Error loading sentiment analysis pipeline: {e}")
                self.sentiment_pipeline = None
        else:
            self.sentiment_pipeline = None
    
    def add_audio_data(self, audio):
        """Add audio data to the queue for processing"""
        if not self.running or audio is None:
            return

        # For debugging
        logger.debug(f"Adding audio data to analysis thread, shape: {audio.shape if isinstance(audio, np.ndarray) else 'unknown'}")
        
        # Start streaming transcription if enabled and not already started
        if GOOGLE_SPEECH_AVAILABLE and ENABLE_SENTIMENT_ANALYSIS:
            if not self.streaming_active and hasattr(self, 'start_streaming_transcription'):
                logger.info("Starting streaming transcription thread")
                self.start_streaming_transcription()
            
            # Add to streaming queue if available
            if self.streaming_active and hasattr(self, 'streaming_thread') and hasattr(self.streaming_thread, 'audio_stream'):
                try:
                    # Add the audio data to the streaming queue
                    self.streaming_thread.audio_stream.put(audio)
                    logger.debug(f"Added audio to streaming queue, size: {len(audio) if hasattr(audio, '__len__') else 'unknown'}")
                except Exception as e:
                    logger.error(f"Error adding audio to streaming queue: {e}")
            else:
                # Add to main queue if streaming not ready
                self.queue.put(audio)
        else:
            # Add to main queue for regular processing
            self.queue.put(audio)
    
    def start_streaming_transcription(self):
        """Start the streaming transcription in a separate thread"""
        if self.streaming_active:
            logger.warning("Streaming transcription already active")
            return
        
        # Create a shared audio stream queue
        audio_stream = queue.Queue()
            
        def streaming_thread_func(audio_stream):
            try:
                audio_stream_closed = False
                
                # Create a generator to yield audio chunks for streaming
                def audio_generator():
                    from google.cloud import speech
                    
                    try:
                        # No need to send initial config request anymore as we're passing it separately
                        logger.debug("Starting to process audio chunks")
                        while not audio_stream_closed:
                            # Wait for audio data with timeout
                            try:
                                audio_chunk = audio_stream.get(timeout=2.0)
                                if audio_chunk is None:
                                    logger.debug("Received None chunk, ending stream")
                                    break  # End of stream
                                    
                                # Convert audio to int16 format
                                if isinstance(audio_chunk, np.ndarray):
                                    # Check if chunk is valid
                                    if len(audio_chunk) == 0:
                                        logger.warning("Skipping empty audio chunk")
                                        continue
                                    
                                    # Normalize if needed
                                    if np.max(np.abs(audio_chunk)) > 0:
                                        audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                                        
                                    # Convert to int16
                                    audio_chunk_int16 = (audio_chunk * 32767).astype(np.int16)
                                    audio_bytes = audio_chunk_int16.tobytes()
                                    
                                    if len(audio_bytes) > 0:
                                        # Yield audio content in a proper request format
                                        logger.debug(f"Sending audio chunk of size {len(audio_bytes)} bytes")
                                        yield speech.StreamingRecognizeRequest(
                                            audio_content=audio_bytes
                                        )
                                    else:
                                        logger.warning("Skipping empty audio bytes")
                            except queue.Empty:
                                # No audio data available, continue waiting
                                logger.debug("No audio data in queue, continuing to wait...")
                                continue
                    except Exception as e:
                        logger.error(f"Error in audio generator: {e}")
                        logger.error(traceback.format_exc())
                
                # Initialize the Google Speech client
                try:
                    from google.cloud import speech
                    client = speech.SpeechClient()
                    logger.info("Successfully created Speech client for streaming")
                    
                    # Start streaming recognition with the audio generator
                    logger.info("Starting streaming transcription...")
                    
                    # Create the streaming configuration
                    streaming_config = speech.StreamingRecognitionConfig(
                        config=speech.RecognitionConfig(
                            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=16000,
                            language_code="en-US",
                            model="phone_call",
                            enable_automatic_punctuation=True,
                            use_enhanced=True,
                        ),
                        interim_results=True
                    )
                    
                    # Pass both streaming_config and requests to the method
                    responses = client.streaming_recognize(config=streaming_config, requests=audio_generator())
                    
                    # Process streaming responses
                    for response in responses:
                        if not self.running:
                            break
                            
                        # Process response results
                        if not response.results:
                            continue
                            
                        for result in response.results:
                            if not result.alternatives:
                                continue
                                
                            transcript = result.alternatives[0].transcript
                            is_final = result.is_final
                            
                            if is_final and len(transcript.strip()) > 0:
                                # For final results, add to transcript and history
                                logger.info(f"Final streaming transcript: '{transcript}'")
                                
                                # Update the current transcript
                                self.transcript = transcript
                                
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
                                
                                # Also emit a notification to the client
                                self.socketio.emit('notification', {
                                    'type': 'transcription',
                                    'title': 'Speech Detected',
                                    'message': transcript,
                                    'timestamp': timestamp
                                })
                                
                                # Analyze sentiment immediately for final transcripts
                                if ENABLE_SENTIMENT_ANALYSIS and self.sentiment_pipeline:
                                    self.analyze_sentiment()
                                
                            elif not is_final and len(transcript.strip()) > 0:
                                # For interim results, just emit updates
                                logger.debug(f"Interim streaming transcript: '{transcript}'")
                                
                                # Emit interim transcript update
                                self.socketio.emit('interim_transcript', {
                                    'transcript': transcript,
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                    
                    logger.info("Streaming recognition completed normally")
                    
                except Exception as e:
                    logger.error(f"Error in streaming transcription: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    # Mark stream as closed
                    audio_stream_closed = True
                    logger.info("Streaming transcription thread stopped")
                    self.streaming_active = False
                
            except Exception as e:
                logger.error(f"Error in streaming transcription thread: {e}")
                logger.error(traceback.format_exc())
            finally:
                logger.info("Streaming transcription thread stopped")
                self.streaming_active = False
        
        # Create a streaming thread with accessible audio_stream attribute
        streaming_thread = threading.Thread(
            target=streaming_thread_func, 
            args=(audio_stream,),
            daemon=True
        )
        
        # Make audio stream accessible from thread
        streaming_thread.audio_stream = audio_stream
        
        # Start the thread
        streaming_thread.start()
        self.streaming_thread = streaming_thread
        self.streaming_active = True
        logger.info("Streaming transcription thread started")
    
    def run(self):
        """Main thread loop for continuous speech analysis"""
        logger.info("Starting continuous speech analysis thread")
        
        while self.running:
            try:
                # Process audio data from the queue if available
                try:
                    audio_data = self.queue.get(timeout=0.5)
                    logger.debug(f"Got audio data from queue, length: {len(audio_data)}")
                    
                    # Process the audio data for transcription if not using streaming
                    if not GOOGLE_SPEECH_AVAILABLE or not ENABLE_SENTIMENT_ANALYSIS or not self.streaming_active:
                        # In this implementation we don't need to process non-streaming data
                        # as everything is handled by the streaming thread
                        logger.debug("Skipping queue processing as streaming is active")
                        
                except queue.Empty:
                    # No data in queue, continue to next iteration
                    continue
                
                # Periodically analyze sentiment
                self.analyze_sentiment()
                
            except Exception as e:
                logger.error(f"Error in continuous speech analysis thread: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1.0)  # Avoid tight loop on error
    
    def process_audio(self, audio_data, sample_rate):
        """Process individual audio segments (now primarily for logging)"""
        if not GOOGLE_SPEECH_AVAILABLE:
            logger.warning("Google Speech API not available, skipping transcription")
            return
        
        # We're using streaming transcription now, so this method is just for logging
        logger.debug(f"Audio segment of length {len(audio_data)} received (streaming handles transcription)")
    
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
        """Stop the continuous analysis thread and streaming transcription"""
        logger.info("Stopping continuous speech analysis thread")
        self.running = False
        
        # Close streaming transcription
        if self.streaming_active:
            try:
                close_streaming_transcription()
                if self.streaming_thread and self.streaming_thread.is_alive():
                    self.streaming_thread.join(timeout=3)
            except Exception as e:
                logger.error(f"Error stopping streaming transcription: {e}")
        
        logger.info("Continuous speech analysis thread and streaming stopped")

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
        continuous_speech_thread.add_audio_data(audio_data)

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
def handle_audio_data_socketio(data):
    """Handle incoming audio data via Socket.IO"""
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
            
            # Store the last audio data and timestamp for debugging
            with last_audio_lock:
                last_audio_data = audio_data
                last_timestamp = time.time()
                
            # Process the audio using our new comprehensive function
            process_audio_data(audio_data, sample_rate)
            
        else:
            logger.warning("No 'data' field in received audio data")
            
    except Exception as e:
        logger.error(f"Error handling audio data: {e}")
        logger.error(traceback.format_exc())

def process_sound_classification(audio_data, sample_rate, best_pred_confidence, best_pred_class, percussive_event, dB):
    """Process sound classification results with improved handling for knocking sounds"""
    
    # Get class names for logging
    class_names = homesounds.get_class_names()
    
    # Map of sound indices for easy reference
    KNOCK_IDX = homesounds.KNOCK_IDX  # Door knock index
    DISHES_IDX = homesounds.DISHES_IDX  # Dishes index
    GLASS_BREAKING_IDX = homesounds.GLASS_BREAKING_IDX  # Glass breaking index
    
    # Adjust confidence for certain sound classes based on audio features
    if best_pred_class is not None:
        # Original confidence for logging
        original_confidence = best_pred_confidence
        original_class = best_pred_class
        
        # Adjust for percussive sounds if detected in audio
        if percussive_event:
            logger.info(f"Percussive event detected")
            
            # Door knock - boost confidence if it's already the best prediction
            if best_pred_class == KNOCK_IDX and best_pred_confidence > 0.05:  # Lower activation threshold (was 0.1)
                best_pred_confidence = min(1.0, best_pred_confidence * 2.0)  # Double confidence (was *1.3)
                logger.info(f"Boosted Door knock confidence: {original_confidence:.4f} -> {best_pred_confidence:.4f}")
            
            # Dishes is often confused with knocking 
            elif best_pred_class == DISHES_IDX and best_pred_confidence > 0.05:
                # Check if Door knock has non-zero confidence in the prediction
                knock_confidence = sound_preds[0][KNOCK_IDX]
                if knock_confidence > 0.01:  # Very low threshold to capture any hint of knocking
                    # Transfer some confidence from Dishes to Door knock
                    transfer_amount = best_pred_confidence * 0.4  # Transfer 40% (was 20%)
                    sound_preds[0][KNOCK_IDX] = knock_confidence + transfer_amount
                    sound_preds[0][DISHES_IDX] = best_pred_confidence - transfer_amount
                    
                    # Recalculate best prediction
                    best_pred_class = np.argmax(sound_preds[0])
                    best_pred_confidence = sound_preds[0][best_pred_class]
                    
                    logger.info(f"Transferred confidence from Dishes to Door knock: {original_confidence:.4f} -> {best_pred_confidence:.4f}, class: {class_names[original_class]} -> {class_names[best_pred_class]}")
                    
            # Handle case where percussive event is detected but confidence is distributed among classes
            # If Door knock has any non-trivial confidence, boost it 
            elif sound_preds[0][KNOCK_IDX] > 0.03:  # Lower threshold for activation (was 0.05)
                original_knock_confidence = sound_preds[0][KNOCK_IDX]
                sound_preds[0][KNOCK_IDX] = min(1.0, sound_preds[0][KNOCK_IDX] * 2.0)  # Double confidence (was 1.5)
                
                # Recalculate best prediction
                best_pred_class = np.argmax(sound_preds[0])
                best_pred_confidence = sound_preds[0][best_pred_class]
                
                logger.info(f"Boosted Door knock among distributed confidence: {original_knock_confidence:.4f} -> {sound_preds[0][KNOCK_IDX]:.4f}, new best: {class_names[best_pred_class]} ({best_pred_confidence:.4f})")
            
            # False negative check: if we detected a strong percussive event but didn't classify as knock or dishes
            elif sound_preds[0][KNOCK_IDX] > 0.01:  # Any non-zero confidence for Door knock
                # A clear percussive event with even a tiny hint of Door knock suggests it might be a knock
                sound_preds[0][KNOCK_IDX] = max(0.25, sound_preds[0][KNOCK_IDX] * 5.0)  # Significant boost for missed knocks
                
                # Recalculate best prediction
                best_pred_class = np.argmax(sound_preds[0])
                best_pred_confidence = sound_preds[0][best_pred_class]
                
                logger.info(f"Captured potential missed knock: door confidence {sound_preds[0][KNOCK_IDX]:.4f}, new best: {class_names[best_pred_class]} ({best_pred_confidence:.4f})")
    
    # Apply sound-specific thresholds
    if best_pred_class is not None:
        # Check if this sound has a custom threshold
        sound_thresh = homesounds.sound_specific_thresholds.get(class_names[best_pred_class], homesounds.PREDICTION_THRES)
        
        if best_pred_confidence >= sound_thresh and dB >= homesounds.DBLEVEL_THRES:
            logger.info(f"Sound detected above threshold: {class_names[best_pred_class]} ({best_pred_confidence:.4f} >= {sound_thresh:.2f}, {dB:.1f} dB)")
            return best_pred_class, best_pred_confidence
        else:
            if best_pred_confidence < sound_thresh:
                logger.info(f"Sound below confidence threshold: {class_names[best_pred_class]} ({best_pred_confidence:.4f} < {sound_thresh:.2f})")
            if dB < homesounds.DBLEVEL_THRES:
                logger.info(f"Sound below dB threshold: {dB:.1f} dB < {homesounds.DBLEVEL_THRES:.1f} dB")
    
    return None, 0.0

@socketio.on('audio_feature_data')
def handle_audio_feature_data(data):
    """Handle incoming audio feature data from phone/watch client"""
    global sound_preds
    
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
                
        # Skip if audio level is too low
        if db_level is not None and db_level < homesounds.DBLEVEL_THRES:
            logger.info(f"Audio level too low: {db_level} dB < threshold {homesounds.DBLEVEL_THRES} dB")
            return
                
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
                sound_preds = sound_model.predict(feature_data, verbose=0)
                logger.info(f"Prediction shape: {sound_preds.shape}, sum: {np.sum(sound_preds[0]):.4f}")
                
                # Find best prediction
                best_pred_class = np.argmax(sound_preds[0])
                best_pred_confidence = sound_preds[0][best_pred_class]
                
                # Debug: log top 3 predictions
                class_names = homesounds.get_class_names()
                top_indices = np.argsort(sound_preds[0])[-3:][::-1]
                top_preds = [(class_names[i], sound_preds[0][i]) for i in top_indices]
                logger.info(f"Top 3 predictions: {top_preds}")
                
                # Check for percussive sound - since we don't have raw audio, use a simplified check
                # based on the prediction confidence for percussive sounds
                is_percussive = False
                percussive_sounds = homesounds.percussive_sounds
                for sound_class_idx in percussive_sounds:
                    if sound_preds[0][sound_class_idx] > 0.05:  # Low threshold to detect any hint of percussive sounds
                        is_percussive = True
                        logger.info(f"Detected potential percussive sound: {class_names[sound_class_idx]} ({sound_preds[0][sound_class_idx]:.4f})")
                        break
                
                # Process sound classification with our enhanced logic
                best_class_idx, best_confidence = process_sound_classification(
                    None, None, best_pred_confidence, best_pred_class, is_percussive, db_level
                )
                
                # If we have a valid detection above threshold, send notification
                if best_class_idx is not None and best_confidence > 0:
                    # Format prediction data
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    predicted_sound = {
                        'sound': class_names[best_class_idx],
                        'confidence': f"{best_confidence:.2f}",
                        'db_level': f"{db_level:.2f}" if db_level is not None else "N/A",
                        'time': formatted_time,
                        'record_time': formatted_time
                    }
                    
                    # Log prediction
                    logger.info(f"Sound detected: {predicted_sound['sound']} ({best_confidence:.2f}), dB: {db_level:.2f if db_level is not None else float('nan')}")
                    
                    # Emit notification to clients
                    logger.info(f"Emitting sound_notification to clients: {predicted_sound}")
                    socketio.emit('sound_notification', predicted_sound)
                    
                    # Add to conversation history
                    add_to_conversation_history(predicted_sound)
            
            except Exception as e:
                logger.error(f"Error in sound prediction: {e}")
                logger.error(traceback.format_exc())
            
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

@app.route('/api/audio', methods=['POST'])
def handle_audio_data():
    global last_audio_data, last_timestamp
    """Handles the incoming audio data from the client"""
    try:
        # Extract audio data from request, which could be a base64-encoded string or raw list
        if request.json and 'audio' in request.json:
            audio_data = request.json['audio']
            # Check if it's base64 encoded
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            elif isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            else:
                logger.warning(f"Received unexpected audio data type: {type(audio_data)}")
                return jsonify({"status": "error", "message": "Invalid audio data format"})
        else:
            logger.warning("No audio data received in request")
            return jsonify({"status": "error", "message": "No audio data received"})

        # Get the sample rate if provided, otherwise use default
        sample_rate = request.json.get('sampleRate', TARGET_SR)
        
        # Log information about audio data
        logger.debug(f"Received audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}, sample rate: {sample_rate}")
        
        # Check if audio data is valid
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Received empty audio data")
            return jsonify({"status": "error", "message": "Empty audio data"})
            
        # Calculate dB level of audio
        if np.max(np.abs(audio_data)) > 0:
            db_level = 20 * np.log10(np.mean(np.abs(audio_data)))
        else:
            db_level = -100  # Arbitrary low value for silence
            
        logger.debug(f"Audio dB level: {db_level:.2f}")
        
        # Skip very quiet audio for classification (but still process for speech if enabled)
        if db_level < DBLEVEL_THRES:
            logger.debug(f"Audio is somewhat quiet (below {DBLEVEL_THRES} dB): {db_level:.2f} dB")
            
            # Even if audio is quiet, we'll still process it for speech if Google Speech API is enabled
            if GOOGLE_SPEECH_AVAILABLE and ENABLE_SENTIMENT_ANALYSIS and continuous_speech_thread is not None:
                logger.debug(f"Still processing quiet audio for speech recognition, length: {len(audio_data)}")
                continuous_speech_thread.add_audio_data(audio_data)
                
            return jsonify({"status": "skipped", "message": "Audio level too low", "db_level": db_level})
            
        # Store the last audio data and timestamp for debugging
        with last_audio_lock:
            last_audio_data = audio_data
            last_timestamp = time.time()
            
        # Process audio for speech recognition
        if GOOGLE_SPEECH_AVAILABLE and ENABLE_SENTIMENT_ANALYSIS and continuous_speech_thread is not None:
            logger.debug(f"Adding audio to speech analysis thread, length: {len(audio_data)}")
            continuous_speech_thread.add_audio_data(audio_data)
            
        # Process audio for sound classification
        with classification_buffer_lock:
            # Extend the classification buffer with the new audio data
            if len(classification_buffer) < MAX_CLASSIFICATION_BUFFER_SIZE:
                classification_buffer.extend(audio_data.tolist())
                logger.debug(f"Extended classification buffer, now {len(classification_buffer)} samples")
                
            # If we have enough samples for classification, process it
            if len(classification_buffer) >= TARGET_SR:
                # Process the audio for classification
                logger.debug(f"Processing classification buffer of size {len(classification_buffer)}")
                process_sound_classification(np.array(classification_buffer[:TARGET_SR], dtype=np.float32), TARGET_SR)
                
                # Reset the buffer, keeping any overflow
                if len(classification_buffer) > TARGET_SR:
                    classification_buffer[:] = classification_buffer[TARGET_SR:]
                    logger.debug(f"Resetted classification buffer, kept {len(classification_buffer)} overflow samples")
                else:
                    classification_buffer.clear()
                    logger.debug("Cleared classification buffer")
        
        return jsonify({"status": "success", "message": "Audio processed"})
    except Exception as e:
        logger.error(f"Error handling audio data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

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

def process_audio_data(audio_data, sample_rate=16000):
    """Process incoming audio data for sound classification and speech recognition"""
    global is_first_audio_packet, active_transcription, sound_preds
    
    try:
        # Check if we have audio data
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Received empty audio data, skipping processing")
            return
            
        # Check audio data shape and properties
        logger.info(f"Processing audio data: shape={audio_data.shape}, sr={sample_rate}")
        
        # Calculate dB level for volume-based decisions
        dB = dbFS(audio_data)
        logger.info(f"Audio level: {dB:.2f} dB")
        
        # Skip processing if audio is too quiet
        if dB < homesounds.DBLEVEL_THRES:
            logger.info(f"Audio too quiet for processing (dB {dB:.2f} < threshold {homesounds.DBLEVEL_THRES}), skipping")
            return
            
        # Check for percussive events like knocking
        is_percussive = homesounds.detect_percussive_event(audio_data, sample_rate, threshold=0.2)
        
        # Perform sound classification
        with model_lock:
            # Ensure model is loaded
            if 'sound_model' not in models:
                logger.info("Sound model not loaded yet, loading models...")
                load_models()
                
            sound_model = models.get('sound_model')
            if sound_model is None:
                logger.error("Sound classification model not loaded")
                return
            
            # Prepare audio features for model input
            try:
                audio_features = homesounds.compute_features(audio_data, sample_rate)
                logger.info(f"Computed audio features with shape: {audio_features.shape}")
            except Exception as e:
                logger.error(f"Error computing audio features: {e}")
                logger.error(traceback.format_exc())
                return
            
            # Make prediction
            logger.info("Making prediction with sound model...")
            try:
                sound_preds = sound_model.predict(audio_features, verbose=0)
                logger.info(f"Prediction shape: {sound_preds.shape}, sum: {np.sum(sound_preds[0]):.4f}")
                
                # Find best prediction
                best_pred_class = np.argmax(sound_preds[0])
                best_pred_confidence = sound_preds[0][best_pred_class]
                
                # Debug: log top 3 predictions
                class_names = homesounds.get_class_names()
                top_indices = np.argsort(sound_preds[0])[-3:][::-1]
                top_preds = [(class_names[i], sound_preds[0][i]) for i in top_indices]
                logger.info(f"Top 3 predictions: {top_preds}")
                
                # Process sound classification with our enhanced logic
                best_class_idx, best_confidence = process_sound_classification(
                    audio_data, sample_rate, best_pred_confidence, best_pred_class, is_percussive, dB
                )
                
                # If we have a valid detection above threshold, send notification
                if best_class_idx is not None and best_confidence > 0:
                    # Format prediction data
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    predicted_sound = {
                        'sound': class_names[best_class_idx],
                        'confidence': f"{best_confidence:.2f}",
                        'db_level': f"{dB:.2f}",
                        'time': formatted_time,
                        'record_time': formatted_time
                    }
                    
                    # Log prediction
                    logger.info(f"Sound detected: {predicted_sound['sound']} ({best_confidence:.2f}), dB: {dB:.2f}")
                    
                    # Emit notification to clients
                    logger.info(f"Emitting sound_notification to clients: {predicted_sound}")
                    socketio.emit('sound_notification', predicted_sound)
                    
                    # Add to conversation history
                    add_to_conversation_history(predicted_sound)
            
            except Exception as e:
                logger.error(f"Error in sound prediction: {e}")
                logger.error(traceback.format_exc())
            
        # Process for speech recognition if enabled and audio is not too quiet
        if ENABLE_SPEECH and dB >= SPEECH_DBLEVEL_THRES:
            # Add to speech buffer
            speech_buffer.extend(audio_data)
            
            # Process speech in buffer if we have enough data
            if len(speech_buffer) >= SPEECH_BUFFER_SIZE:
                process_speech()
                
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        logger.error(traceback.format_exc())

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