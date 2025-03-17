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
from continuous_sentiment_analysis import initialize_sentiment_analyzer, get_sentiment_analyzer, analyze_text_sentiment

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
                self.sentiment_pipeline = None
    
    def add_audio_data(self, audio_data, sample_rate):
        """Add audio data to the processing queue"""
        self.audio_queue.put((audio_data, sample_rate))
    
    def run(self):
        """Main thread loop for continuous speech analysis"""
        logger.info("Starting continuous speech analysis thread")
        
        while self.running:
            try:
                # Process audio data from the queue if available
                try:
                    audio_data, sample_rate = self.audio_queue.get(timeout=0.5)
                    self.process_audio(audio_data, sample_rate)
                except queue.Empty:
                    # No data in queue, continue to next iteration
                    continue
                
                # Check if it's time to analyze sentiment
                current_time = time.time()
                if current_time - self.last_sentiment_time >= self.sentiment_interval and self.transcript:
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
            return
            
        try:
            # Transcribe the audio using Google Speech API
            transcript = transcribe_with_google(audio_data, sample_rate)
            
            if transcript:
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
                self.socketio.emit('transcript_update', {
                    'transcript': transcript,
                    'timestamp': timestamp
                })
                
                logger.info(f"Transcribed: {transcript}")
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
    
    def analyze_sentiment(self):
        """Analyze sentiment of the current transcript"""
        if not self.transcript or not self.sentiment_pipeline:
            return
            
        try:
            # Use the dedicated sentiment analysis function
            sentiment_result = analyze_text_sentiment(self.transcript)
            
            # Update the most recent transcript entry with sentiment
            if transcript_history and sentiment_result:
                transcript_history[-1]["sentiment"] = sentiment_result
            
            # Emit sentiment notification to clients
            sentiment_notification = {
                'text': self.transcript,
                'sentiment': sentiment_result,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.socketio.emit('sentiment_notification', sentiment_notification)
            
            # Clear transcript after analysis for fresh start
            self.transcript = ""
            
            logger.info(f"Sentiment analysis result: {sentiment_result}")
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
    
    def stop(self):
        """Stop the continuous analysis thread"""
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
            # Load sound classification model
            load_sound_classification_model()
            
            # Initialize sentiment analysis model
            initialize_sentiment_analyzer(socketio, sample_rate=16000)
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())

def load_sound_classification_model():
    """Load the sound classification model"""
    global models
    
    try:
        # Check if model already loaded
        if 'sound_model' in models and models['sound_model'] is not None:
            logger.info("Sound classification model already loaded")
            return
            
        logger.info("Loading sound classification model...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Set model paths
        model_dir = Path('models')
        model_path = model_dir / 'homesounds_model.h5'
        
        # Download model if it doesn't exist
        if not model_path.exists():
            logger.info("Downloading sound classification model...")
            download_sound_classification_model(model_path)
        
        # Load the model
        model = keras.models.load_model(str(model_path))
        models['sound_model'] = model
        
        logger.info("Sound classification model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading sound classification model: {e}")
        logger.error(traceback.format_exc())
        raise

def download_sound_classification_model(model_path):
    """Download the sound classification model"""
    try:
        # URL for the model
        model_url = "https://github.com/SmartWatchProject/SoundWatch/raw/master/models/homesounds_model.h5"
        
        # Download the model
        logger.info(f"Downloading model from {model_url}")
        wget.download(model_url, str(model_path))
        logger.info(f"Model downloaded to {model_path}")
        
    except Exception as e:
        logger.error(f"Error downloading sound classification model: {e}")
        logger.error(traceback.format_exc())
        raise

# Handle audio data
@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data"""
    try:
        if 'audio_data' in data:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(data['audio_data'])
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Get sample rate from data or use default
            sample_rate = data.get('sample_rate', 16000)
            
            # Get dB level from data
            db_level = None
            if 'db_level' in data:
                try:
                    db_level = float(data['db_level'])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid dB level format: {data.get('db_level')}")
            
            # Add audio data to continuous speech analysis thread
            add_audio_for_analysis(audio_data, sample_rate)
            
            # Process audio for sound classification (in parallel)
            if len(audio_data) >= 16000:  # Ensure enough audio data for classification
                process_sound_classification(audio_data, sample_rate, db_level)
                
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
        
        # Skip processing if audio is too quiet
        if db_level < DBLEVEL_THRES:
            return
        
        # Perform sound classification with thread-safe model access
        with model_lock:
            # Ensure model is loaded
            if 'sound_model' not in models:
                load_models()
                
            sound_model = models.get('sound_model')
            if sound_model is None:
                logger.error("Sound classification model not loaded")
                return
            
            # Prepare audio for model input (assuming 1-second window)
            audio_features = homesounds.compute_features(audio_data, sample_rate)
            
            # Ensure correct shape for model input
            if audio_features.shape != (1, 96, 64, 1):
                logger.warning(f"Unexpected audio feature shape: {audio_features.shape}, reshaping...")
                audio_features = np.reshape(audio_features, (1, 96, 64, 1))
            
            # Make prediction
            prediction = sound_model.predict(audio_features, verbose=0)[0]
            prediction_count += 1
            
            # Get class with highest probability
            max_idx = np.argmax(prediction)
            max_prob = prediction[max_idx]
            
            # Skip if confidence is too low
            if max_prob < PREDICTION_THRES:
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
        # Check if data field is present
        if 'data' not in data:
            logger.warning("Missing 'data' field in audio feature data")
            return
            
        # Convert data to numpy array
        feature_data = np.array(data['data'], dtype=np.float32)
        
        # Reshape if necessary
        if feature_data.shape != (1, 96, 64, 1):
            logger.warning(f"Unexpected audio feature shape: {feature_data.shape}, reshaping...")
            feature_data = np.reshape(feature_data, (1, 96, 64, 1))
            
        # Get dB level if available
        db_level = None
        if 'db_level' in data:
            try:
                db_level = float(data['db_level'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid dB level format: {data.get('db_level')}")
                db_level = -100  # Default low value
                
        # Process prediction with thread-safe model access
        with model_lock:
            # Ensure model is loaded
            if 'sound_model' not in models:
                load_models()
                
            sound_model = models.get('sound_model')
            if sound_model is None:
                logger.error("Sound classification model not loaded")
                return
                
            # Make prediction
            prediction = sound_model.predict(feature_data, verbose=0)[0]
            prediction_count += 1
            
            # Get class with highest probability
            max_idx = np.argmax(prediction)
            max_prob = prediction[max_idx]
            
            # Skip if confidence is too low
            if max_prob < PREDICTION_THRES:
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
            logger.info(f"Feature prediction {prediction_count}: {sound_class} ({max_prob:.2f}), dB: {db_level}")
            
            # Emit notification to clients
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

if __name__ == '__main__':
    # Parse command-line arguments for port configuration
    parser = argparse.ArgumentParser(description='SoundWatch Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run server in debug mode')
    parser.add_argument('--use-google-speech', action='store_true', default=True, help='Use Google Speech-to-Text API')
    args = parser.parse_args()
    
    # Load models
    if load_models():
        logger.info("Models loaded successfully")
    else:
        logger.error("Failed to load models. Starting server anyway...")
    
    # Start continuous speech analysis thread
    start_continuous_speech_analysis(socketio)
    
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