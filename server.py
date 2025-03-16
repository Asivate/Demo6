from threading import Lock
from flask import Flask, render_template, session, request, copy_current_request_context, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect
import tensorflow as tf
from tensorflow import keras
import numpy as np
from vggish_input import waveform_to_examples
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

# Import Vosk speech recognition
try:
    from vosk_speech import transcribe_with_vosk, VOSK_AVAILABLE
except ImportError:
    VOSK_AVAILABLE = False
    print("WARNING: Vosk speech recognition not available")

# Import continuous sentiment analyzer
from continuous_sentiment_analysis import initialize_sentiment_analyzer, get_sentiment_analyzer

# Thread-safe model access
model_lock = threading.RLock()
models = {}  # Dictionary to store loaded models
prediction_count = 0  # Counter for predictions

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

# Add debug mode flag
DEBUG_MODE = True  # Enable debug mode by default

# Add debug logging decorator
def debug_log(func):
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} with result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

# Add timing decorator for performance debugging
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

# Import our AST model implementation
# import ast_model

# Global flags for speech recognition
USE_GOOGLE_SPEECH_API = True  # Set to False to disable Google Speech API
SPEECH_ENGINE_AUTO = 0  # Automatically select the best engine
SPEECH_ENGINE_GOOGLE = 1  # Force Google Speech API
SPEECH_ENGINE_VOSK = 2  # Force Vosk (offline)
CURRENT_SPEECH_ENGINE = SPEECH_ENGINE_AUTO  # Default to auto selection

# Global variables for conversation history
MAX_CONVERSATION_HISTORY = 1000  # Maximum number of entries to keep
conversation_history = []  # List to store conversation history

# Global variables for sentiment analysis
ENABLE_SENTIMENT_ANALYSIS = True  # Set to False to disable sentiment analysis
SPEECH_BIAS_CORRECTION = 0.15  # Correction factor for speech detection
APPLY_SPEECH_BIAS_CORRECTION = True  # Whether to apply speech bias correction
PREDICTION_THRES = 0.5  # Threshold for sound prediction confidence
DBLEVEL_THRES = 45  # Threshold for dB level to consider sound significant

# Memory optimization settings
MEMORY_OPTIMIZATION_LEVEL = os.environ.get('MEMORY_OPTIMIZATION', '1')  # 0=None, 1=Moderate, 2=Aggressive
if MEMORY_OPTIMIZATION_LEVEL == '1':
    logger.info("Using moderate memory optimization")
    # Set PyTorch to release memory aggressively
    torch.set_num_threads(4)  # Limit threads for CPU usage
    # Setting empty_cache frequency
    EMPTY_CACHE_FREQ = 10  # Empty cache every 10 predictions
elif MEMORY_OPTIMIZATION_LEVEL == '2':
    logger.info("Using aggressive memory optimization")
    # More aggressive memory management
    torch.set_num_threads(2)  # More strict thread limiting
    # Release memory more frequently
    EMPTY_CACHE_FREQ = 5  # Empty cache every 5 predictions
else:
    logger.info("No memory optimization")
    EMPTY_CACHE_FREQ = 0  # Never automatically empty cache

# Global counter for memory cleanup
prediction_counter = 0

# Function to clean up memory periodically
def cleanup_memory():
    """Clean up unused memory to prevent memory leaks"""
    global prediction_counter
    
    prediction_counter += 1
    
    # Skip if memory optimization is disabled
    if EMPTY_CACHE_FREQ == 0:
        return
    
    # Clean up memory periodically
    if prediction_counter % EMPTY_CACHE_FREQ == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            # For CPU, trigger Python garbage collection
            gc.collect()
        logger.info(f"Memory cleanup performed (cycle {prediction_counter})")

# Speech recognition settings
os.path.dirname(os.path.abspath(__file__))

# Keep track of server start time
start_time = time.time()

# Helper function to get the computer's IP addresses
def get_ip_addresses():
    ip_list = []
    try:
        # Get all network interfaces
        hostname = socket.gethostname()
        # Get the primary IP (the one used for external connections)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't have to be reachable
            s.connect(('10.255.255.255', 1))
            primary_ip = s.getsockname()[0]
            ip_list.append(primary_ip)
        except Exception:
            pass
        finally:
            s.close()
            
        # Get all other IPs
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip not in ip_list and not ip.startswith('127.'):
                ip_list.append(ip)
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
    
    return ip_list

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

# Configure TensorFlow to use compatibility mode with TF 1.x code
tf.compat.v1.disable_eager_execution()

# Create a TensorFlow lock for thread safety, and another lock for the AST model
tf_lock = Lock()
tf_graph = tf.compat.v1.Graph()
tf_session = tf.compat.v1.Session(graph=tf_graph)
ast_lock = Lock()
speech_lock = Lock()  # Lock for speech processing

# Dictionary to store model-specific info
model_info = {
    "input_name": None,
    "output_name": None
}

# Prediction aggregation system - store recent predictions to improve accuracy
MAX_PREDICTIONS_HISTORY = 3  # Increased from 2 to 3 for more stable non-speech sound detection
SPEECH_PREDICTIONS_HISTORY = 3  # Decreased from 4 to 3 to balance with other sounds
recent_predictions = []  # Store recent prediction probabilities for each sound category
speech_predictions = []  # Separate storage for speech predictions
prediction_lock = Lock()  # Lock for thread-safe access to prediction history

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'soundwatch_secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

# Add notification cooldown tracking
last_notification_time = 0
last_notification_sound = None
NOTIFICATION_COOLDOWN_SECONDS = 2.0  # Minimum seconds between notifications

# Thresholds
FINGER_SNAP_THRES = 0.4  # Threshold for finger snap detection
SILENCE_THRES = -60  # Threshold for silence detection (increased from -75 to be more practical)
SPEECH_SENTIMENT_THRES = 0.8  # Threshold for speech sentiment analysis
CHOPPING_THRES = 0.7  # Threshold for chopping sound detection
SPEECH_PREDICTION_THRES = 0.65  # Lowered from 0.7 to 0.65 for better speech detection
SPEECH_DETECTION_THRES = 0.55  # Lowered from 0.6 to 0.55 for secondary speech detection

# Define critical sounds that get priority treatment
CRITICAL_SOUNDS = {
    'hazard-alarm': 0.25,     # Fire/Smoke alarm - increased from 0.05
    'knock': 0.25,            # Knock - increased from 0.05
    'doorbell': 0.25,         # Doorbell - increased from 0.07
    'baby-cry': 0.30,         # Baby crying - increased from 0.1
    'water-running': 0.30,    # Water running - increased from 0.1
    'phone-ring': 0.30,       # Phone ringing - increased from 0.1
    'alarm-clock': 0.30,      # Alarm clock - increased from 0.1
    'cooking': 0.40           # Added - Utensils and Cutlery
}

# Define model-specific contexts based on SoundWatch research (20 high-priority sound classes)
core_sounds = [
    # High-priority sounds (critical for safety/awareness)
    'hazard-alarm',    # Fire/smoke alarm - highest priority
    'alarm-clock',     # Alarm clock
    'knock',           # Door knock  
    'doorbell',        # Doorbell
    'phone-ring',      # Phone ringing
    'baby-cry',        # Baby crying
    
    # Medium-priority household sounds
    'door',            # Door opening/closing
    'water-running',   # Running water
    'microwave',       # Microwave beeping
    'speech',          # Human speech
    'dog-bark',        # Dog barking
    'cat-meow',        # Cat meowing
    'cough',           # Coughing
    
    # Common household appliances/activities
    'vacuum',          # Vacuum cleaner
    'blender',         # Blender
    'chopping',        # Chopping (cooking)
    'dishwasher',      # Dishwasher
    'flush',           # Toilet flushing
    'typing',          # Keyboard typing
    'cooking'          # Cooking sounds
]

# Contexts - use only the valid sound labels the model can recognize
context = core_sounds
# Use this context for active detection
active_context = core_sounds

CHANNELS = 1
RATE = 16000
CHUNK = RATE  # 1 second chunks
SPEECH_CHUNK_MULTIPLIER = 2.0  # Decreased from 4.0 to 2.0 for more balanced processing
MICROPHONES_DESCRIPTION = []
FPS = 60.0

# Minimum word length for meaningful transcription
MIN_TRANSCRIPTION_LENGTH = 3  # Minimum characters in transcription to analyze
MIN_MEANINGFUL_WORDS = 2  # Minimum number of meaningful words
COMMON_FALSE_POSITIVES = ["you", "the", "thank you", "thanks", "a", "to", "and", "is", "it", "that"]

# Load sentiment analysis model
try:
    sentiment_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    logger.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentiment model: {str(e)}")
    sentiment_pipeline = None

# Dictionary to map emotion to emojis
EMOTION_TO_EMOJI = {
    "joy": "😄",           # Happy face for joy
    "neutral": "😀",        # Neutral face
    "surprise": "😮",      # Surprised face
    "sadness": "😢",       # Sad face
    "fear": "😨",          # Fearful face
    "anger": "😠",         # Angry face
    "disgust": "🤢"        # Disgusted face
}

# Grouped emotions for simplified categories
EMOTION_GROUPS = {
    "Happy": ["joy", "love", "admiration", "approval", "caring", "excitement", "amusement", "gratitude", "optimism", "pride", "relief"],
    "Neutral": ["neutral", "realization", "curiosity"],
    "Surprised": ["surprise", "confusion", "nervousness"],
    "Unpleasant": ["sadness", "fear", "anger", "disgust", "disappointment", "embarrassment", "grief", "remorse", "annoyance", "disapproval"]
}

# Keep track of active clients
active_clients = set()

# Initialize speech recognition systems
google_speech_processor = None  # Will be lazy-loaded when needed

# Determine if we should attempt to load the AST model
USE_AST_MODEL = False  # Disable AST model completely

# Define model paths
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"

# Create or load the models
def load_models():
    """Load or create the models."""
    global USE_AST_MODEL, models, prediction_function, aggregator
    
    models = {}
    
    # Skip AST model loading completely - we don't want to use it anymore
    print("AST model disabled based on configuration")
    
    # Load TensorFlow model
    model_filename = os.path.abspath(MODEL_PATH)
    print("Loading TensorFlow model...")
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    
    homesounds_model = Path(model_filename)
    if not homesounds_model.is_file():
        print("Downloading example_model.hdf5 [867MB]: ")
        wget.download(MODEL_URL, MODEL_PATH)
    
    print("Using TensorFlow model: %s" % (model_filename))
    
    try:
        with tf_graph.as_default():
            with tf_session.as_default():
                # Load model using a more explicit approach
                models["tensorflow"] = tf.keras.models.load_model(model_filename, compile=False)
                
                # Create function that uses the session directly
                model = models["tensorflow"]
                
                # Create a custom predict function that uses the session directly
                def custom_predict(x):
                    # Get input and output tensors
                    input_tensor = model.inputs[0]
                    output_tensor = model.outputs[0]
                    # Run prediction in the session
                    return tf_session.run(output_tensor, feed_dict={input_tensor: x})
                
                # Replace the model's predict function with our custom one
                models["tensorflow"].predict = custom_predict
                model_info["input_name"] = "custom_input"  # Not actually used with our custom predict
                
                # Test it
                dummy_input = np.zeros((1, 96, 64, 1))
                _ = custom_predict(dummy_input)
                
                # Set this as the default prediction function
                prediction_function = custom_predict
                print("Custom prediction function initialized successfully")
        print("TensorFlow model loaded with compile=False option")
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        traceback.print_exc()
        raise Exception("Could not load TensorFlow model. Server cannot continue.")

    print("Using TensorFlow model as primary model")

    # Load the sentiment analysis model
    try:
        print("Loading sentiment analysis model...")
        # Use the _get_sentiment_model function from sentiment_analyzer
        from sentiment_analyzer import _get_sentiment_model
        _get_sentiment_model()  # This loads the model if not already loaded
        print("Sentiment analysis model loaded successfully")
    except Exception as e:
        print(f"Error loading sentiment analysis model: {e}")
        traceback.print_exc()
        
    # Initialize Google Speech API if specified
    if USE_GOOGLE_SPEECH_API:
        try:
            from google_speech import GoogleSpeechToText
            print("Initializing Google Cloud Speech-to-Text...")
            # We don't need to fully initialize it now, just make sure the module is available
            print("Google Cloud Speech-to-Text will be used for speech recognition")
        except ImportError:
            print("Google Cloud Speech module not available.")
            USE_GOOGLE_SPEECH_API = False

# Add a comprehensive debug function
def debug_predictions(predictions, label_list):
    print("===== DEBUGGING ALL PREDICTIONS (BEFORE THRESHOLD) =====")
    for idx, pred in enumerate(predictions):
        if idx < len(label_list):
            print(f"{label_list[idx]}: {pred:.6f}")
    print("=======================================================")

# Setup Audio Callback
def audio_samples(in_data, frame_count, time_info, status_flags):
    np_wav = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0  # Convert to [-1.0, +1.0]
    # Compute RMS and convert to dB
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)

    # Make predictions
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    
    if x.shape[0] != 0:
        x = x.reshape(len(x), 96, 64, 1)
        print('Reshape x successful', x.shape)
        pred = models["tensorflow"].predict(x)
        predictions.append(pred)
    
    print('Prediction succeeded')
    for prediction in predictions:
        context_prediction = np.take(
            prediction[0], [homesounds.labels[x] for x in active_context])
        m = np.argmax(context_prediction)
        if context_prediction[m] > PREDICTION_THRES and db > DBLEVEL_THRES:
            print("Prediction: %s (%0.2f)" % (
                homesounds.to_human_labels[active_context[m]], context_prediction[m]))

    print("Raw audio min/max:", np.min(np_wav), np.max(np_wav))
    print("Processed audio shape:", x.shape)

    return (in_data, 0)  # pyaudio.paContinue equivalent

# Custom TensorFlow prediction function
def tensorflow_predict(x_input, db_level=None):
    """Make predictions with TensorFlow model in the correct session context."""
    with tf_lock:
        with tf_graph.as_default():
            with tf_session.as_default():
                # Make predictions - expect input shape (1, 96, 64, 1)
                predictions = models["tensorflow"].predict(x_input)
                
                # Apply speech bias correction to reduce false speech detections
                if APPLY_SPEECH_BIAS_CORRECTION:
                    # Skip bias correction entirely if the audio is near-silent
                    if db_level is not None and db_level < SILENCE_THRES + 10:
                        print(f"Skipping speech bias correction for silent audio ({db_level} dB)")
                    else:
                        for i in range(len(predictions)):
                            speech_idx = homesounds.labels.get('speech', 4)  # Default to 4 if not found
                            
                            # Ensure the index is valid
                            if speech_idx < len(predictions[i]):
                                # Check if speech confidence is unusually high for potentially silent audio
                                is_silent = db_level is not None and db_level < DBLEVEL_THRES
                                
                                # Get original speech confidence
                                original_confidence = predictions[i][speech_idx]
                                
                                # Smarter bias correction based on db level and confidence
                                if db_level is not None and db_level > 70 and original_confidence > 0.8:
                                    correction = SPEECH_BIAS_CORRECTION * 0.5  # 50% of normal correction
                                elif is_silent:
                                    correction = SPEECH_BIAS_CORRECTION * 1.5
                                else:
                                    correction = SPEECH_BIAS_CORRECTION
                                
                                # Special case: if knock detection is also high, apply stronger speech correction
                                knock_idx = homesounds.labels.get('knock', 11)
                                if knock_idx < len(predictions[i]) and predictions[i][knock_idx] > 0.05:
                                    knock_confidence = predictions[i][knock_idx]
                                    additional_correction = min(0.8, knock_confidence * 2.0)
                                    correction *= (1.0 + additional_correction)
                                    print(f"Applied additional speech correction due to knock detection ({knock_confidence:.4f})")
                                
                                # Only apply correction if speech is high confidence
                                if original_confidence > 0.6:
                                    predictions[i][speech_idx] -= correction
                                    predictions[i][speech_idx] = max(0.0, predictions[i][speech_idx])
                                    
                                    # Slightly boost high-priority sounds to counter speech bias
                                    for priority_sound in list(CRITICAL_SOUNDS.keys()):
                                        if priority_sound in homesounds.labels:
                                            priority_idx = homesounds.labels[priority_sound]
                                            if priority_idx < len(predictions[i]) and predictions[i][priority_idx] > 0.05:
                                                if priority_sound == 'hazard-alarm':
                                                    boost = 0.15  # Strong boost for fire alarms
                                                else:
                                                    boost = 0.1   # Standard boost for other critical sounds
                                                
                                                predictions[i][priority_idx] += boost
                                                predictions[i][priority_idx] = min(1.0, predictions[i][priority_idx])
                                                print(f"Boosted {priority_sound} from {predictions[i][priority_idx]-boost:.4f} to {predictions[i][priority_idx]:.4f}")
                                    
                                    print(f"Applied speech bias correction: {original_confidence:.4f} -> {predictions[i][speech_idx]:.4f} (correction: {correction:.2f})")
                
                return predictions

# Add these new constants near other threshold definitions
SPEECH_BASE_THRESHOLD = 0.65  # Lowered from 0.7 to 0.65
SPEECH_HIGH_DB_THRESHOLD = 0.6  # Lowered from 0.65 to 0.6 for loud sounds
DB_THRESHOLD_ADJUSTMENT = 5  # dB range for threshold adjustment

# Replace the speech detection logic in handle_source
def get_adaptive_threshold(db_level, base_threshold):
    """Calculate adaptive threshold based on audio level"""
    if db_level > 75:  # Very loud sounds (raised from 70 to 75)
        return base_threshold * 0.75  # More reduction (from 0.8 to 0.75)
    elif db_level > 65:  # Moderately loud (raised from 60 to 65) 
        return base_threshold * 0.85  # More reduction (from 0.9 to 0.85)
    elif db_level < 40:  # Very quiet
        return base_threshold * 1.2
    return base_threshold

# Add these new constants near the other global settings
MIN_TIME_BETWEEN_PREDICTIONS = 0.5  # Minimum seconds between predictions
last_prediction_time = 0  # Track when we last made a prediction
audio_buffer = []  # Buffer to collect audio samples
AUDIO_BUFFER_MAX_SIZE = 3  # Maximum number of audio chunks to buffer before forcing processing
MIN_AUDIO_SAMPLES_FOR_PROCESSING = 16000  # Require at least 1 second of audio (16000 samples)

# Add this function near should_send_notification
def should_process_audio():
    """Check if enough time has passed since the last prediction to process new audio."""
    global last_prediction_time, audio_buffer
    
    current_time = time.time()
    time_since_last = current_time - last_prediction_time
    
    # If buffer is getting too large, force processing
    if len(audio_buffer) >= AUDIO_BUFFER_MAX_SIZE:
        logger.debug(f"Processing audio because buffer size ({len(audio_buffer)}) reached maximum")
        return True
        
    # If enough time has passed, allow processing
    if time_since_last >= MIN_TIME_BETWEEN_PREDICTIONS:
        return True
        
    # Not enough time has passed
    logger.debug(f"Skipping audio processing due to rate limit ({time_since_last:.2f}s < {MIN_TIME_BETWEEN_PREDICTIONS:.2f}s)")
    return False

@socketio.on('audio_data')
def handle_source(json_data):
    """Handle audio data from client."""
    global prediction_count
    
    session['receive_count'] = session.get('receive_count', 0) + 1
    
    # Check for the data field instead of audio_data
    if 'data' not in json_data:
        logger.warning("Received data without data field")
        return

    # Convert the JSON array to a numpy array directly
    try:
        # Convert the array from JSON directly to numpy
        audio_data = np.array(json_data['data'], dtype=np.float32)
        
        # Forward audio to continuous sentiment analyzer if available
        sentiment_analyzer = get_sentiment_analyzer()
        if sentiment_analyzer:
            sentiment_analyzer.add_audio(audio_data)
        
        # Calculate audio properties
        db_level = dbFS(audio_data)
        
        # Only process audio if above threshold or if we should always process
        if should_process_audio():
            process_start_time = time.time()
            
            # Ensure valid audio length
            if len(audio_data) < 16000 * 0.5:  # Less than 0.5 seconds
                return
            
            # Process for speech differently
            is_speech, speech_confidence, speech_features = is_likely_real_speech(audio_data)
            
            if is_speech:
                speech_result = process_speech_with_sentiment(audio_data)
                if speech_result and 'text' in speech_result and speech_result['text']:
                    record_time = time.time()
                    emit_sound_notification("Speech", speech_confidence, db_level, 
                                         str(record_time), str(record_time), speech_result)
                    
                    prediction_count += 1
                    total_processing_time = time.time() - process_start_time
                    logger.debug(f"Speech processing took {total_processing_time:.4f} seconds")
                    cleanup_memory()
                    return
            
            # Continue with regular sound classification
            # ... rest of existing sound processing code ...
                    
    except Exception as e:
        logger.error(f"Error in handle_source: {str(e)}")
        traceback.print_exc()
        emit('error', {'message': f'Server error: {str(e)}'})

# Keep track of recent audio buffers for better speech transcription
recent_audio_buffer = []
MAX_BUFFER_SIZE = 5  # Keep last 5 chunks

# Add this new function after the recent_audio_buffer initialization
def is_likely_real_speech(audio_data, sample_rate=16000):
    """Analyze audio to determine if it contains actual human speech."""
    # Basic validation
    if len(audio_data) < sample_rate * 0.5:  # Need at least 0.5s of audio
        logger.debug("Audio too short for speech analysis")
        return False, 0.0, {"reason": "too_short"}
    
    # Calculate basic audio metrics
    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms < 0.003:  # Lowered from 0.005 to detect quieter speech
        logger.debug(f"Audio too quiet for speech (RMS: {rms:.6f})")
        return False, 0.0, {"reason": "too_quiet", "rms": float(rms)}
    
    # Zero-crossing rate (ZCR) - speech has specific ZCR patterns
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
    zcr = zero_crossings / len(audio_data)
    
    # Calculate spectral features that distinguish speech
    try:
        # Apply short-time Fourier transform (STFT)
        f, t, Zxx = signal.stft(audio_data, fs=sample_rate, nperseg=400)
        
        # Get frequency band energies - focus on speech fundamentals (85-255 Hz)
        speech_band_indices = np.logical_and(f >= 85, f <= 255)
        vowel_indices = np.logical_and(f > 255, f <= 1000)  # Vowel formants
        mid_band_indices = np.logical_and(f > 1000, f <= 3000)  # Consonants
        high_band_indices = f > 3000
        
        # Calculate energy in each band
        speech_energy = np.mean(np.abs(Zxx[speech_band_indices, :]))
        vowel_energy = np.mean(np.abs(Zxx[vowel_indices, :]))
        mid_energy = np.mean(np.abs(Zxx[mid_band_indices, :]))
        high_energy = np.mean(np.abs(Zxx[high_band_indices, :]))
        
        # Calculate speech-to-noise ratio (higher for speech)
        speech_to_noise = (speech_energy + vowel_energy) / (mid_energy + high_energy + 1e-10)
        
        # Analyze energy variance over time (speech has dynamic patterns)
        frame_energy = np.sum(np.abs(Zxx) ** 2, axis=0)
        energy_variance = np.var(frame_energy)
        energy_range = np.max(frame_energy) - np.min(frame_energy)
        
        # Spectral flux - measures frame-to-frame spectral change (speech is dynamic)
        spectral_flux = 0.0
        if len(t) > 1:
            flux_sum = np.sum(np.diff(np.abs(Zxx), axis=1)**2)
            spectral_flux = flux_sum / (len(t)-1)
        
        # Calculate harmonicity - speech has harmonic structure
        harmonic_ratio = (speech_energy + vowel_energy) / (np.mean(np.abs(Zxx)) + 1e-10)
        
        # Store all features for logging and debugging
        features = {
            "rms": float(rms),
            "zcr": float(zcr),
            "speech_energy": float(speech_energy),
            "vowel_energy": float(vowel_energy),
            "mid_energy": float(mid_energy),
            "high_energy": float(high_energy),
            "speech_to_noise": float(speech_to_noise),
            "energy_variance": float(energy_variance),
            "energy_range": float(energy_range),
            "spectral_flux": float(spectral_flux),
            "harmonic_ratio": float(harmonic_ratio)
        }
        
        # Calculate comprehensive speech confidence score (0-1)
        zcr_factor = min(1.0, max(0.0, 1.0 - abs(zcr - 0.1) / 0.1))
        stn_factor = min(1.0, speech_to_noise / 1.8)  # Less strict SNR requirement (was 2.0)
        var_factor = min(1.0, energy_variance / 0.01)
        harmonic_factor = min(1.0, harmonic_ratio / 2.0)
        flux_factor = min(1.0, spectral_flux / 0.5) 
        
        speech_confidence = (0.15 * zcr_factor + 
                            0.25 * stn_factor + 
                            0.2 * var_factor + 
                            0.25 * harmonic_factor +
                            0.15 * flux_factor)
        
        # Debug logging
        logger.debug(f"Speech detection features: ZCR={zcr_factor:.2f}, STN={stn_factor:.2f}, " 
                    f"VAR={var_factor:.2f}, HARM={harmonic_factor:.2f}, FLUX={flux_factor:.2f}")
        
        # Decision thresholds based on audio level - lowered thresholds
        if rms > 0.05:  # Loud audio
            confidence_threshold = 0.40  # Lowered from 0.45
        else:
            confidence_threshold = 0.45  # Lowered from 0.55
        
        # Make final decision - lowered minimum RMS from 0.008 to 0.004
        is_speech = speech_confidence > confidence_threshold and rms > 0.004
        
        # Include decision factors in features
        features["speech_confidence"] = float(speech_confidence)
        features["confidence_threshold"] = float(confidence_threshold)
        features["is_speech"] = is_speech
        
        logger.debug(f"Speech detection result: {is_speech} (confidence: {speech_confidence:.2f}, threshold: {confidence_threshold:.2f})")
        return is_speech, speech_confidence, features
        
    except Exception as e:
        logger.error(f"Error in speech analysis: {str(e)}", exc_info=True)
        return False, 0.0, {"error": str(e)}

# Modify the process_speech_with_sentiment function
def process_speech_with_sentiment(audio_data):
    """Process speech audio to get transcription and sentiment analysis."""
    global recent_audio_buffer
    
    # Validate audio data
    if audio_data is None or len(audio_data) < RATE * 0.5:  # At least 0.5 seconds
        logger.warning("Audio data too short for speech processing")
        return None
    
    # Analyze audio to see if it's likely to contain speech
    is_speech, speech_confidence, speech_features = is_likely_real_speech(audio_data)
    
    if not is_speech and speech_confidence < 0.4:  # Lower threshold from 0.5 to 0.4
        logger.info(f"Audio unlikely to contain speech (confidence: {speech_confidence:.2f})")
        return None
    
    # Add current audio to buffer
    recent_audio_buffer.append(audio_data)
    
    # Keep buffer size limited
    if len(recent_audio_buffer) > MAX_BUFFER_SIZE:
        recent_audio_buffer.pop(0)
    
    # Concatenate recent audio for better context
    concatenated_audio = np.concatenate(recent_audio_buffer)
    
    # Calculate audio properties for logging
    rms = np.sqrt(np.mean(np.square(concatenated_audio)))
    
    # Initialize variables
    transcription = ""
    google_api_error = None
    vosk_fallback_used = False
    
    # Try Google Speech API first if enabled
    if USE_GOOGLE_SPEECH_API and CURRENT_SPEECH_ENGINE != SPEECH_ENGINE_VOSK:
        logger.info("Using Google Speech API for transcription")
        try:
            start_time = time.time()
            transcription, error_info = transcribe_with_google(concatenated_audio, RATE)
            processing_time = time.time() - start_time
            
            if error_info:
                google_api_error = error_info.get("error", "Unknown error")
                logger.warning(f"Google Speech API error: {google_api_error}")
                
                # If we have error details, log them
                if "details" in error_info:
                    logger.debug(f"Error details: {error_info['details']}")
                
                # Emit error notification to client
                socketio.emit('error', {'message': f'Speech recognition issue: {google_api_error}'})
                
                # If we're in auto mode, try Vosk as fallback
                if CURRENT_SPEECH_ENGINE == SPEECH_ENGINE_AUTO and VOSK_AVAILABLE:
                    logger.info("Falling back to Vosk for transcription")
                    vosk_fallback_used = True
                    try:
                        start_time = time.time()
                        transcription = transcribe_with_vosk(concatenated_audio, RATE)
                        processing_time = time.time() - start_time
                        logger.info(f"Vosk transcription completed in {processing_time:.2f}s, result: '{transcription}'")
                    except Exception as e:
                        logger.error(f"Vosk fallback also failed: {str(e)}")
                        # Emit error notification to client
                        socketio.emit('error', {'message': 'Speech recognition failed with all available engines'})
                        return {
                            "text": "",
                            "sentiment": {
                                "category": "Neutral",
                                "original_emotion": "neutral",
                                "confidence": 0.5,
                                "emoji": "😐"
                            },
                            "error": "All speech recognition engines failed"
                        }
                else:
                    # No fallback available or not in auto mode
                    logger.error("Google Speech API failed and no fallback available")
                    # Emit error notification to client
                    socketio.emit('error', {'message': 'Google Speech API failed and no fallback available'})
                    return {
                        "text": "",
                        "sentiment": {
                            "category": "Neutral",
                            "original_emotion": "neutral",
                            "confidence": 0.5,
                            "emoji": "😐"
                        },
                        "google_api_error": google_api_error
                    }
            else:
                logger.info(f"Google transcription completed in {processing_time:.2f}s, result: '{transcription}'")
        except Exception as e:
            logger.error(f"Error with Google Speech API: {str(e)}")
            google_api_error = str(e)
            
            # If we're in auto mode, try Vosk as fallback
            if CURRENT_SPEECH_ENGINE == SPEECH_ENGINE_AUTO and VOSK_AVAILABLE:
                logger.info("Falling back to Vosk for transcription due to Google API exception")
                vosk_fallback_used = True
                try:
                    start_time = time.time()
                    transcription = transcribe_with_vosk(concatenated_audio, RATE)
                    processing_time = time.time() - start_time
                    logger.info(f"Vosk transcription completed in {processing_time:.2f}s, result: '{transcription}'")
                except Exception as e:
                    logger.error(f"Vosk fallback also failed: {str(e)}")
                    # Emit error notification to client
                    socketio.emit('error', {'message': 'Speech recognition failed with all available engines'})
                    return {
                        "text": "",
                        "sentiment": {
                            "category": "Neutral",
                            "original_emotion": "neutral",
                            "confidence": 0.5,
                            "emoji": "😐"
                        },
                        "error": "All speech recognition engines failed"
                    }
            else:
                # No fallback available or not in auto mode
                logger.error("Google Speech API failed and no fallback available")
                # Emit error notification to client
                socketio.emit('error', {'message': 'Google Speech API failed and no fallback available'})
                return {
                    "text": "",
                    "sentiment": {
                        "category": "Neutral",
                        "original_emotion": "neutral",
                        "confidence": 0.5,
                        "emoji": "😐"
                    },
                    "google_api_error": google_api_error
                }

def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count},
                      namespace='/test')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/health')
def health_check():
    """API health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/api/status')
def server_status():
    """Return the current status of the server."""
    return jsonify({
        'status': 'running',
        'uptime': format_time(time.time() - start_time),
        'tensorflow_model_loaded': models.get("tensorflow") is not None,
        'sentiment_analyzer_running': get_sentiment_analyzer() is not None,
        'speech_recognition_engine': CURRENT_SPEECH_ENGINE,
        'using_google_speech': USE_GOOGLE_SPEECH_API,
        'model_info': model_info,
        'memory_optimization': MEMORY_OPTIMIZATION_LEVEL,
        'connected_clients': len(active_clients),
        'version': '1.2.0',
        'active_clients': len(active_clients)
    })

@app.route('/api/toggle-speech-recognition', methods=['POST'])
def toggle_speech_recognition():
    """Toggle Google Cloud Speech-to-Text on or off"""
    global USE_GOOGLE_SPEECH_API, CURRENT_SPEECH_ENGINE
    data = request.get_json()
    
    if data and 'use_google_speech' in data:
        USE_GOOGLE_SPEECH_API = data['use_google_speech']
        logger.info(f"Speech recognition {'enabled' if USE_GOOGLE_SPEECH_API else 'disabled'}")
        return jsonify({
            "success": True,
            "message": f"Speech recognition {'enabled' if USE_GOOGLE_SPEECH_API else 'disabled'}",
            "use_google_speech": USE_GOOGLE_SPEECH_API
        })
    elif data and 'speech_engine' in data:
        engine = data['speech_engine']
        if engine in [SPEECH_ENGINE_AUTO, SPEECH_ENGINE_GOOGLE, SPEECH_ENGINE_VOSK]:
            CURRENT_SPEECH_ENGINE = engine
            USE_GOOGLE_SPEECH_API = (engine != SPEECH_ENGINE_VOSK)
            logger.info(f"Speech recognition engine set to: {CURRENT_SPEECH_ENGINE}")
            return jsonify({
                "success": True,
                "message": f"Speech recognition engine set to: {CURRENT_SPEECH_ENGINE}",
                "speech_engine": CURRENT_SPEECH_ENGINE,
                "use_google_speech": USE_GOOGLE_SPEECH_API
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Invalid speech engine option: {engine}",
                "valid_options": [SPEECH_ENGINE_AUTO, SPEECH_ENGINE_GOOGLE, SPEECH_ENGINE_VOSK]
            }), 400
    else:
        USE_GOOGLE_SPEECH_API = not USE_GOOGLE_SPEECH_API
        CURRENT_SPEECH_ENGINE = SPEECH_ENGINE_GOOGLE if USE_GOOGLE_SPEECH_API else SPEECH_ENGINE_VOSK
        logger.info(f"Speech recognition toggled to: {CURRENT_SPEECH_ENGINE}")
        return jsonify({
            "success": True,
            "message": f"Speech recognition toggled to: {CURRENT_SPEECH_ENGINE}",
            "speech_engine": CURRENT_SPEECH_ENGINE,
            "use_google_speech": USE_GOOGLE_SPEECH_API
        })

@socketio.on('send_message')
def handle_source(json_data):
    print('Receive message...' + str(json_data['message']))
    text = json_data['message'].encode('ascii', 'ignore')
    socketio.emit('echo', {'echo': 'Server Says: ' + str(text)})
    print('Sending message back..')

@socketio.on('disconnect_request', namespace='/test')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)

@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected', request.sid)

@socketio.on('connect')
def handle_connect():
    """Handle client connection events."""
    global active_clients
    active_clients.add(request.sid)
    print(f"Client connected: {request.sid} (Total: {len(active_clients)})")
    emit('server_status', {'status': 'connected', 'message': 'Connected to SoundWatch server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection events."""
    global active_clients
    if request.sid in active_clients:
        active_clients.remove(request.sid)
    print(f"Client disconnected: {request.sid} (Total: {len(active_clients)})")

# Helper function to aggregate predictions
def aggregate_predictions(new_prediction, label_list, is_speech=False):
    """Aggregate predictions from multiple overlapping segments to improve accuracy."""
    global recent_predictions, speech_predictions
    
    with prediction_lock:
        if is_speech:
            speech_predictions.append(new_prediction)
            if len(speech_predictions) > SPEECH_PREDICTIONS_HISTORY:
                speech_predictions = speech_predictions[-SPEECH_PREDICTIONS_HISTORY:]
            predictions_list = speech_predictions
            history_len = SPEECH_PREDICTIONS_HISTORY
            logger.info(f"Using speech-specific aggregation with {len(predictions_list)} samples")
        else:
            recent_predictions.append(new_prediction)
            if len(recent_predictions) > MAX_PREDICTIONS_HISTORY:
                recent_predictions = recent_predictions[-MAX_PREDICTIONS_HISTORY:]
            predictions_list = recent_predictions
            history_len = MAX_PREDICTIONS_HISTORY
        
        if len(predictions_list) > 1:
            expected_shape = predictions_list[0].shape
            valid_predictions = []
            
            for pred in predictions_list:
                if pred.shape == expected_shape:
                    valid_predictions.append(pred)
                else:
                    logger.warning(f"Skipping prediction with incompatible shape: {pred.shape} (expected {expected_shape})")
            
            if valid_predictions:
                weights = np.linspace(0.5, 1.0, len(valid_predictions))
                weights = weights / np.sum(weights)
                
                aggregated = np.zeros_like(valid_predictions[0])
                for i, pred in enumerate(valid_predictions):
                    aggregated += pred * weights[i]
                
                logger.info(f"Aggregating {len(valid_predictions)} predictions {'(speech)' if is_speech else ''}")
            else:
                logger.warning("No predictions with matching shapes, using most recent prediction")
                aggregated = predictions_list[-1]
        else:
            aggregated = new_prediction
        
        orig_top_idx = np.argmax(new_prediction)
        agg_top_idx = np.argmax(aggregated)
        
        if orig_top_idx != agg_top_idx:
            orig_label = label_list[orig_top_idx] if orig_top_idx < len(label_list) else "unknown"
            agg_label = label_list[agg_top_idx] if agg_top_idx < len(label_list) else "unknown"
            logger.info(f"Aggregation changed top prediction: {orig_label} ({new_prediction[orig_top_idx]:.4f}) -> {agg_label} ({aggregated[agg_top_idx]:.4f})")
        else:
            label = label_list[orig_top_idx] if orig_top_idx < len(label_list) else "unknown"
            logger.info(f"Aggregation kept same top prediction: {label}, confidence: {new_prediction[orig_top_idx]:.4f} -> {aggregated[agg_top_idx]:.4f}")
        
        return aggregated

@socketio.on('audio_feature_data')
@performance_timer
def handle_audio_feature_data(json_data):
    """Handle audio feature data from clients."""
    global prediction_count
    
    try:
        # Check if data field exists
        if 'data' not in json_data:
            logger.warning("Received audio feature data without 'data' field")
            return
        
        # Convert data to numpy array
        try:
            audio_features = np.array(json_data['data'])
            
            # Reshape if needed (expected shape: (1, 96, 64, 1))
            if audio_features.shape != (1, 96, 64, 1):
                logger.warning(f"Unexpected audio feature shape: {audio_features.shape}, reshaping...")
                if len(audio_features.shape) == 3:
                    # Add batch dimension if missing
                    audio_features = np.expand_dims(audio_features, axis=0)
                if len(audio_features.shape) == 4 and audio_features.shape[3] != 1:
                    # Add channel dimension if missing
                    audio_features = np.expand_dims(audio_features, axis=3)
        except Exception as e:
            logger.error(f"Error converting audio feature data: {str(e)}")
            return
        
        # Get dB level if available
        db_level = None
        try:
            if 'db_level' in json_data:
                db_level = float(json_data['db_level'])
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing dB level: {str(e)}")
        
        # Process audio features with TensorFlow model
        if audio_features is not None and audio_features.shape[0] > 0:
            # Use model lock to ensure thread safety
            with model_lock:
                # Get TensorFlow model
                tf_model = models.get("tensorflow")
                if tf_model is not None:
                    # Make prediction
                    predictions = tf_model.predict(audio_features)
                    
                    # Process predictions
                    process_predictions(predictions, db_level, json_data)
                    
                    # Increment prediction count
                    prediction_count += 1
                else:
                    logger.warning("TensorFlow model not loaded, cannot process audio features")
    
    except Exception as e:
        logger.error(f"Error handling audio feature data: {str(e)}")
        traceback.print_exc()

# Adjust the notification cooldown settings
NOTIFICATION_COOLDOWN_SECONDS = 2.0  # Increased from 1.0 to 2.0 seconds
SPEECH_NOTIFICATION_COOLDOWN = 3.0  # Longer cooldown specifically for speech
UNRECOGNIZED_SOUND_COOLDOWN = 5.0  # Much longer cooldown for "Unrecognized Sound"

# Add a minimum volume threshold for Unrecognized Sound
UNRECOGNIZED_SOUND_MIN_DB = 55  # Only emit Unrecognized Sound notifications when above this dB level

# Update should_send_notification function
def should_send_notification(sound_label):
    """Check if enough time has passed since the last notification."""
    global last_notification_time, last_notification_sound
    
    current_time = time.time()
    time_since_last = current_time - last_notification_time
    
    if last_notification_time == 0:
        last_notification_time = current_time
        last_notification_sound = sound_label
        return True
    
    if sound_label == "Fire/Smoke Alarm":
        logger.debug(f"Critical sound '{sound_label}' bypassing cooldown")
        last_notification_time = current_time
        last_notification_sound = sound_label
        return True
    
    if sound_label == "Speech":
        required_cooldown = SPEECH_NOTIFICATION_COOLDOWN
    elif sound_label == "Unrecognized Sound":
        required_cooldown = UNRECOGNIZED_SOUND_COOLDOWN
    elif sound_label in ["Knocking", "Water Running", "Doorbell In-Use", "Baby Crying", "Phone Ringing", "Alarm Clock"]:
        required_cooldown = NOTIFICATION_COOLDOWN_SECONDS * 0.75
        if last_notification_sound not in ["Fire/Smoke Alarm", "Knocking", "Water Running", "Doorbell In-Use", 
                                          "Baby Crying", "Phone Ringing", "Alarm Clock"]:
            required_cooldown = 0.5
    elif sound_label == last_notification_sound:
        required_cooldown = NOTIFICATION_COOLDOWN_SECONDS * 1.5
    else:
        required_cooldown = NOTIFICATION_COOLDOWN_SECONDS
    
    if time_since_last >= required_cooldown:
        last_notification_time = current_time
        last_notification_sound = sound_label
        logger.debug(f"Allowing notification for '{sound_label}' after {time_since_last:.2f}s")
        return True
    
    logger.debug(f"Skipping notification for '{sound_label}' due to cooldown ({time_since_last:.2f}s < {required_cooldown:.2f}s)")
    return False

# Function to emit sound notifications with cooldown management
def emit_sound_notification(sound_label, confidence, db_level, current_time, record_time):
    """Emit sound notification to all connected clients."""
    try:
        # Create notification data
        notification_data = {
            'sound': sound_label,
            'confidence': confidence,
            'db_level': db_level,
            'time': current_time,
            'record_time': record_time
        }
        
        # Add to conversation history
        add_to_conversation_history(notification_data)
        
        # Emit to all clients
        socketio.emit('sound_notification', notification_data)
        logger.info(f"Emitted sound notification: {notification_data}")
        
        # Check if sentiment analysis is enabled
        if ENABLE_SENTIMENT_ANALYSIS and sound_label.lower() == 'speech':
            # Trigger sentiment analysis if speech is detected
            analyze_sentiment(notification_data)
    
    except Exception as e:
        logger.error(f"Error emitting sound notification: {str(e)}")
        traceback.print_exc()

def cleanup_memory():
    """Perform memory cleanup to prevent memory leaks."""
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear any temporary variables
        if 'tensorflow' in sys.modules:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}")

@app.route('/api/conversation-history')
def get_conversation_history():
    """API endpoint to get conversation history with sentiment analysis."""
    try:
        # Get sentiment analyzer
        sentiment_analyzer = get_sentiment_analyzer()
        if not sentiment_analyzer:
            return jsonify({"error": "Sentiment analyzer not initialized", "history": []}), 500
        
        # Get limit parameter (default: 20)
        try:
            limit = int(request.args.get('limit', 20))
        except (ValueError, TypeError):
            limit = 20
        
        # Get history
        history = sentiment_analyzer.get_conversation_history(limit)
        
        return jsonify({
            "history": history,
            "count": len(history)
        })
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "history": []}), 500

# Add a new route for the conversation history web interface
@app.route('/conversation-history')
def conversation_history_page():
    """Display conversation history with sentiment analysis in a web interface."""
    return send_from_directory(app.static_folder, 'conversation-history.html')

def process_predictions(predictions, db_level, json_data):
    """Process predictions from the model and emit notifications to clients."""
    try:
        # Apply speech bias correction if needed
        if APPLY_SPEECH_BIAS_CORRECTION:
            for i in range(len(predictions)):
                speech_idx = homesounds.labels.get('speech', 4)  # Default to 4 if not found
                
                if speech_idx < len(predictions[i]):
                    # Get original speech confidence
                    original_confidence = predictions[i][speech_idx]
                    
                    # Apply appropriate correction based on audio level
                    if db_level is not None and db_level > 70 and original_confidence > 0.8:
                        correction = SPEECH_BIAS_CORRECTION * 0.5  # 50% of normal correction
                    elif db_level is not None and db_level < DBLEVEL_THRES:
                        correction = SPEECH_BIAS_CORRECTION * 1.5  # Higher correction for silence
                    else:
                        correction = SPEECH_BIAS_CORRECTION
                    
                    # Apply the correction
                    predictions[i][speech_idx] -= correction
                    predictions[i][speech_idx] = max(0.0, predictions[i][speech_idx])
        
        # Get record time if available
        record_time = json_data.get('record_time', str(time.time()))
        current_time = json_data.get('time', str(time.time()))
        
        # Aggregate predictions for better accuracy
        for i, pred in enumerate(predictions):
            # Find the most likely sound and its confidence
            best_idx = np.argmax(pred)
            confidence = pred[best_idx]
            sound_label = homesounds.to_human_labels.get(best_idx, "Unknown")
            
            # Check if confidence exceeds threshold
            if confidence > PREDICTION_THRES and (db_level is None or db_level > DBLEVEL_THRES):
                logger.info(f"Sound detected: {sound_label} with confidence {confidence:.4f}")
                
                # Emit sound notification to clients
                emit_sound_notification(sound_label, str(confidence), 
                                     str(db_level) if db_level is not None else "0", 
                                     current_time, record_time)
                
                # Perform memory cleanup
                cleanup_memory()
                
    except Exception as e:
        logger.error(f"Error processing predictions: {str(e)}")
        traceback.print_exc()

def add_to_conversation_history(notification_data):
    """Add notification to conversation history."""
    try:
        # Create a copy of the notification data with additional fields
        history_entry = notification_data.copy()
        history_entry['timestamp'] = time.time()
        
        # Add to conversation history list (limit size to prevent memory issues)
        conversation_history.append(history_entry)
        if len(conversation_history) > MAX_CONVERSATION_HISTORY:
            conversation_history.pop(0)  # Remove oldest entry
            
        # Save to file periodically
        if len(conversation_history) % 10 == 0:
            save_conversation_history()
            
    except Exception as e:
        logger.error(f"Error adding to conversation history: {str(e)}")

def save_conversation_history():
    """Save conversation history to a file."""
    try:
        with open('conversation_history.json', 'w') as f:
            json.dump(conversation_history, f)
        logger.info(f"Saved conversation history with {len(conversation_history)} entries")
    except Exception as e:
        logger.error(f"Error saving conversation history: {str(e)}")

def analyze_sentiment(notification_data):
    """Analyze sentiment of speech and emit sentiment notification."""
    try:
        # For now, just generate a random sentiment as a placeholder
        # In a real implementation, this would use a sentiment analysis model
        import random
        sentiments = ['positive', 'neutral', 'negative']
        sentiment = random.choice(sentiments)
        confidence = random.uniform(0.7, 0.95)
        
        # Create sentiment notification data
        sentiment_data = {
            'sentiment': sentiment,
            'confidence': str(confidence),
            'original_notification': notification_data
        }
        
        # Emit sentiment notification
        socketio.emit('sentiment_notification', sentiment_data)
        logger.info(f"Emitted sentiment notification: {sentiment_data}")
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        traceback.print_exc()

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

if __name__ == '__main__':
    # Parse command-line arguments for port configuration
    parser = argparse.ArgumentParser(description='Sonarity Audio Analysis Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--use-google-speech', action='store_true', default=True, help='Use Google Cloud Speech-to-Text (enabled by default)')
    parser.add_argument('--speech-engine', type=str, choices=[SPEECH_ENGINE_AUTO, SPEECH_ENGINE_GOOGLE, SPEECH_ENGINE_VOSK], 
                        default=SPEECH_ENGINE_AUTO, help='Speech recognition engine to use')
    parser.add_argument('--disable-continuous-sentiment', action='store_true', help='Disable continuous sentiment analysis')
    args = parser.parse_args()
    
    # Enable debug mode if specified
    if args.debug:
        DEBUG_MODE = True
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Update speech recognition setting based on command line arguments
    if args.speech_engine:
        CURRENT_SPEECH_ENGINE = args.speech_engine
        USE_GOOGLE_SPEECH_API = (CURRENT_SPEECH_ENGINE != SPEECH_ENGINE_VOSK)
        logger.info(f"Using speech recognition engine: {CURRENT_SPEECH_ENGINE}")
    elif not args.use_google_speech:
        USE_GOOGLE_SPEECH_API = False
        CURRENT_SPEECH_ENGINE = SPEECH_ENGINE_VOSK
        logger.info("Using Vosk for speech recognition (Google Speech API disabled)")
    else:
        USE_GOOGLE_SPEECH_API = True
        logger.info("Using Google Cloud Speech-to-Text for speech recognition")
    
    # Initialize and load all models
    logger.info("Setting up sound recognition models...")
    load_models()
    
    # Initialize the continuous sentiment analyzer
    if not args.disable_continuous_sentiment:
        initialize_sentiment_analyzer(socketio, RATE)
        sentiment_analyzer = get_sentiment_analyzer()
        sentiment_analyzer.start()
        logger.info("Continuous sentiment analysis service started")
    
    # Get all available IP addresses
    ip_addresses = get_ip_addresses()
    
    logger.info("="*60)
    logger.info("SONARITY SERVER STARTED")
    logger.info("="*60)
    
    if ip_addresses:
        logger.info("Server is available at:")
        for i, ip in enumerate(ip_addresses):
            logger.info(f"{i+1}. http://{ip}:{args.port}")
            logger.info(f"   WebSocket: ws://{ip}:{args.port}")
        
        logger.info("\nExternal access: http://34.16.101.179:%d" % args.port)
        logger.info("External WebSocket: ws://34.16.101.179:%d" % args.port)
        
        logger.info("\nPreferred connection address: http://%s:%d" % (ip_addresses[0], args.port))
        logger.info("Preferred WebSocket address: ws://%s:%d" % (ip_addresses[0], args.port))
    else:
        logger.warning("Could not determine IP address. Make sure you're connected to a network.")
        logger.info(f"Try connecting to your server's IP address on port {args.port}")
        logger.info("\nExternal access: http://34.16.101.179:%d" % args.port)
        logger.info("External WebSocket: ws://34.16.101.179:%d" % args.port)
    
    logger.info("="*60 + "\n")
    
    # Get port from environment variable if set
    port = int(os.environ.get('PORT', args.port))
    
    # Run the server on all network interfaces
    socketio.run(app, host='0.0.0.0', port=port, debug=args.debug)