from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
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

# Import Vosk speech recognition
try:
    from vosk_speech import transcribe_with_vosk, VOSK_AVAILABLE
except ImportError:
    VOSK_AVAILABLE = False
    print("WARNING: Vosk speech recognition not available")

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
import ast_model

# Global flags for speech recognition - defined BEFORE they're referenced
USE_GOOGLE_SPEECH_API = True  # Set to True to use Google Cloud Speech API (default)

# Define speech recognition engine options
SPEECH_ENGINE_AUTO = "auto"      # Use Google with Vosk fallback (default)
SPEECH_ENGINE_GOOGLE = "google"  # Use Google only (no fallback)
SPEECH_ENGINE_VOSK = "vosk"      # Use Vosk only (offline mode)

# Current speech recognition engine setting
SPEECH_RECOGNITION_ENGINE = SPEECH_ENGINE_AUTO  # Default to automatic mode

# Import our sentiment analysis modules
from sentiment_analyzer import analyze_sentiment
from google_speech import transcribe_with_google, GoogleSpeechToText

# Memory optimization settings
import os
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
            import gc
            gc.collect()
        logger.info(f"Memory cleanup performed (cycle {prediction_counter})")

# Speech recognition settings

# Add the current directory to the path so we can import our modules
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
# This is needed to ensure compatibility with old model format
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
PREDICTION_THRES = 0.15  # Lower from 0.25/0.30 to 0.15
FINGER_SNAP_THRES = 0.4  # Threshold for finger snap detection
DBLEVEL_THRES = -60  # Minimum decibel level for sound detection
SILENCE_THRES = -60  # Threshold for silence detection (increased from -75 to be more practical)
SPEECH_SENTIMENT_THRES = 0.8  # Threshold for speech sentiment analysis
CHOPPING_THRES = 0.7  # Threshold for chopping sound detection
SPEECH_PREDICTION_THRES = 0.65  # Lowered from 0.7 to 0.65 for better speech detection
SPEECH_DETECTION_THRES = 0.55  # Lowered from 0.6 to 0.55 for secondary speech detection
SPEECH_BIAS_CORRECTION = 0.25  # Reduced from 0.3 to 0.25 to avoid excessive correction
KNOCK_LOWER_THRESHOLD = 0.05  # Lower threshold for knock detection

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

# Apply stronger speech bias correction since the model is heavily biased towards speech
APPLY_SPEECH_BIAS_CORRECTION = True  # Flag to enable/disable bias correction

# Define model-specific contexts based on SoundWatch research (20 high-priority sound classes)
# High-priority sounds (like alarms, knocks) are listed first as they are most important for DHH users
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
MIN_MEANINGFUL_WORDS = 2  # Minimum number of meaningful words (not just "you" or "thank you")
# Common short transcription results to ignore
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

# Dictionary to store our models
models = {
    "tensorflow": None,
    "ast": None,
    "feature_extractor": None
}

# Keep track of active clients
active_clients = set()

# Initialize speech recognition systems
google_speech_processor = None  # Will be lazy-loaded when needed

# Load models
def load_models():
    """Load all required models for sound recognition and speech processing."""
    global models, USE_AST_MODEL, USE_GOOGLE_SPEECH_API
    
    # Initialize models dictionary
    models = {
        "tensorflow": None,
        "ast": None,
        "feature_extractor": None,
        "sentiment_analyzer": None
    }
    
    # Flag to determine which model to use
    USE_AST_MODEL = os.environ.get('USE_AST_MODEL', '1') == '1'  # Default to enabled
    print(f"AST model {'enabled' if USE_AST_MODEL else 'disabled'} based on environment settings")
    
    # TensorFlow model settings
    MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
    MODEL_PATH = "models/example_model.hdf5"
    
    # Load AST model first
    try:
        print("Loading AST model...")
        with ast_lock:
            # AST model loading settings
            ast_kwargs = {
                "torch_dtype": torch.float32  # Always use float32 for maximum compatibility
            }
            
            # Use SDPA if available for better performance
            if torch.__version__ >= '2.1.1':
                ast_kwargs["attn_implementation"] = "sdpa"
                print("Using Scaled Dot Product Attention (SDPA) for faster inference")
            
            print("Using standard precision (float32) for maximum compatibility")
            
            # Load model with explicit float32 precision
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            models["ast"], models["feature_extractor"] = ast_model.load_ast_model(
                model_name=model_name,
                **ast_kwargs
            )
            
            # Initialize class labels for aggregation
            ast_model.initialize_class_labels(models["ast"])
            print("AST model loaded successfully")
    except Exception as e:
        print(f"Error loading AST model: {e}")
        traceback.print_exc()
        USE_AST_MODEL = False  # Fall back to TensorFlow model if AST fails to load

    # Optionally load TensorFlow model (as fallback or if USE_AST_MODEL is False)
    model_filename = os.path.abspath(MODEL_PATH)
    if not USE_AST_MODEL or True:  # We'll load it anyway as backup
        print("Loading TensorFlow model as backup...")
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
                    _ = models["tensorflow"].predict(dummy_input)
                    print("Custom prediction function initialized successfully")
            print("TensorFlow model loaded with compile=False option")
        except Exception as e2:
            print(f"Error with fallback method: {e2}")
            traceback.print_exc()
            try:
                print("Trying third fallback method with explicit tensor names...")
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
                        _ = models["tensorflow"].predict(dummy_input)
                        print("Custom prediction function initialized successfully")
                print("Third fallback method succeeded")
            except Exception as e3:
                print(f"Error with third fallback method: {e3}")
                traceback.print_exc()
                if not USE_AST_MODEL:
                    raise Exception("Could not load TensorFlow model with any method, and AST model is not enabled")

    print(f"Using {'AST' if USE_AST_MODEL else 'TensorFlow'} model as primary model")

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
    """Make predictions with TensorFlow model in the correct session context.
    
    Args:
        x_input: Input data in the correct shape for the model (1, 96, 64, 1)
        db_level: Optional decibel level of the audio, used for intelligent bias correction
    
    Returns:
        List of predictions with speech bias correction applied if enabled
    """
    with tf_lock:
        with tf_graph.as_default():
            with tf_session.as_default():
                # Make predictions - expect input shape (1, 96, 64, 1) as per SoundWatch specs
                predictions = models["tensorflow"].predict(x_input)
                
                # Apply speech bias correction to reduce false speech detections
                # According to research, speech is often over-predicted by the model
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
                                # 1. High db (>70dB) with high confidence (>0.8) = likely real speech = less correction
                                # 2. Low db with high confidence = likely false positive = more correction
                                # 3. Medium case = standard correction
                                if db_level is not None and db_level > 70 and original_confidence > 0.8:
                                    # Very loud, high confidence speech - apply minimal correction
                                    correction = SPEECH_BIAS_CORRECTION * 0.5  # 50% of normal correction
                                elif is_silent:
                                    # Silent/near-silent but high speech confidence = false positive
                                    correction = SPEECH_BIAS_CORRECTION * 1.5
                                else:
                                    # Standard case
                                    correction = SPEECH_BIAS_CORRECTION
                                
                                # Special case: if knock detection is also high, apply stronger speech correction
                                # This helps prevent speech from overwhelming knock detection
                                knock_idx = homesounds.labels.get('knock', 11)
                                if knock_idx < len(predictions[i]) and predictions[i][knock_idx] > 0.05:
                                    # The more confident the knock, the more we reduce speech
                                    knock_confidence = predictions[i][knock_idx]
                                    # Scale correction: 0.05 confidence = +10%, 0.2 confidence = +40% correction
                                    additional_correction = min(0.8, knock_confidence * 2.0)
                                    correction *= (1.0 + additional_correction)
                                    print(f"Applied additional speech correction due to knock detection ({knock_confidence:.4f})")
                                
                                # Only apply correction if speech is high confidence
                                if original_confidence > 0.6:
                                    # Apply correction factor to reduce speech confidence
                                    predictions[i][speech_idx] -= correction
                                    
                                    # Ensure it doesn't go below 0
                                    predictions[i][speech_idx] = max(0.0, predictions[i][speech_idx])
                                    
                                    # Slightly boost high-priority sounds to counter speech bias
                                    # This helps ensure important sounds like alarms are detected properly
                                    for priority_sound in list(CRITICAL_SOUNDS.keys()):
                                        if priority_sound in homesounds.labels:
                                            priority_idx = homesounds.labels[priority_sound]
                                            if priority_idx < len(predictions[i]) and predictions[i][priority_idx] > 0.05:
                                                # Boost critical sounds more aggressively when they have some signal
                                                # Higher boost for safety-critical sounds
                                                if priority_sound == 'hazard-alarm':
                                                    boost = 0.15  # Strong boost for fire alarms
                                                else:
                                                    boost = 0.1   # Standard boost for other critical sounds
                                                
                                                predictions[i][priority_idx] += boost
                                                predictions[i][priority_idx] = min(1.0, predictions[i][priority_idx])  # Cap at 1.0
                                                print(f"Boosted {priority_sound} from {predictions[i][priority_idx]-boost:.4f} to {predictions[i][priority_idx]:.4f}")
                                    
                                    # Debug print to show the effect of bias correction
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

# Add these new constants near the other global settings (around line 150-200)
# Rate limiting settings to prevent rapid-fire predictions
MIN_TIME_BETWEEN_PREDICTIONS = 0.5  # Minimum seconds between predictions
last_prediction_time = 0  # Track when we last made a prediction
audio_buffer = []  # Buffer to collect audio samples
AUDIO_BUFFER_MAX_SIZE = 3  # Maximum number of audio chunks to buffer before forcing processing
MIN_AUDIO_SAMPLES_FOR_PROCESSING = 16000  # Require at least 1 second of audio (16000 samples)

# Add this function near should_send_notification
def should_process_audio():
    """
    Check if enough time has passed since the last prediction to process new audio.
    Also returns True if audio buffer gets too large to prevent backlog.
    
    Returns:
        True if audio should be processed, False otherwise
    """
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

# Modify the handle_source function at the beginning to include rate limiting and buffering
@socketio.on('audio_data')
def handle_source(json_data):
    """Handle audio data sent from client."""
    global last_prediction_time, audio_buffer
    
    try:
        # Extract data from the request
        data = json_data.get('data', [])
        db = json_data.get('db', 0)
        time_data = json_data.get('time', 0)
        record_time = json_data.get('record_time', None)
        
        # Debug basic audio info - fix the shape attribute error by using len() instead
        logger.debug(f"Received audio chunk: {len(data)} samples, {db} dB")
        
        # Convert list to numpy array before adding to buffer
        np_data = np.array(data, dtype=np.float32)
        
        # Add to audio buffer
        audio_buffer.append(np_data)
        
        # Check if we should process audio now
        if not should_process_audio():
            # Not enough time has passed, buffer this audio for later
            return
            
        # We're going to process audio now - reset the prediction time
        last_prediction_time = time.time()
        
        # Combine buffered audio
        combined_audio = np.concatenate(audio_buffer)
        audio_buffer.clear()  # Clear buffer after using it
        
        # Ensure we have enough audio, skip if not
        if len(combined_audio) < MIN_AUDIO_SAMPLES_FOR_PROCESSING * 0.5:  # At least half the minimum required
            logger.debug(f"Skipping audio processing - not enough samples ({len(combined_audio)} < {MIN_AUDIO_SAMPLES_FOR_PROCESSING * 0.5})")
            return
            
        # Calculate average decibel level
        rms = np.sqrt(np.mean(combined_audio**2))
        combined_db = dbFS(rms)
        logger.debug(f"Processing combined audio: {len(combined_audio)} samples, {combined_db} dB")
        
        # ENHANCED SILENCE DETECTION: Check if audio is silent
        if combined_db < SILENCE_THRES:
            logger.debug(f"Combined audio is silent: {combined_db} dB < {SILENCE_THRES} dB threshold")
            emit_sound_notification('Silent', '1.0', combined_db, time_data, record_time)
            return
            
        # Proceed with the existing code but using combined_audio instead of data
        # and combined_db instead of db...
        
        # Ensure we have exactly 1 second of audio (16000 samples at 16KHz)
        original_size = combined_audio.size
        if original_size < RATE:
            # Pad with zeros if shorter than 1 second (16000 samples)
            padding = np.zeros(RATE - original_size, dtype=np.float32)
            combined_audio = np.concatenate([combined_audio, padding])
            logger.debug(f"Audio padded to {RATE} samples (1 second)")
        elif original_size > RATE:
            # If more than 1 second, take the last 1 second of audio (most recent)
            combined_audio = combined_audio[-RATE:]
            logger.debug(f"Audio trimmed to {RATE} samples (1 second)")
        
        # Convert waveform to log mel-spectrogram features
        x = waveform_to_examples(combined_audio, RATE)
        
        # Continue with the rest of the existing function using combined_audio
        # Just replace all instances of data with combined_audio and db with combined_db
        
        # Basic sanity check for x
        if x.shape[0] == 0:
            print("Error: Empty audio features")
            emit_sound_notification('Error: Empty Audio Features', '0.0', combined_db, time_data, record_time)
            return
        
        # Reshape for model input - model expects shape (1, 96, 64, 1)
        # 96 time frames, 64 mel bins, 1 channel
        if x.shape[0] > 1:
            print(f"Using first frame from multiple frames: {x.shape}")
            x_input = x[0].reshape(1, 96, 64, 1)
        else:
            x_input = x.reshape(1, 96, 64, 1)
            
        print(f"Processed audio features shape: {x_input.shape}")
        
        # Make prediction with TensorFlow model
        print("Making prediction with TensorFlow model...")
        predictions = tensorflow_predict(x_input, combined_db)
        
        # Debug all raw predictions (before thresholding)
        debug_predictions(predictions[0], homesounds.everything)
        
        # Use only valid indices for the active context
        valid_indices = []
        for x in active_context:
            if x in homesounds.labels and homesounds.labels[x] < len(predictions[0]):
                valid_indices.append(homesounds.labels[x])
            else:
                print(f"Warning: Label '{x}' maps to an invalid index or is not found in homesounds.labels")
        
        if not valid_indices:
            print("Error: No valid sound labels found in current context")
            emit_sound_notification('Error: Invalid Sound Context', '0.0', combined_db, time_data, record_time)
            return
        
        # Take predictions from the valid indices
        context_prediction = np.take(predictions[0], valid_indices)
        m = np.argmax(context_prediction)
        
            # Get the corresponding label from the valid indices
        predicted_label = active_context[valid_indices.index(valid_indices[m])]
        human_label = homesounds.to_human_labels[predicted_label]
        
        # Check for critical sounds with special thresholds before main processing
        for sound, threshold in CRITICAL_SOUNDS.items():
            if sound in homesounds.labels:
                sound_idx = homesounds.labels[sound]
                if sound_idx < len(predictions[0]) and predictions[0][sound_idx] > threshold:
                    # For loud sounds or sounds with higher confidence, prioritize them
                    confidence = predictions[0][sound_idx]
                    human_sound_label = homesounds.to_human_labels[sound]
                    
                    # Use different thresholds based on sound type and dB level
                    trigger_confidence = threshold
                    if sound == 'hazard-alarm' and combined_db > 60:  # Fire alarms are usually loud
                        trigger_confidence = threshold * 0.8  # Lower threshold for loud fire alarms
                    elif sound == 'water-running' and combined_db > 50:  # Water is usually mid-level volume
                        trigger_confidence = threshold * 0.9  # Lower threshold for audible water
                    
                    if confidence > trigger_confidence:
                        print(f"Critical sound '{human_sound_label}' detected with {confidence:.4f} confidence at {combined_db} dB")
                        emit_sound_notification(human_sound_label, str(confidence), combined_db, time_data, record_time)
                        return
        
        # Check for knock specifically (with lower threshold) before main processing
        knock_idx = homesounds.labels.get('knock', 11)
        if knock_idx < len(predictions[0]) and predictions[0][knock_idx] > KNOCK_LOWER_THRESHOLD:
            print(f"Found potential knock with confidence: {predictions[0][knock_idx]:.4f}")
            
            # If knock confidence is significant or higher than other predictions except speech
            # or if we have a moderate knock confidence with loud audio, emit knock notification
            if (predictions[0][knock_idx] > 0.2 or 
                (predictions[0][knock_idx] > 0.1 and combined_db > 60) or
                (predictions[0][knock_idx] == context_prediction[m] and predicted_label == 'knock')):
                print(f"Knock detection triggered with {predictions[0][knock_idx]:.4f} confidence at {combined_db} dB")
                emit_sound_notification('Knocking', str(predictions[0][knock_idx]), combined_db, time_data, record_time)
                return
        
        # Print prediction information
        print(f"Top prediction: {human_label} ({context_prediction[m]:.4f}, db: {combined_db})")

        # ENHANCED THRESHOLD CHECK: Verify both prediction confidence AND decibel level
        if context_prediction[m] > PREDICTION_THRES and combined_db > DBLEVEL_THRES:
            print(f"Top prediction: {human_label} ({context_prediction[m]:.4f}) at {combined_db} dB")
            
            # Special case for "Chopping" - use higher threshold to prevent false positives
            if human_label == "Chopping" and context_prediction[m] < CHOPPING_THRES:
                print(f"Ignoring Chopping sound with confidence {context_prediction[m]:.4f} < {CHOPPING_THRES} threshold")
                emit_sound_notification('Unrecognized Sound', '0.2', combined_db, time_data, record_time)
                return
                
            # Special case for "Speech" - use higher threshold and verify with Google Speech API
            if human_label == "Speech":
                logger.debug("Processing speech detection...")
                # Process as speech
                adaptive_threshold = get_adaptive_threshold(combined_db, SPEECH_BASE_THRESHOLD)
                if context_prediction[m] > adaptive_threshold:
                    # Speech detection passed threshold - emit the prediction and handle speech processing
                    logger.debug(f"Speech detected: confidence {context_prediction[m]:.4f} > {adaptive_threshold} threshold at {combined_db} dB")
                    
                    # Process the speech for sentiment analysis
                    try:
                        # Use the same audio data that was used for detection
                        sentiment_result = process_speech_with_sentiment(combined_audio)
                        
                        if sentiment_result and 'sentiment' in sentiment_result:
                            # Extract and log the speech engine used
                            engine_used = sentiment_result.get('transcription_engine', 'unknown')
                            transcription = sentiment_result.get('text', '')
                            logger.info(f"Speech processed using {engine_used} engine: '{transcription}'")
                            logger.debug(f"Speech sentiment: {sentiment_result['sentiment']['category']} with emoji {sentiment_result['sentiment']['emoji']}")
                            
                            # Log Google API errors if they occurred but Vosk was successful
                            if engine_used == 'vosk' and sentiment_result.get('google_api_error'):
                                logger.warning(f"Google Speech API error: {sentiment_result['google_api_error']}, but Vosk fallback worked")
                            
                            # Emit notification with sentiment and engine data
                            emit_sound_notification(
                                human_label, 
                                str(context_prediction[m]), 
                                combined_db, 
                                time_data,
                                record_time,
                                sentiment_result
                            )
                        else:
                            # If sentiment processing failed to return useful data, just emit regular notification
                            logger.debug("Speech sentiment processing returned no useful results")
                            emit_sound_notification(human_label, str(context_prediction[m]), combined_db, time_data, record_time)
                    except Exception as e:
                        logger.error(f"Error processing speech sentiment: {str(e)}", exc_info=True)
                        # Still emit the basic speech notification even if sentiment processing failed
                        emit_sound_notification(human_label, str(context_prediction[m]), combined_db, time_data, record_time)
                    
                    # If we're in debug mode, emit a debug message
                    if DEBUG_MODE:
                        logger.debug("Speech recognition passed threshold and was emitted to client")
                    else:
                        logger.debug(f"Ignoring Speech with confidence {context_prediction[m]:.4f} < {adaptive_threshold} threshold")
                        emit_sound_notification('Unrecognized Sound', '0.2', combined_db, time_data, record_time)
                        return
                else:
                    # For non-speech sounds, emit the prediction
                    logger.debug(f"Emitting non-speech prediction: {human_label}")
                    emit_sound_notification(human_label, str(context_prediction[m]), combined_db, time_data, record_time)
            else:
                # For non-speech sounds, emit the prediction
                logger.debug(f"Emitting non-speech prediction: {human_label}")
                emit_sound_notification(human_label, str(context_prediction[m]), combined_db, time_data, record_time)
        else:
            # Sound didn't meet thresholds
            reason = "confidence too low" if context_prediction[m] <= PREDICTION_THRES else "db level too low"
            print(f"Sound didn't meet thresholds: {reason} (prediction: {context_prediction[m]:.2f}, db: {combined_db})")
            
            # Check for knock with lower threshold as a fallback
            knock_idx = homesounds.labels.get('knock', 11)
            if knock_idx < len(predictions[0]) and predictions[0][knock_idx] > KNOCK_LOWER_THRESHOLD and combined_db > DBLEVEL_THRES:
                print(f"Detected knock with {predictions[0][knock_idx]:.4f} confidence as fallback!")
                emit_sound_notification('Knocking', str(predictions[0][knock_idx]), combined_db, time_data, record_time)
                return
            
            emit_sound_notification('Unrecognized Sound', '0.5', combined_db, time_data, record_time)
            print(f"Emitting: Unrecognized Sound (prediction: {context_prediction[m]:.2f}, db: {combined_db})")
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        # Send error message to client
        emit_sound_notification('Error', '0.0', 
            str(combined_db) if 'combined_db' in locals() else '-100',
            str(time_data) if 'time_data' in locals() else '0',
            str(record_time) if 'record_time' in locals() and record_time else '')

# Keep track of recent audio buffers for better speech transcription
recent_audio_buffer = []
MAX_BUFFER_SIZE = 5  # Keep last 5 chunks

# Add this new function after the recent_audio_buffer initialization but before process_speech_with_sentiment
def is_likely_real_speech(audio_data, sample_rate=16000):
    """
    Analyze audio to determine if it contains actual human speech using
    voice activity detection (VAD) techniques based on acoustic properties.
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate of the audio
        
    Returns:
        tuple: (is_speech, confidence, features_dict) 
    """
    # Basic validation
    if len(audio_data) < sample_rate * 0.5:  # Need at least 0.5s of audio
        logger.debug("Audio too short for speech analysis")
        return False, 0.0, {"reason": "too_short"}
    
    # Calculate basic audio metrics
    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms < 0.005:  # Very quiet audio is unlikely to be speech
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
        # Simplified measure: ratio of energy in harmonic bands vs total
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
        # Each factor contributes to the final confidence score
        
        # 1. ZCR factor - speech has characteristic ZCR
        # Typical speech ZCR is around 0.05-0.15
        zcr_factor = min(1.0, max(0.0, 1.0 - abs(zcr - 0.1) / 0.1))
        
        # 2. Speech-to-noise ratio - higher for speech 
        stn_factor = min(1.0, speech_to_noise / 2.0)
        
        # 3. Energy variance - speech has temporal variation
        var_factor = min(1.0, energy_variance / 0.01)
        
        # 4. Harmonic structure - speech has harmonic components
        harmonic_factor = min(1.0, harmonic_ratio / 2.0)
        
        # 5. Spectral flux - speech changes over time
        flux_factor = min(1.0, spectral_flux / 0.5) 
        
        # Combine all factors with appropriate weights
        # More weight on the most reliable indicators
        speech_confidence = (0.15 * zcr_factor + 
                            0.25 * stn_factor + 
                            0.2 * var_factor + 
                            0.25 * harmonic_factor +
                            0.15 * flux_factor)
        
        # Debug logging
        logger.debug(f"Speech detection features: ZCR={zcr_factor:.2f}, STN={stn_factor:.2f}, " 
                    f"VAR={var_factor:.2f}, HARM={harmonic_factor:.2f}, FLUX={flux_factor:.2f}")
        
        # Decision thresholds based on audio level
        if rms > 0.05:  # Loud audio
            confidence_threshold = 0.45  # Lower threshold for loud audio
        else:
            confidence_threshold = 0.55  # Higher threshold for quiet audio
        
        # Make final decision
        is_speech = speech_confidence > confidence_threshold and rms > 0.008
        
        # Include decision factors in features
        features["speech_confidence"] = float(speech_confidence)
        features["confidence_threshold"] = float(confidence_threshold)
        features["is_speech"] = is_speech
        
        logger.debug(f"Speech detection result: {is_speech} (confidence: {speech_confidence:.2f}, threshold: {confidence_threshold:.2f})")
        return is_speech, speech_confidence, features
        
    except Exception as e:
        logger.error(f"Error in speech analysis: {str(e)}", exc_info=True)
        return False, 0.0, {"error": str(e)}

# Modify the process_speech_with_sentiment function to use our new speech detection function
def process_speech_with_sentiment(audio_data):
    """
    Process speech audio, transcribe it and analyze sentiment.
    
    Args:
        audio_data: Raw audio data
        
    Returns:
        Dictionary with transcription and sentiment
    """
    # Settings for improved speech processing
    SPEECH_MAX_BUFFER_SIZE = 16  # Increased from 8 to 16 for much longer buffer (about 8-10 seconds of audio)
    MIN_WORD_COUNT = 2   # Reduced from 3 to 2 for better sensitivity
    MIN_CONFIDENCE = 0.7  # Minimum confidence level for speech detection
    
    # Initialize or update audio buffer (stored in function attributes)
    if not hasattr(process_speech_with_sentiment, "recent_audio_buffer"):
        process_speech_with_sentiment.recent_audio_buffer = []
    
    # Add current audio to buffer
    process_speech_with_sentiment.recent_audio_buffer.append(audio_data)
    
    # Keep only the most recent chunks
    if len(process_speech_with_sentiment.recent_audio_buffer) > SPEECH_MAX_BUFFER_SIZE:
        process_speech_with_sentiment.recent_audio_buffer = process_speech_with_sentiment.recent_audio_buffer[-SPEECH_MAX_BUFFER_SIZE:]
    
    # For better transcription, use concatenated audio from multiple chunks if available
    if len(process_speech_with_sentiment.recent_audio_buffer) > 1:
        # Use up to 16 chunks for speech recognition (about 8-10 seconds for better speech API performance)
        num_chunks = min(SPEECH_MAX_BUFFER_SIZE, len(process_speech_with_sentiment.recent_audio_buffer))
        logger.info(f"Using concatenated audio from {num_chunks} chunks for speech transcription")
        
        # Concatenate audio chunks
        concatenated_audio = np.concatenate(process_speech_with_sentiment.recent_audio_buffer[-num_chunks:])
    else:
        concatenated_audio = audio_data
    
    # Ensure minimum audio length for better transcription - increase to 8.0 seconds
    min_samples = RATE * 8.0  # At least 8.0 seconds of audio for speech (doubled from 4.0)
    if len(concatenated_audio) < min_samples:
        pad_size = int(min_samples) - len(concatenated_audio)
        # Use reflect padding to extend short audio naturally
        concatenated_audio = np.pad(concatenated_audio, (0, pad_size), mode='reflect')
        logger.info(f"Padded speech audio to size: {len(concatenated_audio)} samples ({len(concatenated_audio)/RATE:.1f} seconds)")
    
    # NEW: First run speech detection to verify this is real speech before attempting transcription
    is_speech, speech_confidence, speech_features = is_likely_real_speech(concatenated_audio, RATE)
    
    # Log the speech detection results
    logger.info(f"Speech verification: is_speech={is_speech}, confidence={speech_confidence:.2f}")
    
    # Only proceed with transcription if we're confident this is real speech
    if not is_speech:
        logger.info(f"Audio doesn't contain real human speech (confidence: {speech_confidence:.2f})")
        return {
            "text": "",
            "sentiment": {
                "category": "Neutral", 
                "original_emotion": "neutral",
                "confidence": 0.5,
                "emoji": "😐"
            },
            "speech_features": speech_features  # Return features for debugging
        }
    
    # Calculate the RMS value of the audio to gauge its "loudness"
    rms = np.sqrt(np.mean(np.square(concatenated_audio)))
    if rms < 0.001:  # Reduced threshold from 0.01 to 0.001 to be more sensitive
        logger.info(f"Audio too quiet (RMS: {rms:.4f}), skipping transcription")
        return {
            "text": "",
            "sentiment": {
                "category": "Neutral",
                "original_emotion": "neutral",
                "confidence": 0.5,
                "emoji": "😐"
            }
        }
    
    # Enhance audio signal for better transcription if the volume is low
    if rms < 0.05:  # If audio is quiet but above the minimum threshold
        # Apply audio normalization to boost the signal
        logger.info(f"Boosting audio signal (original RMS: {rms:.4f})")
        
        # Method 1: Simple normalization to get RMS to target level
        target_rms = 0.1  # Target RMS value
        gain_factor = target_rms / (rms + 1e-10)  # Avoid division by zero
        enhanced_audio = concatenated_audio * gain_factor
        
        # Check that we don't have clipping after amplification
        if np.max(np.abs(enhanced_audio)) > 0.99:
            # If clipping would occur, use a different approach
            logger.info("Using peak normalization to avoid clipping")
            peak_value = np.max(np.abs(concatenated_audio))
            if peak_value > 0:
                gain_factor = 0.95 / peak_value  # Target peak at 95% to avoid distortion
                enhanced_audio = concatenated_audio * gain_factor
            else:
                enhanced_audio = concatenated_audio
        
        # Use the enhanced audio for transcription
        new_rms = np.sqrt(np.mean(np.square(enhanced_audio)))
        logger.info(f"Audio boosted from RMS {rms:.4f} to {new_rms:.4f}")
        concatenated_audio = enhanced_audio
    
    # Apply high-pass filter to reduce background noise
    try:
        # Create a high-pass filter to reduce background noise
        b, a = signal.butter(2, 40/(RATE/2), 'highpass')  # Changed from 4th order 80Hz to 2nd order 40Hz filter
        filtered_audio = signal.filtfilt(b, a, concatenated_audio)
        logger.info("Applied gentle high-pass filter (40Hz) for noise reduction")
        
        # Apply a low-pass filter to reduce high-frequency noise above speech range
        try:
            b, a = signal.butter(3, 8000/(RATE/2), 'lowpass')  # Keep frequencies up to 8000Hz (speech range)
            filtered_audio = signal.filtfilt(b, a, filtered_audio)
            logger.info("Applied low-pass filter to focus on speech frequencies")
        except Exception as e:
            logger.warning(f"Error applying low-pass filter: {str(e)}")
            
        concatenated_audio = filtered_audio
    except Exception as e:
        logger.warning(f"Error applying high-pass filter: {str(e)}")
    
    logger.info("Transcribing speech to text...")
    
    # Variable to store transcription
    transcription = ""
    google_api_error = None
    vosk_fallback_used = False
    
    # Check for configured speech recognition engine
    if SPEECH_RECOGNITION_ENGINE == SPEECH_ENGINE_VOSK:
        # Use Vosk Only
        if VOSK_AVAILABLE:
            logger.info("Using Vosk for speech recognition (as configured)")
            try:
                start_time = time.time()
                transcription = transcribe_with_vosk(concatenated_audio, RATE)
                processing_time = time.time() - start_time
                logger.info(f"Vosk transcription completed in {processing_time:.2f}s, result: '{transcription}'")
                vosk_fallback_used = True
            except Exception as e:
                logger.error(f"Error with Vosk speech recognition: {str(e)}")
                return {
                    "text": "",
                    "sentiment": {
                        "category": "Neutral",
                        "original_emotion": "neutral",
                        "confidence": 0.5,
                        "emoji": "😐"
                    },
                    "transcription_engine": "vosk",
                    "error": str(e)
                }
        else:
            logger.error("Vosk is configured but not available. Please install it with: pip install vosk")
            return {
                "text": "",
                "sentiment": {
                    "category": "Neutral",
                    "original_emotion": "neutral",
                    "confidence": 0.5,
                    "emoji": "😐"
                },
                "error": "Vosk is not available but was selected as speech engine"
            }
    elif SPEECH_RECOGNITION_ENGINE == SPEECH_ENGINE_GOOGLE:
        # Use Google Only - No Fallback
        if USE_GOOGLE_SPEECH_API:
            try:
                logger.info(f"Using Google Speech API only (as configured)")
                start_time = time.time()
                # Set enhanced configuration options for better results
                options = {
                    "language_code": "en-US",
                    "sample_rate_hertz": RATE,
                    "enable_automatic_punctuation": True,
                    "use_enhanced": True,  # Use enhanced model for better accuracy
                    "model": "command_and_search",  # Better for short commands/phrases
                    "audio_channel_count": 1
                }
                
                # Make the API call with the enhanced settings
                transcription = transcribe_with_google(concatenated_audio, RATE, **options)
                processing_time = time.time() - start_time
                logger.info(f"Google transcription completed in {processing_time:.2f}s, result: '{transcription}'")
            except Exception as e:
                google_api_error = str(e)
                logger.error(f"Error with Google Speech API (no fallback configured): {google_api_error}")
                return {
                    "text": "",
                    "sentiment": {
                        "category": "Neutral",
                        "original_emotion": "neutral",
                        "confidence": 0.5,
                        "emoji": "😐"
                    },
                    "transcription_engine": "google",
                    "google_api_error": google_api_error
                }
        else:
            logger.error("Google Speech API is required but not available")
            return {
                "text": "",
                "sentiment": {
                    "category": "Neutral",
                    "original_emotion": "neutral",
                    "confidence": 0.5,
                    "emoji": "😐"
                },
                "error": "Google Speech API is not available but was selected as speech engine"
            }
    else:
        # Auto Mode (default) - Try Google first, then Vosk as fallback
        if USE_GOOGLE_SPEECH_API:
            try:
                # Add debug print to see audio length
                logger.info(f"Using Auto mode: Google first, then Vosk fallback. Sending {len(concatenated_audio)/RATE:.2f} seconds of audio to Google Speech API")
                
                # Set enhanced configuration options for better results
                options = {
                    "language_code": "en-US",
                    "sample_rate_hertz": RATE,
                    "enable_automatic_punctuation": True,
                    "use_enhanced": True,  # Use enhanced model for better accuracy
                    "model": "command_and_search",  # Better for short commands/phrases
                    "audio_channel_count": 1
                }
                
                start_time = time.time()
                # Make the API call with the enhanced settings
                transcription = transcribe_with_google(concatenated_audio, RATE, **options)
                processing_time = time.time() - start_time
                logger.info(f"Google transcription completed in {processing_time:.2f}s, result: '{transcription}'")
                
                # If Google returns empty result, try Vosk as fallback
                if not transcription and VOSK_AVAILABLE:
                    logger.info("Google Speech API returned empty result, trying Vosk as fallback")
                    vosk_fallback_used = True
                    start_time = time.time()
                    transcription = transcribe_with_vosk(concatenated_audio, RATE)
                    processing_time = time.time() - start_time
                    logger.info(f"Vosk fallback transcription completed in {processing_time:.2f}s, result: '{transcription}'")
            except Exception as e:
                google_api_error = str(e)
                logger.error(f"Error with Google Speech API: {google_api_error}")
                
                # Try Vosk as fallback if Google fails
                if VOSK_AVAILABLE:
                    logger.info("Trying Vosk as fallback due to Google Speech API error")
                    vosk_fallback_used = True
                    try:
                        start_time = time.time()
                        transcription = transcribe_with_vosk(concatenated_audio, RATE)
                        processing_time = time.time() - start_time
                        logger.info(f"Vosk fallback transcription completed in {processing_time:.2f}s, result: '{transcription}'")
                    except Exception as vosk_error:
                        logger.error(f"Error with Vosk fallback: {str(vosk_error)}")
                        # Log both errors for debugging
                        logger.error(f"Speech recognition failed with both engines - Google: {google_api_error}, Vosk: {str(vosk_error)}")
                        return {
                            "text": "",
                            "sentiment": {
                                "category": "Neutral",
                                "original_emotion": "neutral",
                                "confidence": 0.5,
                                "emoji": "😐"
                            },
                            "errors": {
                                "google": google_api_error,
                                "vosk": str(vosk_error)
                            }
                        }
                else:
                    logger.error("No fallback available: Vosk is not installed or available")
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
        elif VOSK_AVAILABLE:
            # If Google Speech API is disabled but we're in auto mode, try Vosk
            logger.info("Auto mode with Google disabled, using Vosk for transcription")
            vosk_fallback_used = True
            try:
                start_time = time.time()
                transcription = transcribe_with_vosk(concatenated_audio, RATE)
                processing_time = time.time() - start_time
                logger.info(f"Vosk transcription completed in {processing_time:.2f}s, result: '{transcription}'")
            except Exception as e:
                logger.error(f"Error with Vosk: {str(e)}")
                return {
                    "text": "",
                    "sentiment": {
                        "category": "Neutral",
                        "original_emotion": "neutral",
                        "confidence": 0.5,
                        "emoji": "😐"
                    },
                    "error": str(e)
                }
        else:
            # No speech recognition available
            logger.error("No speech recognition system available")
            return {
                "text": "",
                "sentiment": {
                    "category": "Neutral",
                    "original_emotion": "neutral",
                    "confidence": 0.5,
                    "emoji": "😐"
                },
                "error": "No speech recognition system available"
            }
    
    # Check for valid transcription with sufficient content
    if not transcription:
        logger.info("No valid transcription found")
        return {
            "text": "",
            "sentiment": {
                "category": "Neutral",
                "original_emotion": "neutral",
                "confidence": 0.5,
                "emoji": "😐"
            },
            "transcription_engine": "vosk" if vosk_fallback_used else "google",
            "google_api_error": google_api_error
        }
    
    # Filter out short or meaningless transcriptions
    common_words = ["the", "a", "an", "and", "but", "or", "if", "then", "so", "to", "of", "for", "in", "on", "at"]
    meaningful_words = [word for word in transcription.lower().split() if word not in common_words]
    
    if len(meaningful_words) < MIN_WORD_COUNT:
        logger.info(f"Transcription has too few meaningful words: '{transcription}'")
        return {
            "text": transcription,
            "sentiment": {
                "category": "Neutral",
                "original_emotion": "neutral",
                "confidence": 0.5,
                "emoji": "😐"
            },
            "transcription_engine": "vosk" if vosk_fallback_used else "google",
            "google_api_error": google_api_error
        }
    
    logger.info(f"Valid transcription found: '{transcription}'")
    
    # Analyze sentiment
    sentiment = analyze_sentiment(transcription)
    
    if sentiment:
        result = {
            "text": transcription,
            "sentiment": sentiment,
            "transcription_engine": "vosk" if vosk_fallback_used else "google",
            "google_api_error": google_api_error
        }
        logger.info(f"Sentiment analysis result: Speech {sentiment['category']} with emoji {sentiment['emoji']} (using {result['transcription_engine']})")
        return result
    
    return {
        "text": transcription,
        "sentiment": {
            "category": "Neutral",
            "original_emotion": "neutral",
            "confidence": 0.5,
            "emoji": "😐"
        },
        "transcription_engine": "vosk" if vosk_fallback_used else "google",
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

@app.route('/status')
def status():
    """Return the status of the server, including model loading status."""
    # Get the list of available IP addresses
    ip_addresses = get_ip_addresses()
    
    # Return the status information
    return jsonify({
        'status': 'running',
        'tensorflow_model_loaded': models["tensorflow"] is not None,
        'ast_model_loaded': models["ast"] is not None,
        'using_ast_model': USE_AST_MODEL,
        'speech_recognition': SPEECH_RECOGNITION_ENGINE,  # Updated to show current engine
        'speech_google_available': USE_GOOGLE_SPEECH_API,
        'speech_vosk_available': VOSK_AVAILABLE,
        'sentiment_analysis_enabled': True,
        'ip_addresses': ip_addresses,
        'uptime': time.time() - start_time,
        'version': '1.2.0',
        'active_clients': len(active_clients)
    })

@app.route('/api/toggle-speech-recognition', methods=['POST'])
def toggle_speech_recognition():
    """Toggle Google Cloud Speech-to-Text on or off"""
    global USE_GOOGLE_SPEECH_API, SPEECH_RECOGNITION_ENGINE
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
        # New option to set specific speech engine
        engine = data['speech_engine']
        if engine in [SPEECH_ENGINE_AUTO, SPEECH_ENGINE_GOOGLE, SPEECH_ENGINE_VOSK]:
            SPEECH_RECOGNITION_ENGINE = engine
            
            # Update USE_GOOGLE_SPEECH_API flag for backward compatibility
            USE_GOOGLE_SPEECH_API = (engine != SPEECH_ENGINE_VOSK)
            
            logger.info(f"Speech recognition engine set to: {SPEECH_RECOGNITION_ENGINE}")
            return jsonify({
                "success": True,
                "message": f"Speech recognition engine set to: {SPEECH_RECOGNITION_ENGINE}",
                "speech_engine": SPEECH_RECOGNITION_ENGINE,
                "use_google_speech": USE_GOOGLE_SPEECH_API
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Invalid speech engine option: {engine}",
                "valid_options": [SPEECH_ENGINE_AUTO, SPEECH_ENGINE_GOOGLE, SPEECH_ENGINE_VOSK]
            }), 400
    else:
        # Toggle the current value if no specific value provided
        USE_GOOGLE_SPEECH_API = not USE_GOOGLE_SPEECH_API
        
        # Update the engine setting to match
        SPEECH_RECOGNITION_ENGINE = SPEECH_ENGINE_GOOGLE if USE_GOOGLE_SPEECH_API else SPEECH_ENGINE_VOSK
        
        logger.info(f"Speech recognition toggled to: {SPEECH_RECOGNITION_ENGINE}")
        return jsonify({
            "success": True,
            "message": f"Speech recognition toggled to: {SPEECH_RECOGNITION_ENGINE}",
            "speech_engine": SPEECH_RECOGNITION_ENGINE,
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
    # For this emit we use a callback function
    # When the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
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
    """
    Handle client connection events.
    """
    global active_clients
    active_clients.add(request.sid)
    print(f"Client connected: {request.sid} (Total: {len(active_clients)})")
    # Send confirmation to client
    emit('server_status', {'status': 'connected', 'message': 'Connected to SoundWatch server'})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle client disconnection events.
    """
    global active_clients
    if request.sid in active_clients:
        active_clients.remove(request.sid)
    print(f"Client disconnected: {request.sid} (Total: {len(active_clients)})")

# Helper function to aggregate predictions from multiple overlapping segments
def aggregate_predictions(new_prediction, label_list, is_speech=False):
    """
    Aggregate predictions from multiple overlapping segments to improve accuracy.
    
    Args:
        new_prediction: The new prediction probabilities
        label_list: List of sound labels
        is_speech: Whether this is a speech prediction (uses different parameters)
        
    Returns:
        Aggregated prediction with highest confidence
    """
    global recent_predictions, speech_predictions
    
    with prediction_lock:
        # Add the new prediction to appropriate history
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
        
        # If we have multiple predictions, use weighted average favoring recent predictions
        if len(predictions_list) > 1:
            # Get the shape of the first prediction to determine the expected size
            expected_shape = predictions_list[0].shape
            valid_predictions = []
            
            # Filter predictions with matching shapes
            for pred in predictions_list:
                if pred.shape == expected_shape:
                    valid_predictions.append(pred)
                else:
                    logger.warning(f"Skipping prediction with incompatible shape: {pred.shape} (expected {expected_shape})")
            
            # Proceed only if we have valid predictions with matching shapes
            if valid_predictions:
                # Use weighted average giving more weight to recent predictions
                weights = np.linspace(0.5, 1.0, len(valid_predictions))
                weights = weights / np.sum(weights)  # Normalize weights
                
                aggregated = np.zeros_like(valid_predictions[0])
                for i, pred in enumerate(valid_predictions):
                    aggregated += pred * weights[i]
                
                # Debug the aggregation
                logger.info(f"Aggregating {len(valid_predictions)} predictions {'(speech)' if is_speech else ''}")
            else:
                # If no matching predictions, just use the most recent one
                logger.warning("No predictions with matching shapes, using most recent prediction")
                aggregated = predictions_list[-1]
        else:
            # Just return the single prediction if we don't have history yet
            aggregated = new_prediction
        
        # Compare original vs aggregated for top predictions
        orig_top_idx = np.argmax(new_prediction)
        agg_top_idx = np.argmax(aggregated)
        
        if orig_top_idx != agg_top_idx:
            # The top prediction changed after aggregation
            orig_label = label_list[orig_top_idx] if orig_top_idx < len(label_list) else "unknown"
            agg_label = label_list[agg_top_idx] if agg_top_idx < len(label_list) else "unknown"
            logger.info(f"Aggregation changed top prediction: {orig_label} ({new_prediction[orig_top_idx]:.4f}) -> {agg_label} ({aggregated[agg_top_idx]:.4f})")
        else:
            # Same top prediction, but confidence may have changed
            label = label_list[orig_top_idx] if orig_top_idx < len(label_list) else "unknown"
            logger.info(f"Aggregation kept same top prediction: {label}, confidence: {new_prediction[orig_top_idx]:.4f} -> {aggregated[agg_top_idx]:.4f}")
        
        return aggregated

@socketio.on('audio_feature_data')
@debug_log
@timing_decorator
def handle_source(json_data):
    """Handle pre-processed audio feature data sent from client."""
    global last_prediction_time
    
    try:
        # Check if we should process this audio now
        current_time = time.time()
        time_since_last = current_time - last_prediction_time
        
        if time_since_last < MIN_TIME_BETWEEN_PREDICTIONS:
            # Not enough time has passed
            logger.debug(f"Skipping audio feature processing due to rate limit ({time_since_last:.2f}s < {MIN_TIME_BETWEEN_PREDICTIONS:.2f}s)")
            return
            
        # We're going to process audio now - reset the prediction time
        last_prediction_time = current_time
        
        # Continue with the original function...
        # Extract data from the request
        data = json_data.get('data', [])
        db = json_data.get('db', 0)
        time_data = json_data.get('time', 0)
        record_time = json_data.get('record_time', None)
        
        # Debug info - use len instead of shape for lists
        logger.debug(f"Received audio data: db={db}, time={time_data}, record_time={record_time}")
        logger.debug(f"Data shape: {len(data)} elements")
        
        # ENHANCED SILENCE DETECTION: Check if audio is silent
        if db < SILENCE_THRES:
            logger.debug(f"Audio is silent: {db} dB < {SILENCE_THRES} dB threshold")
            emit_sound_notification('Silent', '1.0', db, time_data, record_time)
            return
            
        # Convert feature data to numpy array - handle both new and old formats
        try:
            # Try to convert directly (if data is a list of floats)
            x = np.array(data, dtype=np.float32)
            logger.debug("Successfully converted data to numpy array directly")
        except (ValueError, TypeError):
            # If direct conversion fails, try parsing from string (old format)
            if isinstance(data, str):
                data = data.strip("[]")
                x = np.fromstring(data, dtype=np.float32, sep=',')
                logger.debug("Successfully converted string data to numpy array")
            else:
                logger.error("Could not convert feature data to numpy array")
                raise ValueError("Could not convert feature data to numpy array")
                
        # Reshape to expected model input: (1, 96, 64, 1)
        x = x.reshape(1, 96, 64, 1)
        logger.debug(f"Successfully reshaped audio features: {x.shape}")
        
        # Make prediction with TensorFlow model
        logger.debug("Making prediction with TensorFlow model...")
        pred = tensorflow_predict(x, db)
        logger.debug(f"Raw prediction shape: {pred[0].shape}")
        
        # Process the prediction results and emit notifications
        # (This is your existing code for handling predictions)
        # ... 
        
    except Exception as e:
        logger.error(f"Error in audio_feature_data handler: {str(e)}", exc_info=True)
        traceback.print_exc()

# Adjust the notification cooldown settings
NOTIFICATION_COOLDOWN_SECONDS = 2.0  # Increased from 1.0 to 2.0 seconds
SPEECH_NOTIFICATION_COOLDOWN = 3.0  # Longer cooldown specifically for speech
UNRECOGNIZED_SOUND_COOLDOWN = 5.0  # Much longer cooldown for "Unrecognized Sound" to reduce frequency

# Add a minimum volume threshold for Unrecognized Sound to reduce frequency
UNRECOGNIZED_SOUND_MIN_DB = 55  # Only emit Unrecognized Sound notifications when above this dB level

# Update should_send_notification function to use sound-specific cooldowns
def should_send_notification(sound_label):
    """
    Check if enough time has passed since the last notification.
    Also prevents duplicate notifications of the same sound in rapid succession.
    
    Args:
        sound_label: The sound label to potentially notify
        
    Returns:
        True if notification should be sent, False otherwise
    """
    global last_notification_time, last_notification_sound
    
    current_time = time.time()
    time_since_last = current_time - last_notification_time
    
    # Always allow the first notification
    if last_notification_time == 0:
        last_notification_time = current_time
        last_notification_sound = sound_label
        return True
    
    # Critical sounds bypass cooldown - fire alarms must always be notified
    if sound_label == "Fire/Smoke Alarm":
        logger.debug(f"Critical sound '{sound_label}' bypassing cooldown")
        last_notification_time = current_time
        last_notification_sound = sound_label
        return True
    
    # Use sound-specific cooldowns
    if sound_label == "Speech":
        required_cooldown = SPEECH_NOTIFICATION_COOLDOWN  # Longer cooldown for speech 
    # For Unrecognized Sound, use much longer cooldown to reduce frequency
    elif sound_label == "Unrecognized Sound":
        required_cooldown = UNRECOGNIZED_SOUND_COOLDOWN
    # Important sounds (like water running, doorbell) use reduced cooldown
    elif sound_label in ["Knocking", "Water Running", "Doorbell In-Use", "Baby Crying", "Phone Ringing", "Alarm Clock"]:
        # For important sounds, apply a consistent cooldown and interrupt other sounds
        required_cooldown = NOTIFICATION_COOLDOWN_SECONDS * 0.75  # Use consistent 1.5s cooldown for all critical sounds
        # Allow interruption of non-critical sounds
        if last_notification_sound not in ["Fire/Smoke Alarm", "Knocking", "Water Running", "Doorbell In-Use", 
                                          "Baby Crying", "Phone Ringing", "Alarm Clock"]:
            required_cooldown = 0.5  # Slightly longer cooldown to interrupt non-critical sounds
    # If we're sending the same sound again, require longer cooldown
    elif sound_label == last_notification_sound:
        required_cooldown = NOTIFICATION_COOLDOWN_SECONDS * 1.5
    else:
        required_cooldown = NOTIFICATION_COOLDOWN_SECONDS
    
    # If enough time has passed, allow the notification
    if time_since_last >= required_cooldown:
        last_notification_time = current_time
        last_notification_sound = sound_label
        logger.debug(f"Allowing notification for '{sound_label}' after {time_since_last:.2f}s")
        return True
    
    # Not enough time has passed
    logger.debug(f"Skipping notification for '{sound_label}' due to cooldown ({time_since_last:.2f}s < {required_cooldown:.2f}s)")
    return False

# Function to emit sound notifications with cooldown management
def emit_sound_notification(label, accuracy, db, time_data="", record_time="", sentiment_data=None):
    """
    Emit sound notification with cooldown management.
    
    Args:
        label: Sound label to emit
        accuracy: Confidence value
        db: Decibel level
        time_data: Time data from client
        record_time: Record time from client
        sentiment_data: Optional sentiment analysis data
    """
    # For Unrecognized Sound, enforce a minimum dB level to reduce notifications in quieter environments
    if label == 'Unrecognized Sound':
        try:
            db_level = float(db)
            if db_level < UNRECOGNIZED_SOUND_MIN_DB:
                logger.debug(f"Suppressing Unrecognized Sound notification due to low volume: {db_level} dB < {UNRECOGNIZED_SOUND_MIN_DB} dB threshold")
                return
        except (ValueError, TypeError):
            # If db can't be converted to float, just continue with notification
            pass
            
    if should_send_notification(label):
        logger.debug(f"Emitting notification: {label} ({accuracy})")
        
        # Prepare notification data
        notification_data = {
            'label': label,
            'accuracy': str(accuracy),
            'db': str(db),
            'time': str(time_data),
            'record_time': str(record_time) if record_time else ''
        }
        
        # Add sentiment data if available
        if sentiment_data and isinstance(sentiment_data, dict):
            # Include transcription and sentiment information
            if 'text' in sentiment_data:
                notification_data['transcription'] = sentiment_data['text']
            
            if 'sentiment' in sentiment_data:
                notification_data['sentiment'] = sentiment_data['sentiment']
                
                # Add emoji for visual representation
                if 'emoji' in sentiment_data['sentiment']:
                    notification_data['emoji'] = sentiment_data['sentiment']['emoji']
            
            # Add speech recognition engine information
            if 'transcription_engine' in sentiment_data:
                notification_data['transcription_engine'] = sentiment_data['transcription_engine']
                logger.info(f"Speech recognized using {sentiment_data['transcription_engine']} engine")
            
            # Include error information for debugging if available
            if 'google_api_error' in sentiment_data and sentiment_data['google_api_error']:
                notification_data['google_api_error'] = sentiment_data['google_api_error']
                logger.debug(f"Google API error included in notification: {sentiment_data['google_api_error']}")
        
        # Emit the notification
        socketio.emit('audio_label', notification_data)
    else:
        logger.debug(f"Notification for '{label}' suppressed due to cooldown")

if __name__ == '__main__':
    # Parse command-line arguments for port configuration
    parser = argparse.ArgumentParser(description='Sonarity Audio Analysis Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--use-google-speech', action='store_true', default=True, help='Use Google Cloud Speech-to-Text (enabled by default)')
    parser.add_argument('--speech-engine', type=str, choices=[SPEECH_ENGINE_AUTO, SPEECH_ENGINE_GOOGLE, SPEECH_ENGINE_VOSK], 
                        default=SPEECH_ENGINE_AUTO, help='Speech recognition engine to use')
    args = parser.parse_args()
    
    # Enable debug mode if specified
    if args.debug:
        DEBUG_MODE = True
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Update speech recognition setting based on command line arguments
    if args.speech_engine:
        SPEECH_RECOGNITION_ENGINE = args.speech_engine
        # Update USE_GOOGLE_SPEECH_API flag based on speech engine selection
        USE_GOOGLE_SPEECH_API = (SPEECH_RECOGNITION_ENGINE != SPEECH_ENGINE_VOSK)
        logger.info(f"Using speech recognition engine: {SPEECH_RECOGNITION_ENGINE}")
    elif not args.use_google_speech:
        # For backward compatibility with --use-google-speech flag
        USE_GOOGLE_SPEECH_API = False
        SPEECH_RECOGNITION_ENGINE = SPEECH_ENGINE_VOSK
        logger.info("Using Vosk for speech recognition (Google Speech API disabled)")
    else:
        USE_GOOGLE_SPEECH_API = True
        logger.info("Using Google Cloud Speech-to-Text for speech recognition")
    
    # Initialize and load all models
    logger.info("Setting up sound recognition models...")
    load_models()
    
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
        
        # Add external IP information
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
    
    # Get port from environment variable if set (for cloud platforms)
    port = int(os.environ.get('PORT', args.port))
    
    # Run the server on all network interfaces (0.0.0.0) so external devices can connect
    socketio.run(app, host='0.0.0.0', port=port, debug=args.debug)