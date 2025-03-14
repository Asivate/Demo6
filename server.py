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
USE_GOOGLE_SPEECH_API = True  # Set to True to use Google Cloud Speech API if available
USE_GOOGLE_SPEECH = False  # Set to True to use Google Cloud Speech-to-Text instead of Whisper

# Import our sentiment analysis modules
from sentiment_analyzer import analyze_sentiment
from speech_to_text import transcribe_audio, SpeechToText

# Conditionally import Google speech module
if USE_GOOGLE_SPEECH_API:
    try:
        from google_speech import transcribe_with_google, GoogleSpeechToText
    except ImportError:
        print("Google Cloud Speech module not available. Using Whisper instead.")
        USE_GOOGLE_SPEECH_API = False

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
            import gc
            gc.collect()
        logger.info(f"Memory cleanup performed (cycle {prediction_counter})")

# Speech recognition settings
USE_GOOGLE_SPEECH = False  # Set to True to use Google Cloud Speech-to-Text instead of Whisper

# Add the current directory to the path so we can import our modules
os.path.dirname(os.path.abspath(__file__))

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
    'hazard-alarm': 0.05,   # Fire alarm - extremely critical with very low threshold
    'knock': 0.05,          # Knock - already implemented
    'doorbell': 0.07,       # Doorbell - important for awareness
    'baby-cry': 0.1,        # Baby crying - important for caregivers
    'water-running': 0.1,   # Water running - requested by user
    'phone-ring': 0.1,      # Phone ringing - important communication
    'alarm-clock': 0.1      # Alarm clock - important for time management
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

# Initialize speech recognition systems
speech_processor = SpeechToText()
google_speech_processor = None  # Will be lazy-loaded when needed

# Load models
def load_models():
    """Load all required models for sound recognition and speech processing."""
    global models, USE_AST_MODEL
    
    # Initialize models dictionary
    models = {
        "tensorflow": None,
        "ast": None,
        "feature_extractor": None,
        "sentiment_analyzer": None,
        "speech_processor": None
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

    # Load the Whisper speech recognition model if needed
    try:
        if not USE_GOOGLE_SPEECH:
            print("Loading Whisper model for speech recognition...")
            speech_processor = SpeechToText()
            print("Whisper model loaded successfully")
            models["speech_processor"] = speech_processor
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        traceback.print_exc()

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
                    emit_sound_notification(human_label, str(context_prediction[m]), combined_db, time_data, record_time)
                    
                    # Continue with speech processing if needed
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

# Helper function to process speech detection with sentiment analysis
def process_speech_with_sentiment(audio_data):
    """
    Process speech audio, transcribe it and analyze sentiment.
    
    Args:
        audio_data: Raw audio data
        
    Returns:
        Dictionary with transcription and sentiment
    """
    # Settings for improved speech processing
    SPEECH_MAX_BUFFER_SIZE = 8  # Number of audio chunks to keep in buffer for speech only
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
        # Use up to 8 chunks for speech recognition
        num_chunks = min(SPEECH_MAX_BUFFER_SIZE, len(process_speech_with_sentiment.recent_audio_buffer))
        logger.info(f"Using concatenated audio from {num_chunks} chunks for speech transcription")
        
        # Concatenate audio chunks
        concatenated_audio = np.concatenate(process_speech_with_sentiment.recent_audio_buffer[-num_chunks:])
    else:
        concatenated_audio = audio_data
    
    # Ensure minimum audio length for better transcription - increase to 4.0 seconds
    min_samples = RATE * 4.0  # At least 4.0 seconds of audio for speech
    if len(concatenated_audio) < min_samples:
        pad_size = int(min_samples) - len(concatenated_audio)
        # Use reflect padding to extend short audio naturally
        concatenated_audio = np.pad(concatenated_audio, (0, pad_size), mode='reflect')
        logger.info(f"Padded speech audio to size: {len(concatenated_audio)} samples ({len(concatenated_audio)/RATE:.1f} seconds)")
    
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
        b, a = signal.butter(4, 80/(RATE/2), 'highpass')  # 80Hz high-pass filter
        filtered_audio = signal.filtfilt(b, a, concatenated_audio)
        logger.info("Applied high-pass filter for noise reduction")
        concatenated_audio = filtered_audio
    except Exception as e:
        logger.warning(f"Error applying high-pass filter: {str(e)}")
    
    logger.info("Transcribing speech to text...")
    
    # Transcribe audio using the selected speech-to-text processor
    if USE_GOOGLE_SPEECH and USE_GOOGLE_SPEECH_API:
        # Use Google Cloud Speech-to-Text if available
        try:
            transcription = transcribe_with_google(concatenated_audio, RATE)
            logger.info(f"Google transcription result: '{transcription}'")
        except Exception as e:
            logger.error(f"Error with Google Speech API: {str(e)}. Falling back to Whisper.")
            transcription = speech_processor.transcribe(concatenated_audio, RATE)
            logger.info(f"Whisper transcription result: '{transcription}'")
    else:
        # Use Whisper (default)
        transcription = speech_processor.transcribe(concatenated_audio, RATE)
        logger.info(f"Whisper transcription result: '{transcription}'")
    
    # Check for valid transcription with sufficient content
    if not transcription:
        logger.info("No valid transcription found")
        return None
    
    # Filter out short or meaningless transcriptions
    common_words = ["the", "a", "an", "and", "but", "or", "if", "then", "so", "to", "of", "for", "in", "on", "at"]
    meaningful_words = [word for word in transcription.lower().split() if word not in common_words]
    
    if len(meaningful_words) < MIN_WORD_COUNT:
        logger.info(f"Transcription has too few meaningful words: '{transcription}'")
        return None
    
    logger.info(f"Valid transcription found: '{transcription}'")
    
    # Analyze sentiment
    sentiment = analyze_sentiment(transcription)
    
    if sentiment:
        result = {
            "text": transcription,
            "sentiment": sentiment
        }
        logger.info(f"Sentiment analysis result: Speech {sentiment['category']} with emoji {sentiment['emoji']}")
        return result
    
    return None

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
        'speech_recognition': 'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper',
        'sentiment_analysis_enabled': True,
        'ip_addresses': ip_addresses,
        'uptime': time.time() - start_time,
        'version': '1.2.0',
        'active_clients': len(active_clients)
    })

@app.route('/api/toggle-speech-recognition', methods=['POST'])
def toggle_speech_recognition():
    """Toggle between Whisper and Google Cloud Speech-to-Text"""
    global USE_GOOGLE_SPEECH
    data = request.get_json()
    
    if data and 'use_google_speech' in data:
        USE_GOOGLE_SPEECH = data['use_google_speech']
        logger.info(f"Speech recognition system changed to: {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}")
        return jsonify({
            "success": True,
            "message": f"Speech recognition system set to {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}",
            "use_google_speech": USE_GOOGLE_SPEECH
        })
    else:
        # Toggle the current value if no specific value provided
        USE_GOOGLE_SPEECH = not USE_GOOGLE_SPEECH
        logger.info(f"Speech recognition system toggled to: {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}")
        return jsonify({
            "success": True,
            "message": f"Speech recognition system toggled to {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}",
            "use_google_speech": USE_GOOGLE_SPEECH
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
    print(f"Client connected: {request.sid}")
    # Send confirmation to client
    emit('server_status', {'status': 'connected', 'message': 'Connected to SoundWatch server'})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle client disconnection events.
    """
    print(f"Client disconnected: {request.sid}")

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
    # Important sounds (like water running, doorbell) use reduced cooldown
    elif sound_label in ["Knocking", "Water Running", "Doorbell In-Use", "Baby Crying", "Phone Ringing", "Alarm Clock"]:
        # For important sounds, apply a shorter cooldown and interrupt other sounds
        required_cooldown = NOTIFICATION_COOLDOWN_SECONDS * 0.5
        # Allow interruption of non-critical sounds
        if last_notification_sound not in ["Fire/Smoke Alarm", "Knocking", "Water Running", "Doorbell In-Use", 
                                          "Baby Crying", "Phone Ringing", "Alarm Clock"]:
            required_cooldown = 0.2  # Very short cooldown to interrupt non-critical sounds
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
def emit_sound_notification(label, accuracy, db, time_data="", record_time=""):
    """
    Emit sound notification with cooldown management.
    
    Args:
        label: Sound label to emit
        accuracy: Confidence value
        db: Decibel level
        time_data: Time data from client
        record_time: Record time from client
    """
    if should_send_notification(label):
        logger.debug(f"Emitting notification: {label} ({accuracy})")
        socketio.emit('audio_label', {
            'label': label,
            'accuracy': str(accuracy),
            'db': str(db),
            'time': str(time_data),
            'record_time': str(record_time) if record_time else ''
        })
    else:
        logger.debug(f"Notification for '{label}' suppressed due to cooldown")

if __name__ == '__main__':
    # Parse command-line arguments for port configuration
    parser = argparse.ArgumentParser(description='Sonarity Audio Analysis Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--use-google-speech', action='store_true', help='Use Google Cloud Speech-to-Text instead of Whisper')
    args = parser.parse_args()
    
    # Enable debug mode if specified
    if args.debug:
        DEBUG_MODE = True
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Update speech recognition setting based on command line argument
    if args.use_google_speech:
        USE_GOOGLE_SPEECH = True
        logger.info("Using Google Cloud Speech-to-Text for speech recognition")
    else:
        USE_GOOGLE_SPEECH = False
        logger.info("Using Whisper for speech recognition")
    
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