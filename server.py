from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context, Response, jsonify
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

# Import our AST model implementation
import ast_model

# Import our sentiment analysis modules
from sentiment_analyzer import analyze_sentiment
from speech_to_text import transcribe_audio, SpeechToText
from google_speech import transcribe_with_google, GoogleSpeechToText

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
tf_graph = tf.Graph()
tf_session = tf.compat.v1.Session(graph=tf_graph)
ast_lock = Lock()
speech_lock = Lock() # Lock for speech processing

# Prediction aggregation system - store recent predictions to improve accuracy
MAX_PREDICTIONS_HISTORY = 3  # Increased from 2 to 3 for more stable non-speech sound detection
SPEECH_PREDICTIONS_HISTORY = 3  # Decreased from 4 to 3 to balance with other sounds
recent_predictions = []  # Store recent prediction probabilities for each sound category
speech_predictions = []  # Separate storage for speech predictions
prediction_lock = Lock()  # Lock for thread-safe access to prediction history

app = Flask(__name__)
app.config['SECRET_KEY'] = 'soundwatch_secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

# contexts
context = homesounds.everything
# use this to change context -- see homesounds.py
active_context = homesounds.everything

# thresholds
PREDICTION_THRES = 0.5  # Changed from 0.15 to 0.5 to match original server
FINGER_SNAP_THRES = 0.4  # Increased from 0.03 to make it harder to detect false finger snaps
DBLEVEL_THRES = -30  # Changed from -60 to -30 to match original server
SILENCE_THRES = -75  # Unchanged (Threshold for silence detection)
SPEECH_SENTIMENT_THRES = 0.6  # Increased from 0.35 to 0.6 to reduce false speech detection
CHOPPING_THRES = 0.7  # Unchanged - higher threshold for chopping sounds
SPEECH_PREDICTION_THRES = 0.7  # Unchanged - higher threshold for speech
SPEECH_DETECTION_THRES = 0.5  # Increased from 0.30 to 0.5 to reduce false speech detection

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
        if (not homesounds_model.is_file()):
            print("Downloading example_model.hdf5 [867MB]: ")
            wget.download(MODEL_URL, MODEL_PATH)
        
        print("Using TensorFlow model: %s" % (model_filename))
        
        try:
            # Try to load the model with the new API
            with tf_graph.as_default():
                with tf_session.as_default():
                    models["tensorflow"] = keras.models.load_model(model_filename)
                    # Make a dummy prediction to ensure the predict function is initialized
                    # This is critical for TensorFlow 1.x compatibility
                    print("Initializing TensorFlow model with a dummy prediction...")
                    dummy_input = np.zeros((1, 96, 64, 1))
                    _ = models["tensorflow"].predict(dummy_input)
                    print("Model prediction function initialized successfully")
            print("TensorFlow model loaded successfully")
            if not USE_AST_MODEL:
                print("TensorFlow model will be used as primary model")
                models["tensorflow"].summary()
        except Exception as e:
            print(f"Error loading TensorFlow model with standard method: {e}")
            # Fallback for older model formats
            try:
                with tf_graph.as_default():
                    with tf_session.as_default():
                        models["tensorflow"] = tf.keras.models.load_model(model_filename, compile=False)
                        # Make a dummy prediction to ensure the predict function is initialized
                        # This is critical for TensorFlow 1.x compatibility
                        print("Initializing TensorFlow model with a dummy prediction...")
                        dummy_input = np.zeros((1, 96, 64, 1))
                        _ = models["tensorflow"].predict(dummy_input)
                        print("Model prediction function initialized successfully")
                print("TensorFlow model loaded with compile=False option")
            except Exception as e2:
                print(f"Error with fallback method: {e2}")
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

# ###########################
# # Setup models - we'll support both old and new models
# ###########################

# Old TensorFlow model settings
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
print("=====")
print("Setting up sound recognition models...")

# Flag to determine which model to use
USE_AST_MODEL = os.environ.get('USE_AST_MODEL', '1') == '1'  # Default to enabled
print(f"AST model {'enabled' if USE_AST_MODEL else 'disabled'} based on environment settings")

# Load the AST model
try:
    print("Loading AST model...")
    # Use SDPA (Scaled Dot Product Attention) for better performance on CPU
    # Define the best AST model for our use case
    ast_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    
    # Check if torch version supports SDPA
    ast_kwargs = {}
    if torch.__version__ >= '2.1.1':
        ast_kwargs["attn_implementation"] = "sdpa"
        print("Using Scaled Dot Product Attention (SDPA) for faster inference")
    
    # Load in a CPU-optimized way - lower precision for faster inference
    # but only if adequate hardware support exists
    # Always use float32 for maximum compatibility
    ast_kwargs = {
        "torch_dtype": torch.float32  # Always use float32 for maximum compatibility
    }
    
    # Load model with optimizations
    with ast_lock:
        # Load the AST model with optimizations
        models["ast"], models["feature_extractor"] = ast_model.load_ast_model(
            model_name=ast_model_name,
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
    if (not homesounds_model.is_file()):
        print("Downloading example_model.hdf5 [867MB]: ")
        wget.download(MODEL_URL, MODEL_PATH)
    
    print("Using TensorFlow model: %s" % (model_filename))
    
    try:
        # Try to load the model with the new API
        with tf_graph.as_default():
            with tf_session.as_default():
                models["tensorflow"] = keras.models.load_model(model_filename)
                # Make a dummy prediction to ensure the predict function is initialized
                # This is critical for TensorFlow 1.x compatibility
                print("Initializing TensorFlow model with a dummy prediction...")
                dummy_input = np.zeros((1, 96, 64, 1))
                _ = models["tensorflow"].predict(dummy_input)
                print("Model prediction function initialized successfully")
        print("TensorFlow model loaded successfully")
        if not USE_AST_MODEL:
            print("TensorFlow model will be used as primary model")
            models["tensorflow"].summary()
    except Exception as e:
        print(f"Error loading TensorFlow model with standard method: {e}")
        # Fallback for older model formats
        try:
            with tf_graph.as_default():
                with tf_session.as_default():
                    models["tensorflow"] = tf.keras.models.load_model(model_filename, compile=False)
                    # Make a dummy prediction to ensure the predict function is initialized
                    # This is critical for TensorFlow 1.x compatibility
                    print("Initializing TensorFlow model with a dummy prediction...")
                    dummy_input = np.zeros((1, 96, 64, 1))
                    _ = models["tensorflow"].predict(dummy_input)
                    print("Model prediction function initialized successfully")
            print("TensorFlow model loaded with compile=False option")
        except Exception as e2:
            print(f"Error with fallback method: {e2}")
            if not USE_AST_MODEL:
                raise Exception("Could not load TensorFlow model with any method, and AST model is not enabled")

print(f"Using {'AST' if USE_AST_MODEL else 'TensorFlow'} model as primary model")

# ##############################
# # Setup Audio Callback
# ##############################
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
        if (context_prediction[m] > PREDICTION_THRES and db > DBLEVEL_THRES):
            print("Prediction: %s (%0.2f)" % (
                homesounds.to_human_labels[active_context[m]], context_prediction[m]))

    print("Raw audio min/max:", np.min(np_wav), np.max(np_wav))
    print("Processed audio shape:", x.shape)

    return (in_data, 0)  # pyaudio.paContinue equivalent


@socketio.on('audio_feature_data')
def handle_source(json_data):
    """Handle audio features sent from client.
    
    Args:
        json_data: JSON object containing audio data
    """
    try:
        data = str(json_data.get('data', ''))
        db = json_data.get('db', 0)
        time_data = json_data.get('time', 0)
        record_time = json_data.get('record_time', None)
        
        # Convert feature data to numpy array
        data = data.strip('[]')  # Remove square brackets if present
        x = np.fromstring(data, dtype=np.float16, sep=',')
        print('Data after to numpy', x)
        
        # Reshape for TensorFlow model
        x = x.reshape(1, 96, 64, 1)
        print('Successfully reshape audio features', x.shape)
        
        # Make prediction with TensorFlow model
        with tf_lock:
            pred = models["tensorflow"].predict(x)
        
        # Find maximum prediction for active context
        context_prediction = np.take(pred[0], [homesounds.labels[x] for x in active_context])
        m = np.argmax(context_prediction)
        print('Max prediction', homesounds.to_human_labels[active_context[m]], context_prediction[m])
        
        # Check if prediction confidence is high enough
        if context_prediction[m] > PREDICTION_THRES and -db > DBLEVEL_THRES:
            label = homesounds.to_human_labels[active_context[m]]
            print(f"Prediction: {label} ({context_prediction[m]:.4f})")
            
            # Special case for "Chopping" - use higher threshold
            if label == "Chopping" and context_prediction[m] < CHOPPING_THRES:
                print(f"Ignoring Chopping sound with confidence {context_prediction[m]:.4f} < {CHOPPING_THRES} threshold")
                socketio.emit('audio_label', {
                    'label': 'Unrecognized Sound',
                    'accuracy': '1.0',
                    'db': str(db),
                    'time': str(time_data),
                    'record_time': str(record_time) if record_time else ''
                })
                return
                
            # Emit the sound label
            socketio.emit('audio_label', {
                'label': label,
                'accuracy': str(context_prediction[m]),
                'db': str(db),
                'time': str(time_data),
                'record_time': str(record_time) if record_time else ''
            })
        else:
            # Sound didn't meet confidence threshold
            socketio.emit('audio_label', {
                'label': 'Unrecognized Sound',
                'accuracy': '1.0',
                'db': str(db),
                'time': str(time_data),
                'record_time': str(record_time) if record_time else ''
            })
    except Exception as e:
        print(f"Error in handle_source: {str(e)}")
        traceback.print_exc()
        socketio.emit('audio_label', {
            'label': 'Error Processing Audio Features',
            'accuracy': '0.0',
            'db': str(db) if 'db' in locals() else '-100'
        })


@socketio.on('audio_data')
def handle_source(json_data):
    """Handle audio data sent from client.
    
    Args:
        json_data: JSON object containing audio data
    """
    try:
        # Extract data from the request
        data = json_data.get('data', [])
        db = json_data.get('db', 0)
        time_data = json_data.get('time', 0)
        record_time = json_data.get('record_time', None)
        
        # Convert audio data to numpy array
        np_wav = np.array(data, dtype=np.float32)
        
        # Debug information
        print(f"Successfully convert to NP rep {np_wav}")
        
        # Calculate decibel level if not provided
        if not db:
            rms = np.sqrt(np.mean(np_wav**2))
            db = dbFS(rms)
        print(f"Db... {db}")
        
        # Check original audio size
        original_size = np_wav.size
        print(f"Original np_wav shape: {np_wav.shape}, size: {original_size}")
        
        # Pad audio if needed to match expected size for models
        if original_size < RATE:
            # Pad with zeros to reach 1 second (16000 samples)
            padding = np.zeros(RATE - original_size)
            np_wav = np.concatenate([np_wav, padding])
            print(f"Padded audio data to size: {RATE} samples (1 second)")
        
        # Convert waveform to examples (spectrogram features)
        x = waveform_to_examples(np_wav, RATE)
        
        # Basic sanity check for x
        if x.shape[0] == 0:
            print("Error: Empty audio features")
            socketio.emit('audio_label', {
                'label': 'Error: Empty Audio Features',
                'accuracy': '0.0',
                'db': str(db)
            })
            return
        
        # Use the first frame (or whichever is appropriate)
        if x.shape[0] > 1:
            print(f"Using first frame from multiple frames: {x.shape[1:3]}")
            x_input = x[0].reshape(1, 96, 64, 1)  # Reshape for model input
        else:
            x_input = x.reshape(1, 96, 64, 1)
            
        print(f"Processed audio data shape: {x_input.shape}")
        
        # Make prediction with TensorFlow model
        print("Making prediction with TensorFlow model...")
        with tf_lock:
            predictions = models["tensorflow"].predict(x_input)
        
        # Debug all raw predictions (before thresholding)
        debug_predictions(predictions[0], homesounds.everything)
        
        # Find maximum prediction for active context
        context_prediction = np.take(
            predictions[0], [homesounds.labels[x] for x in active_context])
        m = np.argmax(context_prediction)
        
        # Check if prediction confidence is high enough
        if context_prediction[m] > PREDICTION_THRES and -db > DBLEVEL_THRES:
            print(f"Top prediction: {homesounds.to_human_labels[active_context[m]]} ({context_prediction[m]:.4f})")
            
            # Process sound-specific logic
            human_label = homesounds.to_human_labels[active_context[m]]
            
            # Special case for "Chopping" - use higher threshold to prevent false positives
            if human_label == "Chopping" and context_prediction[m] < CHOPPING_THRES:
                print(f"Ignoring Chopping sound with confidence {context_prediction[m]:.4f} < {CHOPPING_THRES} threshold")
                socketio.emit('audio_label', {
                    'label': 'Unrecognized Sound',
                    'accuracy': '0.2',
                    'db': str(db)
                })
                return
                
            # Check for speech
            if human_label == "Speech" and context_prediction[m] > SPEECH_SENTIMENT_THRES:
                # Process speech with sentiment analysis
                if sentiment_pipeline is not None:
                    print("Speech detected. Processing sentiment...")
                    sentiment_result = process_speech_with_sentiment(np_wav)
                    
                    if sentiment_result and 'sentiment' in sentiment_result:
                        label = f"Speech {sentiment_result['sentiment']['category']}"
                        socketio.emit('audio_label', {
                            'label': label,
                            'accuracy': str(sentiment_result['sentiment']['confidence']),
                            'db': str(db),
                            'emoji': sentiment_result['sentiment']['emoji'],
                            'transcription': sentiment_result['text'],
                            'emotion': sentiment_result['sentiment']['original_emotion'],
                            'sentiment_score': str(sentiment_result['sentiment']['confidence'])
                        })
                        print(f"EMITTING: {label}")
                        return
            
            # Regular sound detection
            socketio.emit('audio_label', {
                'label': human_label,
                'accuracy': str(context_prediction[m]),
                'db': str(db),
                'time': str(time_data),
                'record_time': str(record_time) if record_time else ''
            })
            print(f"EMITTING: {human_label} ({context_prediction[m]:.2f})")
        else:
            # Sound didn't meet thresholds
            socketio.emit('audio_label', {
                'label': 'Unrecognized Sound',
                'accuracy': '0.5',
                'db': str(db),
                'time': str(time_data),
                'record_time': str(record_time) if record_time else ''
            })
            print(f"EMITTING: Unrecognized Sound (prediction: {context_prediction[m]:.2f}, db: {db})")
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        socketio.emit('audio_label', {
            'label': 'Error Processing',
            'accuracy': '0.0',
            'db': str(db) if 'db' in locals() else '-100'
        })

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
    MIN_WORD_COUNT = 3   # Minimum number of meaningful words for valid transcription
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
    
    logger.info("Transcribing speech to text...")
    
    # Transcribe audio using the selected speech-to-text processor
    if USE_GOOGLE_SPEECH:
        # Use Google Cloud Speech-to-Text
        transcription = transcribe_with_google(concatenated_audio, RATE)
        logger.info(f"Used Google Cloud Speech-to-Text for transcription")
    else:
        # Use Whisper (default)
        transcription = speech_processor.transcribe(concatenated_audio, RATE)
        logger.info(f"Used Whisper for transcription")
    
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
    
    logger.info(f"Transcription: {transcription}")
    
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
    """Render the main page"""
    return render_template('index.html')

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
    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
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
                
                # Apply a slight boost to non-speech sounds to counteract speech bias
                if not is_speech and len(label_list) > 0:
                    speech_idx = -1
                    for i, label in enumerate(label_list):
                        if label == 'speech':
                            speech_idx = i
                            break
                    
                    if speech_idx >= 0 and speech_idx < len(aggregated):
                        # Slightly reduce speech probability to give other sounds a better chance
                        speech_prob = aggregated[speech_idx]
                        aggregated[speech_idx] = speech_prob * 0.8
                
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
            logger.info(f"Aggregation kept same top prediction: {label}, confidence: {new_prediction[orig_top_idx]:.4f} -> {aggregated[orig_top_idx]:.4f}")
        
        return aggregated

if __name__ == '__main__':
    # Parse command-line arguments for port configuration
    parser = argparse.ArgumentParser(description='Sonarity Audio Analysis Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--use-google-speech', action='store_true', help='Use Google Cloud Speech-to-Text instead of Whisper')
    args = parser.parse_args()
    
    # Update speech recognition setting based on command line argument
    if args.use_google_speech:
        USE_GOOGLE_SPEECH = True
        logger.info("Using Google Cloud Speech-to-Text for speech recognition")
    else:
        USE_GOOGLE_SPEECH = False
        logger.info("Using Whisper for speech recognition")
    
    # Initialize and load all models
    print("=====")
    print("Setting up sound recognition models...")
    load_models()
    
    # Get all available IP addresses
    ip_addresses = get_ip_addresses()
    
    print("\n" + "="*60)
    print("SONARITY SERVER STARTED")
    print("="*60)
    
    if ip_addresses:
        print("Server is available at:")
        for i, ip in enumerate(ip_addresses):
            print(f"{i+1}. http://{ip}:{args.port}")
            print(f"   WebSocket: ws://{ip}:{args.port}")
        
        # Add external IP information
        print("\nExternal access: http://34.16.101.179:%d" % args.port)
        print("External WebSocket: ws://34.16.101.179:%d" % args.port)
        
        print("\nPreferred connection address: http://%s:%d" % (ip_addresses[0], args.port))
        print("Preferred WebSocket address: ws://%s:%d" % (ip_addresses[0], args.port))
    else:
        print("Could not determine IP address. Make sure you're connected to a network.")
        print(f"Try connecting to your server's IP address on port {args.port}")
        print("\nExternal access: http://34.16.101.179:%d" % args.port)
        print("External WebSocket: ws://34.16.101.179:%d" % args.port)
    
    print("="*60 + "\n")
    
    # Get port from environment variable if set (for cloud platforms)
    port = int(os.environ.get('PORT', args.port))
    
    # Run the server on all network interfaces (0.0.0.0) so external devices can connect
    socketio.run(app, host='0.0.0.0', port=port, debug=args.debug)
