from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
from keras.models import load_model
import tensorflow as tf
import numpy as np
from vggish_input import waveform_to_examples
import vggish_params
import homesounds
from pathlib import Path
import time
import argparse
import wget
from helpers import dbFS
import os
from google.cloud import speech
from google.cloud import language_v1
import io
import threading
import queue
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re
import sys
import pyaudio
import math
import os
from debug_audio_dump import save_debug_audio

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = "eventlet"  # Use eventlet for better socket handling

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Initialize SocketIO with CORS support and ensure it works with Engine.IO v4
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*", ping_timeout=60)
thread = None
thread_lock = Lock()

# contexts
context = homesounds.everything
# use this to change context -- see homesounds.py
active_context = homesounds.everything

# thresholds - aligned with client settings
PREDICTION_THRES = 0.4  # confidence (matching client)
DBLEVEL_THRES = -40.0  # dB (matching client)
SILENCE_RMS_THRESHOLD = 0.0001  # Threshold to detect silent frames

CHANNELS = 1
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0

###########################
# Initialize Google Cloud clients
###########################
# Initialize Google Cloud clients - use environment credentials
try:
    # Check if credentials are properly set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print(f"Current environment variables: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
        print("Attempting to initialize clients without explicit credentials (will use default credentials if available)...")
    else:
        print(f"Using credentials from: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        # Check if the file exists and is readable
        cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if os.path.exists(cred_path):
            print(f"Credentials file exists at {cred_path}")
            if os.access(cred_path, os.R_OK):
                print("Credentials file is readable.")
            else:
                print("WARNING: Credentials file exists but is not readable!")
        else:
            print(f"WARNING: Credentials file does not exist at {cred_path}")
    
    # Initialize clients
    speech_client = speech.SpeechClient()
    language_client = language_v1.LanguageServiceClient()
    print("Google Cloud clients initialized successfully")
    GOOGLE_CLOUD_ENABLED = True
except Exception as e:
    print("Failed to initialize Google Cloud clients. Speech-to-text and sentiment analysis will be disabled.")
    print(f"Error: {e}")
    print("To fix this issue:")
    print("1. Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set to your credentials JSON file")
    print("2. Verify the credentials file exists and is readable")
    print("3. Make sure the credentials have access to Speech-to-Text and Natural Language APIs")
    GOOGLE_CLOUD_ENABLED = False

# Dual-stream buffers for separate audio processing paths
# Buffer for speech recognition (holds audio data for processing)
speech_buffer = []
speech_buffer_lock = threading.Lock()
speech_processing_queue = queue.Queue()

# Short buffer for sound classification (1 second chunks)
short_buffer = []
# Long buffer for speech transcription (accumulates longer audio)
long_buffer = []
long_buffer_lock = threading.Lock()

# Buffer configuration
LONG_BUFFER_MAX_SECONDS = 10  # Maximum seconds to accumulate for transcription
LONG_BUFFER_SAMPLES = RATE * LONG_BUFFER_MAX_SECONDS  # 16000 samples/sec * 10 seconds = 160000 samples
LONG_BUFFER_MIN_SECONDS = 3  # Minimum seconds required before attempting transcription
LONG_BUFFER_MIN_SAMPLES = RATE * LONG_BUFFER_MIN_SECONDS  # 48000 samples

# Timing controls
last_transcription_time = time.time()
MIN_TRANSCRIPTION_INTERVAL = 2  # Minimum seconds between transcription requests

# Dual-stream buffers for separate audio processing paths
# Short buffer for sound classification (1 second chunks)
short_buffer = []
# Long buffer for speech transcription (accumulates 8 seconds)
long_buffer = []
long_buffer_lock = threading.Lock()
# Buffer configuration
LONG_BUFFER_SECONDS = 8  # Accumulate 8 seconds of audio for better transcription
LONG_BUFFER_SAMPLES = RATE * LONG_BUFFER_SECONDS  # 16000 samples/sec * 8 seconds = 128000 samples
LONG_BUFFER_OVERLAP = 2  # 2 second overlap between consecutive long buffers
LONG_BUFFER_OVERLAP_SAMPLES = RATE * LONG_BUFFER_OVERLAP  # 32000 samples
last_transcription_time = time.time()
MIN_TRANSCRIPTION_INTERVAL = 3  # Minimum seconds between transcription requests

# For streaming speech recognition
STREAMING_LIMIT = 290000  # ~5 minutes
streaming_speech_thread = None
streaming_active = True
streaming_recognizer = None

class SimpleStreamingRecognizer:
    """Simple streaming speech recognition based on proven implementation"""
    
    def __init__(self):
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="phone_call",
            audio_channel_count=CHANNELS,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
        )
        self.audio_queue = queue.Queue()
        self.is_active = True
        self.last_transcript = ""
        self.last_response_time = time.time()
        self.last_audio_time = time.time()
        self.thread = None
        
        # Chunk size management
        self.MAX_CHUNK_SIZE = 15360  # Maximum chunk size in bytes (as per Google's documentation)
        self.optimal_chunk_size = 1600  # Default size (will be adjusted based on sample rate)
        self.adjust_chunk_size()
        
        # VAD parameters
        self.vad_enabled = True
        self.silence_threshold = 0.01  # RMS threshold for silence detection
        self.silence_duration = 0  # Counter for silence duration
        self.max_silence_duration = 5  # Maximum silence duration in seconds before reconnection
        self.speech_active = False
        
        # Keep-alive parameters - IMPROVED
        self.keep_alive_audio = np.zeros(int(RATE * 0.3), dtype=np.float32)  # 300ms of silence
        self.keep_alive_interval = 1.0  # Reduced from 2.0 to 1.0 seconds to prevent timeouts
        self.last_keep_alive_time = time.time()
        
        # Reconnection parameters
        self.reconnect_count = 0
        self.max_reconnect_attempts = 10
        self.reconnect_backoff = 1.0  # Start with 1 second backoff, will increase
        self.max_stream_duration = 240  # Maximum stream duration in seconds (4 minutes)
        self.stream_start_time = time.time()
        
        # New timeout management
        self.api_timeout_threshold = 5.0  # Send keepalive if no data sent for 5 seconds
        self.force_keepalive_timer = threading.Timer(self.api_timeout_threshold, self.send_forced_keepalive)
        self.force_keepalive_timer.daemon = True
        self.force_keepalive_timer.start()

    def send_forced_keepalive(self):
        """Force send a keepalive packet if no audio has been sent recently"""
        try:
            current_time = time.time()
            time_since_last_audio = current_time - self.last_audio_time
            
            if time_since_last_audio > self.api_timeout_threshold and self.is_active:
                print(f"Forcing keepalive packet after {time_since_last_audio:.1f}s of inactivity")
                # Generate a short beep tone instead of silence for better detection
                t = np.linspace(0, 0.2, int(RATE * 0.2), False)
                keep_alive_audio = np.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz tone at 10% volume
                keep_alive_int16 = (keep_alive_audio * 32768).astype(np.int16)
                self.audio_queue.put(keep_alive_int16.tobytes())
                self.last_audio_time = current_time
                self.last_keep_alive_time = current_time
        except Exception as e:
            print(f"Error in send_forced_keepalive: {e}")
        finally:
            # Reschedule timer if still active
            if self.is_active:
                self.force_keepalive_timer = threading.Timer(self.api_timeout_threshold, self.send_forced_keepalive)
                self.force_keepalive_timer.daemon = True
                self.force_keepalive_timer.start()
    
    def adjust_chunk_size(self):
        """Adjust optimal chunk size based on sample rate and channels"""
        # Calculate bytes per second based on sample rate, channels, and bit depth
        bytes_per_second = RATE * CHANNELS * 2  # 2 bytes per sample for 16-bit audio
        
        # Set optimal chunk size to be ~100ms of audio but ensure it's below the maximum
        self.optimal_chunk_size = min(int(bytes_per_second * 0.1), self.MAX_CHUNK_SIZE)
        print(f"Set optimal chunk size to {self.optimal_chunk_size} bytes")
    
    def is_silent(self, audio_data):
        """Detect if audio is silent based on RMS"""
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < self.silence_threshold
    
    def add_audio(self, audio_data):
        """Add audio data to the processing queue with VAD"""
        if not self.is_active:
            return
            
        # Convert float32 [-1.0, 1.0] to int16
        int16_data = (audio_data * 32768).astype(np.int16)
        
        # Voice Activity Detection
        if self.vad_enabled:
            is_silent = self.is_silent(audio_data)
            
            if is_silent:
                self.silence_duration += len(audio_data) / RATE
                
                # If we've been silent for too long, consider sending keep-alive
                if self.silence_duration >= self.keep_alive_interval:
                    current_time = time.time()
                    # Only send keep-alive if we haven't sent one recently
                    if current_time - self.last_keep_alive_time >= self.keep_alive_interval:
                        print(f"Sending keep-alive packet after {self.silence_duration:.1f}s of silence")
                        self.last_keep_alive_time = current_time
                        # Generate a short sine wave instead of zeros for better detection
                        t = np.linspace(0, 0.2, int(RATE * 0.2), False)
                        keep_alive_audio = np.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz tone at 10% volume
                        keep_alive_int16 = (keep_alive_audio * 32768).astype(np.int16)
                        self.audio_queue.put(keep_alive_int16.tobytes())
                        self.last_audio_time = current_time
                
                # If silence persists too long, consider reconnecting the stream
                if self.silence_duration > self.max_silence_duration and self.speech_active:
                    print(f"Silence detected for {self.silence_duration:.1f}s, will initiate graceful reconnection")
                    self.speech_active = False
                    # Don't immediately reconnect - just mark that speech is no longer active
            else:
                # Reset silence duration when sound is detected
                self.silence_duration = 0
                self.speech_active = True
        
        # Check if we need to split the chunk due to size constraints
        audio_bytes = int16_data.tobytes()
        if len(audio_bytes) > self.MAX_CHUNK_SIZE:
            # Split the audio into appropriate sized chunks
            for i in range(0, len(audio_bytes), self.optimal_chunk_size):
                chunk = audio_bytes[i:i+self.optimal_chunk_size]
                self.audio_queue.put(chunk)
            print(f"Split large audio chunk ({len(audio_bytes)} bytes) into {math.ceil(len(audio_bytes)/self.optimal_chunk_size)} smaller chunks")
        else:
            self.audio_queue.put(audio_bytes)
        
        self.last_audio_time = time.time()
    
    def stop(self):
        """Stop the streaming recognition"""
        self.is_active = False
        # Stop the keepalive timer
        if hasattr(self, 'force_keepalive_timer') and self.force_keepalive_timer is not None:
            self.force_keepalive_timer.cancel()
        self.audio_queue.put(None)  # Signal to stop
    
    def start(self):
        """Start the streaming recognition in a background thread"""
        if self.thread is not None and self.thread.is_alive():
            print("Streaming thread already running, stopping it first")
            self.stop()
            time.sleep(0.5)  # Give it a moment to stop
        
        # Create and start the background thread
        def streaming_thread_func():
            """Thread function that handles the actual streaming recognition"""
            try:
                print("Starting streaming recognition thread")
                self.stream_start_time = time.time()
                self.last_audio_time = time.time()
                self.last_response_time = time.time()
                
                while self.is_active:
                    # Create a new streaming session for this iteration
                    requests = (speech.StreamingRecognizeRequest(audio_content=content)
                                for content in self.audio_generator())
                    
                    try:
                        responses = self.client.streaming_recognize(
                            config=self.streaming_config,
                            requests=requests
                        )
                        
                        # Process streaming responses
                        found_result = False
                        for response in responses:
                            self.last_response_time = time.time()
                            
                            # --- DEBUG: Print full response ---
                            print(f"[DEBUG] Google Speech API response: {response}")
                            
                            if not response.results:
                                continue
                            
                            result = response.results[0]
                            if not result.alternatives:
                                continue
                            
                            transcript = result.alternatives[0].transcript
                            
                            # Check if stability field exists before accessing it
                            stability = 0.0
                            try:
                                stability = result.alternatives[0].stability
                            except (AttributeError, ValueError) as e:
                                # Ignore stability field errors
                                pass
                            
                            is_final = result.is_final
                            
                            if is_final:
                                found_result = True
                                self.last_transcript = transcript
                                print(f"Final transcript: {transcript}")
                                
                                # Process for sentiment if appropriate
                                if len(transcript.strip()) > 0:
                                    # Convert to raw audio for sentiment analysis later
                                    audio_np = np.concatenate([
                                        np.frombuffer(chunk, dtype=np.int16) 
                                        for chunk in list(self.audio_generator())
                                    ]).astype(np.float32) / 32768.0
                                    
                                    # Add to processing queue for sentiment analysis
                                    speech_processing_queue.put(audio_np)
                                    
                                    # Generate basic sentiment values for instant feedback
                                    # This provides immediate sentiment while proper analysis happens in background
                                    basic_sentiment_score = 0.0  # Neutral default
                                    basic_emotion = "Neutral"
                                    basic_emoji = "😐"
                                    
                                    # Emit through Socket.IO with consistent field names and additional sentiment fields
                                    socketio.emit('transcript_update', {
                                        'transcription': transcript,  # Keep consistent with other places
                                        'confidence': 1.0,           # High confidence for final results
                                        'emotion': basic_emotion,    # Placeholder until real sentiment analysis
                                        'emoji': basic_emoji,        # Placeholder emoji
                                        'sentiment_score': basic_sentiment_score,
                                        'sentiment_magnitude': 0.0,
                                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        'is_final': True
                                    })
                            elif stability > 0.5:
                                # Only emit interim results with decent stability
                                print(f"Interim transcript (stability {stability:.2f}): {transcript}")
                                
                                # Provide basic sentiment values for interim results too
                                basic_sentiment_score = 0.0  # Neutral default
                                basic_emotion = "Neutral"
                                basic_emoji = "😐"
                                
                                socketio.emit('transcript_update', {
                                    'transcription': transcript,    # Keep consistent field naming
                                    'confidence': stability,        # Use stability as confidence
                                    'emotion': basic_emotion,       # Basic placeholder
                                    'emoji': basic_emoji,           # Basic placeholder
                                    'sentiment_score': basic_sentiment_score,
                                    'sentiment_magnitude': 0.0,
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                    'is_final': False
                                })
                        # --- DEBUG: If no result found, log and reference the debug audio file ---
                        if not found_result:
                            print("[DEBUG][WARNING] No recognition results returned from API for this buffer. Check the corresponding debug audio file.")
                    except Exception as e:
                        print(f"[DEBUG][ERROR] Exception in streaming_recognize: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # If we got here, stream ended or timed out but thread is still active
                    # Check if we should reconnect
                    if self.is_active:
                        print("Stream ended, reconnecting...")
                        self.reconnect_count += 1
                        self.stream_start_time = time.time()  # Reset stream age
                
                print("Streaming recognition thread ended")
            
            except Exception as e:
                print(f"Unexpected error in streaming thread: {e}")
                import traceback
                traceback.print_exc()
        
        # Create and start the thread
        self.thread = threading.Thread(target=streaming_thread_func)
        self.thread.daemon = True
        self.thread.start()
        print("Streaming recognition thread started")
    
    def audio_generator(self):
        """Generate audio chunks from the queue with improved keep-alive handling"""
        buffer = b''
        last_yield_time = time.time()
        
        while self.is_active:
            # Check if stream has been active too long and needs rotation
            current_time = time.time()
            if current_time - self.stream_start_time > self.max_stream_duration:
                print(f"Stream active for {self.max_stream_duration}s, initiating planned rotation")
                return  # End this stream to allow a new one to start
            
            # Force send data if we haven't sent anything for too long
            # This is a safety mechanism to prevent API timeouts
            if current_time - last_yield_time > 3.0:  # 3 seconds max without sending data
                if buffer:
                    print(f"Yielding accumulated buffer to prevent timeout ({len(buffer)} bytes)")
                    yield buffer
                    buffer = b''
                    last_yield_time = current_time
                else:
                    # If no data in buffer, create a minimal audio packet
                    print("Creating minimal keep-alive packet to prevent timeout")
                    minimal_audio = np.zeros(800, dtype=np.int16).tobytes()  # Very small packet
                    yield minimal_audio
                    last_yield_time = current_time
            
            try:
                chunk = self.audio_queue.get(timeout=0.5)  # Short timeout to check for stream duration
                if chunk is None:
                    return
                
                # Accumulate audio in buffer
                buffer += chunk
                
                # If buffer reaches optimal size, yield it
                if len(buffer) >= self.optimal_chunk_size:
                    yield buffer
                    buffer = b''
                    last_yield_time = time.time()
                
                # Get any additional chunks without blocking
                while True:
                    try:
                        chunk = self.audio_queue.get(block=False)
                        if chunk is None:
                            return
                        
                        # If adding this chunk would exceed max size, yield current buffer first
                        if len(buffer) + len(chunk) > self.MAX_CHUNK_SIZE:
                            if buffer:  # Only yield if buffer has content
                                yield buffer
                                buffer = b''
                                last_yield_time = time.time
                        
                        buffer += chunk
                    except queue.Empty:
                        break
                
                # If we have accumulated data, yield it
                if buffer:
                    yield buffer
                    buffer = b''
                    last_yield_time = time.time()
                    
            except queue.Empty:
                # Timeout occurred, continue loop to check if stream needs rotation
                continue
            
        # Yield any remaining data before exiting
        if buffer:
            yield buffer

def start_streaming_recognition():
    """Start the streaming recognition thread using the simplified implementation"""
    global streaming_recognizer, streaming_active, long_buffer, last_transcription_time
    
    if not GOOGLE_CLOUD_ENABLED:
        print("Google Cloud services not enabled. Cannot start streaming recognition.")
        return
    
    streaming_active = True
    
    # Reset our long buffer and timing variables
    with long_buffer_lock:
        long_buffer = []
    last_transcription_time = time.time()
    
    # Use the simpler streaming recognizer implementation
    streaming_recognizer = SimpleStreamingRecognizer()
    
    # Configure for enhanced speech recognition
    # These settings improve handling of longer audio segments
    streaming_recognizer.config.enable_automatic_punctuation = True
    streaming_recognizer.config.enable_word_time_offsets = True
    streaming_recognizer.config.model = "phone_call"  # Use phone_call model for better results with extended audio
    
    # Print information about the environment
    print("Starting dual-stream speech recognition with the following settings:")
    print(f"- Sample rate: {RATE} Hz")
    print(f"- Channels: {CHANNELS}")
    print(f"- Long buffer size: {LONG_BUFFER_SECONDS} seconds ({LONG_BUFFER_SAMPLES} samples)")
    print(f"- Buffer overlap: {LONG_BUFFER_OVERLAP} seconds")
    print(f"- Google credentials: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    
    # Start in a background thread
    streaming_recognizer.start()
    
    print("Streaming recognition thread started with dual-buffer architecture")
    return streaming_recognizer

def stop_streaming_recognition():
    """Stop the streaming recognition"""
    global streaming_recognizer, streaming_active
    if streaming_recognizer:
        streaming_active = False
        streaming_recognizer.stop()
        print("Streaming recognition stopped")

def restart_streaming_recognition():
    """Restart the streaming recognition system"""
    stop_streaming_recognition()
    time.sleep(2)  # Allow time for cleanup
    return start_streaming_recognition()

###########################
# Download model, if it doesn't exist
###########################
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
print("=====")
print("2 / 2: Checking model... ")
print("=====")
model_filename = "models/example_model.hdf5"
# Create the models directory if it doesn't exist
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
homesounds_model = Path(model_filename)
if (not homesounds_model.is_file()):
    print("Downloading example_model.hdf5 [867MB]: ")
    wget.download(MODEL_URL, MODEL_PATH)

##############################
# Load Deep Learning Model - Fixed TensorFlow Session Management
##############################
print("Using deep learning model: %s" % (model_filename))
# Create a session that will be used throughout the application
sess = tf.compat.v1.Session()
# Set this session as the default for this application
tf.compat.v1.keras.backend.set_session(sess)
# Load the model within this session
with sess.as_default():
    model = load_model(model_filename)
    graph = tf.compat.v1.get_default_graph()

##############################
# Setup Audio Callback
##############################

@socketio.on('sentiment_audio_data')
def handle_sentiment_audio(json_data):
    """
    Receives audio data from the client specifically for sentiment analysis (separate from classification).
    Expects json_data['data'] as a string of comma-separated int16 samples.
    """
    try:
        print(f"[sentiment_audio_data] Received sentiment audio from client: {request.sid[:10]}...")
        data = str(json_data['data'])
        data = data[1:-1]
        np_wav = np.fromstring(data, dtype=np.int16, sep=',') / 32768.0  # Convert to [-1.0, +1.0]

        # Process for sentiment analysis
        transcript_entry = process_speech_for_sentiment(np_wav)
        if transcript_entry:
            # Optionally emit a dedicated sentiment notification event (for watch/mobile notifications)
            socketio.emit('sentiment_notification', transcript_entry, room=request.sid)
            print(f"[sentiment_audio_data] Emitted sentiment_notification: {transcript_entry}")
        else:
            print("[sentiment_audio_data] No transcript entry generated.")
    except Exception as e:
        print(f"[sentiment_audio_data] Error: {e}")
        import traceback
        traceback.print_exc()

def audio_samples(in_data, frame_count, time_info, status_flags):
    global graph, model, sess
    np_wav = np.fromstring(in_data, dtype=np.int16) / \
        32768.0  # Convert to [-1.0, +1.0]
    # Compute RMS and convert to dB
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)

    # Make predictions
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    with sess.as_default():
        with graph.as_default():
            if x.shape[0] != 0:
                x = x.reshape(len(x), 96, 64, 1)
                print('Reshape x successful', x.shape)
                pred = model.predict(x)
                predictions.append(pred)
            print('Prediction succeeded')
            for prediction in predictions:
                context_prediction = np.take(
                    prediction[0], [homesounds.labels[x] for x in active_context])
                m = np.argmax(context_prediction)
                if (context_prediction[m] > PREDICTION_THRES and db > DBLEVEL_THRES):
                    print("Prediction: %s (%0.2f)" % (
                        homesounds.to_human_labels[active_context[m]], context_prediction[m]))

    return (in_data, pyaudio.paContinue)


@socketio.on('audio_feature_data')
def handle_source_features(json_data):
    '''Acoustic features: processes a list of features for predictions'''
    global graph, model, sess
    
    # Parse the input
    data = str(json_data['data'])
    data = data[1:-1]
    print('Data before transform to np', data)
    try:
        x = np.fromstring(data, dtype=np.float16, sep=',')
        print('Data after to numpy:', x.shape, 'min:', x.min(), 'max:', x.max())
    
        # Handle multiple frames if present
        if len(x) > vggish_params.NUM_FRAMES * vggish_params.NUM_BANDS:
            print(f"Warning: Received {len(x)} values, expected {vggish_params.NUM_FRAMES * vggish_params.NUM_BANDS}")
            x = x[:vggish_params.NUM_FRAMES * vggish_params.NUM_BANDS]
        elif len(x) < vggish_params.NUM_FRAMES * vggish_params.NUM_BANDS:
            print(f"Warning: Received {len(x)} values, padding to {vggish_params.NUM_FRAMES * vggish_params.NUM_BANDS}")
            padding = np.zeros(vggish_params.NUM_FRAMES * vggish_params.NUM_BANDS - len(x))
            x = np.concatenate((x, padding))
        
        # Reshape using vggish_params constants
        x = x.reshape(1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1)
        print('Successfully reshaped audio features to', x.shape)
    
        predictions = []
        
        # Get the dB level from the JSON data or use a default value
        db = float(json_data.get('db', DBLEVEL_THRES))
        
        # Use the global session and graph context to run predictions
        with sess.as_default():
            with graph.as_default():
                pred = model.predict(x)
                predictions.append(pred)
                
                for prediction in predictions:
                    context_prediction = np.take(
                        prediction[0], [homesounds.labels[x] for x in active_context])
                    m = np.argmax(context_prediction)
                    print('Max prediction', str(
                        homesounds.to_human_labels[active_context[m]]), str(context_prediction[m]))
                    
                    # Modified condition - look at db and confidence together for debugging
                    print(f"Prediction confidence: {context_prediction[m]}, Threshold: {PREDICTION_THRES}")
                    print(f"db level: {db}, Threshold: {DBLEVEL_THRES}")
                    
                    # Original condition was too strict - many sounds were being classified as "Unrecognized"
                    if context_prediction[m] > PREDICTION_THRES:
                        # Send result using both event names to support both implementations
                        prediction_result = {
                            'label': str(homesounds.to_human_labels[active_context[m]]),
                            'accuracy': str(context_prediction[m]),
                            'db': str(db)
                        }
                        
                        # Emit as 'audio_label' for the updated client
                        socketio.emit('audio_label', prediction_result, room=request.sid)
                        
                        # Also emit as 'predict_sound' for backward compatibility
                        socketio.emit('predict_sound', {
                            'label': str(homesounds.to_human_labels[active_context[m]]),
                            'confidence': float(context_prediction[m])
                        }, room=request.sid)
                        
                        print("Prediction: %s (%0.2f)" % (
                            homesounds.to_human_labels[active_context[m]], context_prediction[m]))
                    else:
                        # Send unrecognized sound notification
                        socketio.emit('audio_label',
                                    {
                                        'label': 'Unrecognized Sound',
                                        'accuracy': '1.0',
                                        'db': str(db)
                                    },
                                    room=request.sid)
    except Exception as e:
        print(f"Error during feature prediction: {e}")
        socketio.emit('audio_label',
                    {
                        'label': 'Processing Error',
                        'accuracy': '1.0',
                        'error': str(e),
                        'db': str('-60.0')  # Default value if db extraction fails
                    },
                    room=request.sid)

def assess_audio_quality(audio_data):
    """
    Analyze audio data to determine if it's suitable for speech recognition
    Returns: (is_suitable, reason, processed_audio)
    """
    # Calculate audio metrics
    rms = np.sqrt(np.mean(audio_data**2))
    peak = np.max(np.abs(audio_data))
    db = dbFS(rms)
    
    # Check for silence
    if rms < 0.001:  # Extremely quiet
        return False, f"Audio too quiet (RMS: {rms:.6f}, dB: {db:.2f})", None
        
    # Check for near-zero audio
    if peak < 0.01:  # Very low peak amplitude
        return False, f"Audio peak too low (peak: {peak:.6f})", None
    
    # Process the audio to improve quality
    processed_audio = audio_data.copy()
    
    # Apply noise reduction (cut off very quiet sounds)
    noise_floor = 0.002  # Lower from original 0.005 to be less aggressive
    processed_audio[np.abs(processed_audio) < noise_floor] = 0
    
    # Apply normalization if audio is quiet but not silent
    if peak < 0.3 and peak > 0.01:
        gain_factor = min(0.5 / peak, 5.0)  # Target 50% of max, but cap at 5x gain
        processed_audio = np.clip(processed_audio * gain_factor, -1.0, 1.0)
        print(f"Applied gain of {gain_factor:.2f} to audio (original peak: {peak:.3f})")
    
    # Calculate processed metrics
    processed_rms = np.sqrt(np.mean(processed_audio**2))
    processed_peak = np.max(np.abs(processed_audio))
    processed_db = dbFS(processed_rms)
    
    print(f"Audio quality: Original(RMS: {rms:.4f}, peak: {peak:.4f}, dB: {db:.1f}) → " +
          f"Processed(RMS: {processed_rms:.4f}, peak: {processed_peak:.4f}, dB: {processed_db:.1f})")
    
    return True, "Audio suitable for recognition", processed_audio

@socketio.on('audio_data')
def handle_source(json_data):
    """
    Handles audio data from the client for sound event detection and speech transcription
    Implements a dual-stream approach: short chunks for sound classification and longer buffers for transcription
    """
    try:
        print(f"Received audio data from client: {request.sid[:10]}...")
        data = str(json_data['data'])
        data = data[1:-1]
        global graph, model, sess, streaming_recognizer, long_buffer, last_transcription_time
        np_wav = np.fromstring(data, dtype=np.int16, sep=',') / \
            32768.0  # Convert to [-1.0, +1.0]
        
        # Compute RMS and convert to dB
        rms = np.sqrt(np.mean(np_wav**2))
        db = dbFS(rms)
        
        # Log basic audio metrics
        print(f'Audio received - length: {len(np_wav)}, RMS: {rms:.4f}, dB: {db:.1f}')
        
        # ======= DUAL STREAM PROCESSING ======= #
        # 1. SHORT STREAM: Process for sound classification (keep as is)
        # 2. LONG STREAM: Accumulate audio for speech transcription
        
        # Add current audio to the long buffer for transcription
        with long_buffer_lock:
            # Save 800 samples for overlap between chunks (helps with continuity)
            if len(np_wav) > 800:
                print(f"Saved 800 samples for overlap")
                overlap_samples = np_wav[-800:]
            else:
                overlap_samples = np_wav
                
            # Add the new audio to the long buffer
            long_buffer.append(np_wav)
            
            # Calculate total samples in long buffer
            total_samples = sum(len(chunk) for chunk in long_buffer)
            
            # If buffer exceeds maximum size, trim it while maintaining overlap
            if total_samples > LONG_BUFFER_SAMPLES:
                # Keep removing chunks from the start until we're under the limit
                while sum(len(chunk) for chunk in long_buffer) > LONG_BUFFER_SAMPLES and len(long_buffer) > 1:
                    long_buffer.pop(0)
            
            # Check if we have enough audio and enough time has passed since last transcription
            current_time = time.time()
            buffer_duration = sum(len(chunk) for chunk in long_buffer) / RATE
            time_since_last_transcription = current_time - last_transcription_time
            
            has_minimum_audio = sum(len(chunk) for chunk in long_buffer) >= LONG_BUFFER_MIN_SAMPLES
            enough_time_passed = time_since_last_transcription >= MIN_TRANSCRIPTION_INTERVAL
            
            # Process the long buffer for transcription if conditions are met
            if has_minimum_audio and enough_time_passed and GOOGLE_CLOUD_ENABLED:
                # Concatenate all chunks in the buffer
                long_audio = np.concatenate(long_buffer)

                # --- DEBUG: Save audio for inspection ---
                try:
                    debug_path = save_debug_audio(long_audio, rate=RATE, tag=f"buffer_{int(time.time())}")
                    print(f"[DEBUG] Saved long buffer audio for ASR to: {debug_path}")
                except Exception as e:
                    print(f"[DEBUG] Failed to save debug audio: {e}")

                # Send to streaming recognizer for real-time transcription
                if streaming_active and 'streaming_recognizer' in globals() and streaming_recognizer is not None:
                    print(f"Send raw audio to server")
                    print(f"Using local socket (connected: {streaming_recognizer.is_active})")
                    streaming_recognizer.add_audio(long_audio)
                    last_transcription_time = current_time
        
        # Skip sound classification processing if the audio is silent (very low RMS)
        if rms < SILENCE_RMS_THRESHOLD:
            print(f"Detected silent audio frame. Skipping classification.")
            socketio.emit('audio_label',
                        {
                            'label': 'Silent',
                            'accuracy': '1.0',
                            'db': str(db)
                        },
                        room=request.sid)
            return
        
        # Ensure we have enough audio data for feature extraction
        if len(np_wav) < 16000:
            print(f"Warning: Audio length {len(np_wav)} is less than 1 second (16000 samples)")
            # Pad with zeros to reach 1 second if needed
            padding = np.zeros(16000 - len(np_wav))
            np_wav = np.concatenate((np_wav, padding))
            print(f"Padded audio to {len(np_wav)} samples")
        
        # Sound classification processing
        try:
            # Make predictions
            print('Making sound classification prediction...')
            x = waveform_to_examples(np_wav, RATE)
            
            # Handle multiple frames - take the first frame if multiple are generated
            if x.shape[0] > 1:
                print(f"Multiple frames detected ({x.shape[0]}), using first frame")
                x = x[0:1]
            
            # Check if x is empty (shape[0] == 0)
            if x.shape[0] == 0:
                print("Warning: waveform_to_examples returned empty array. Creating dummy features.")
                # Create dummy features for testing - one frame of the right dimensions
                x = np.zeros((1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS))
            
            # Add the channel dimension required by the model
            x = x.reshape(1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1)
            
            # Use the global session and graph context to run predictions
            with sess.as_default():
                with graph.as_default():
                    pred = model.predict(x)
                    
                    context_prediction = np.take(
                        pred[0], [homesounds.labels[x] for x in active_context])
                    m = np.argmax(context_prediction)
                    print(f'Sound classification: {homesounds.to_human_labels[active_context[m]]} ({context_prediction[m]:.4f})')
                    
                    # If confidence is high enough, send the label
                    if context_prediction[m] > PREDICTION_THRES:
                        # Send result using both event names for compatibility
                        prediction_result = {
                            'label': str(homesounds.to_human_labels[active_context[m]]),
                            'accuracy': str(context_prediction[m]),
                            'db': str(db)
                        }
                        
                        # Emit as 'audio_label' for updated client
                        socketio.emit('audio_label', prediction_result, room=request.sid)
                        
                        # Also emit as 'predict_sound' for backward compatibility
                        socketio.emit('predict_sound', {
                            'label': str(homesounds.to_human_labels[active_context[m]]),
                            'confidence': float(context_prediction[m])
                        }, room=request.sid)
                        
                        print(f"Emitted prediction via both audio_label and predict_sound events: {prediction_result['label']}")
                    else:
                        socketio.emit('audio_label',
                                    {
                                        'label': 'Unrecognized Sound',
                                        'accuracy': '1.0',
                                        'db': str(db)
                                    },
                                    room=request.sid)
        except Exception as e:
            print(f"Error during prediction: {e}")
            socketio.emit('audio_label',
                        {
                            'label': 'Processing Error',
                            'accuracy': '1.0',
                            'error': str(e),
                            'db': str(db)
                        },
                        room=request.sid)
    except Exception as e:
        print(f"Error processing audio data: {e}")
        socketio.emit('audio_label',
                    {
                        'label': 'Processing Error',
                        'accuracy': '1.0',
                        'error': str(e),
                        'db': str(db)
                    },
                    room=request.sid)


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count},
                      namespace='/test')

def get_sentiment_emoji(score):
    """Convert sentiment score to appropriate emoji"""
    if score > 0.7:
        return "😄"
    elif score > 0.3:
        return "🙂"
    elif score > -0.3:
        return "😐"
    elif score > -0.7:
        return "😕"
    else:
        return "😢"
        
def get_sentiment_emotion(score):
    """Convert sentiment score to emotion label"""
    if score > 0.7:
        return "Very Positive"
    elif score > 0.3:
        return "Positive"
    elif score > -0.3:
        return "Neutral"
    elif score > -0.7:
        return "Negative"
    else:
        return "Very Negative"

def process_speech_for_sentiment(audio_data):
    """Process audio data for speech recognition and sentiment analysis"""
    if not GOOGLE_CLOUD_ENABLED:
        print("Google Cloud services not enabled. Skipping speech processing.")
        return
    
    try:
        # Convert to bytes for Google API
        audio_bytes = audio_data.astype(np.int16).tobytes()
        
        # Calculate audio metrics for debugging
        rms = np.sqrt(np.mean(audio_data**2))
        db = dbFS(rms)
        peak = np.max(np.abs(audio_data))
        print(f"Audio metrics before API call: RMS={rms:.6f}, dB={db:.2f}, peak={peak:.6f}, length={len(audio_data)}")
        
        # Process with Speech API
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="phone_call",
            audio_channel_count=CHANNELS,
        )
        
        print("Sending audio to Google Speech-to-Text API...")
        response = speech_client.recognize(config=config, audio=audio)
        
        # Detailed logging of the response
        print(f"Speech API response: {response}")
        if hasattr(response, 'results') and response.results:
            for i, result in enumerate(response.results):
                print(f"Result {i+1}: {result}")
                if result.alternatives:
                    for j, alt in enumerate(result.alternatives):
                        print(f"  Alternative {j+1}: '{alt.transcript}' (confidence: {alt.confidence})")
                else:
                    print("  No alternatives found in this result")
        else:
            print("No recognition results returned from API")
        
        if not response.results or not response.results[0].alternatives:
            print("No speech detected in audio segment")
            return
            
        transcript = response.results[0].alternatives[0].transcript
        confidence = response.results[0].alternatives[0].confidence
        
        # If transcript is empty, skip sentiment analysis
        if not transcript.strip():
            print("Empty transcript, skipping sentiment analysis")
            return
            
        print(f"Transcript: '{transcript}' (confidence: {confidence})")
            
        # Process with Natural Language API for sentiment
        document = language_v1.Document(
            content=transcript,
            type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        print("Analyzing sentiment with Google Natural Language API...")
        sentiment_response = language_client.analyze_sentiment(document=document)
        sentiment = sentiment_response.document_sentiment
        
        # Map sentiment score to emoji and emotion
        emoji = get_sentiment_emoji(sentiment.score)
        emotion = get_sentiment_emotion(sentiment.score)
        
        print(f"Sentiment analysis: score={sentiment.score}, magnitude={sentiment.magnitude}, emotion={emotion}, emoji={emoji}")
        
        # Calculate approximate dB level
        rms = np.sqrt(np.mean(audio_data**2))
        db = dbFS(rms)
        
        # Create timestamp for the transcript
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Create transcript entry for history
        transcript_entry = {
            "transcription": transcript,  # Keep consistent with streaming format
            "confidence": float(confidence),
            "emotion": emotion,
            "emoji": emoji,
            "sentiment_score": float(sentiment.score),
            "sentiment_magnitude": float(sentiment.magnitude),
            "timestamp": timestamp,
            "is_final": True   # Always final for sentiment analysis
        }
        
        # Send to connected clients
        socketio.emit('audio_label', {
            'label': 'Speech',
            'accuracy': '1.0',
            'db': str(db),
            'emoji': emoji,
            'emotion': emotion,
            'transcription': transcript,  # Consistent field naming
            'confidence': confidence,
            'sentiment_score': sentiment.score,
            'sentiment_magnitude': sentiment.magnitude,
            'timestamp': timestamp
        })
        
        # Also emit a transcript update event for clients that maintain transcript history
        socketio.emit('transcript_update', transcript_entry)
        
        return transcript_entry
        
    except Exception as e:
        print(f"Error processing speech: {e}")
        import traceback
        traceback.print_exc()
        return None

# Speech processing thread function
def speech_processing_thread_func():
    """Background thread that processes the speech queue"""
    print("Speech processing thread started")
    while True:
        try:
            audio_data = speech_processing_queue.get(timeout=1)
            process_speech_for_sentiment(audio_data)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in speech processing thread: {e}")

# Start the speech processing thread
speech_thread = threading.Thread(target=speech_processing_thread_func)
speech_thread.daemon = True
speech_thread.start()

@app.route('/test_sentiment')
def test_sentiment():
    """
    Test endpoint to verify sentiment analysis functionality
    Access this at http://localhost:8080/test_sentiment
    """
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Cannot test sentiment analysis."
    
    try:
        # Test phrases with different sentiments
        test_phrases = [
            "I am very happy today!",
            "This is terrible, I don't like it at all.",
            "The weather seems fine today."
        ]
        
        results = []
        for phrase in test_phrases:
            # Process with Natural Language API for sentiment
            document = language_v1.Document(
                content=phrase,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            sentiment = language_client.analyze_sentiment(document=document).document_sentiment
            
            # Map sentiment score to emoji and emotion
            emoji = get_sentiment_emoji(sentiment.score)
            emotion = get_sentiment_emotion(sentiment.score)
            
            results.append({
                'phrase': phrase,
                'sentiment_score': sentiment.score,
                'sentiment_magnitude': sentiment.magnitude,
                'emotion': emotion,
                'emoji': emoji
            })
        
        # Format results as HTML
        html = "<h1>Sentiment Analysis Test</h1>"
        html += "<p>Testing Google Cloud Natural Language API sentiment analysis</p>"
        html += "<ul>"
        for result in results:
            html += f"<li><strong>Phrase:</strong> '{result['phrase']}'<br>"
            html += f"<strong>Sentiment:</strong> {result['emotion']} {result['emoji']}<br>"
            html += f"<strong>Score:</strong> {result['sentiment_score']}, "
            html += f"<strong>Magnitude:</strong> {result['sentiment_magnitude']}</li>"
        html += "</ul>"
        
        return html
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error testing sentiment analysis: {str(e)}<br><pre>{error_trace}</pre>"

@app.route('/test_speech')
def test_speech():
    """
    Test endpoint that simulates speech recognition with a test audio file
    This helps isolate and test the Google Speech API integration
    Access this at http://localhost:8080/test_speech
    """
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Cannot test speech recognition."
    
    try:
        # Create a test tone that resembles speech (sine wave with varying frequency)
        duration = 3  # seconds
        fs = RATE  # sample rate
        samples = int(fs * duration)
        
        # Create sample speech-like audio (frequency modulated sine wave)
        t = np.linspace(0, duration, samples, False)
        
        # Option 1: Generate synthetic speech-like audio
        # This creates a frequency modulated tone that has some speech-like characteristics
        carrier = 220  # base frequency in Hz
        modulator = 10  # modulation frequency
        index = 5  # modulation index
        audio = 0.5 * np.sin(2 * np.pi * carrier * t + index * np.sin(2 * np.pi * modulator * t))
        
        # Add some "syllable" like amplitude modulation
        syllable_env = np.ones_like(t)
        for i in range(5):  # Create 5 "syllables"
            start = int(i * samples/5)
            mid = int((i + 0.5) * samples/5)
            end = int((i + 1) * samples/5)
            syllable_env[start:mid] = np.linspace(0.2, 1.0, mid-start)
            syllable_env[mid:end] = np.linspace(1.0, 0.2, end-mid)
        
        audio = audio * syllable_env
        
        # Add a little noise
        noise = np.random.normal(0, 0.01, samples)
        audio = audio + noise
        
        # Ensure it's within [-1, 1]
        audio = np.clip(audio, -1, 1)
        
        # Convert to int16 for the API
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        
        print(f"TEST SPEECH - Created {duration}s test audio sample")
        
        # Create audio content
        audio = speech.RecognitionAudio(content=audio_bytes)
        
        # Send synchronous request
        print("TEST SPEECH - Sending synchronous recognition request")
        response = speech_client.recognize(config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="phone_call",
            audio_channel_count=CHANNELS,
        ), audio=audio)
        
        # Process response
        if response.results:
            for result in response.results:
                for alternative in result.alternatives:
                    print(f"TEST SPEECH - Transcript: '{alternative.transcript}'")
                    print(f"TEST SPEECH - Confidence: {alternative.confidence}")
            print("TEST SPEECH - Successfully received transcription from API")
        else:
            print("TEST SPEECH - No transcription results returned from API")
            
        # Try streaming recognition as well
        print("TEST SPEECH - Now testing streaming recognition")
        
        # Configure streaming recognition
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="en-US",
                enable_automatic_punctuation=True,
                model="phone_call",
                audio_channel_count=CHANNELS,
            ),
            interim_results=True,
        )
        
        # Split audio into chunks to simulate streaming
        chunk_duration_ms = 100  # 100ms chunks
        samples_per_chunk = int(RATE * chunk_duration_ms / 1000)
        chunks = [audio_int16[i:i+samples_per_chunk] for i in range(0, len(audio_int16), samples_per_chunk)]
        
        # Create generator for chunks
        def audio_chunk_generator():
            for chunk in chunks:
                yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
        
        # Start streaming recognition
        print(f"TEST SPEECH - Sending {len(chunks)} streaming chunks")
        streaming_responses = speech_client.streaming_recognize(streaming_config, audio_chunk_generator())
        
        # Process streaming responses
        any_results = False
        for response in streaming_responses:
            if response.results:
                any_results = True
                for result in response.results:
                    for alternative in result.alternatives:
                        result_type = "Final" if result.is_final else "Interim"
                        print(f"TEST SPEECH - {result_type} transcript: '{alternative.transcript}'")
        
        if not any_results:
            print("TEST SPEECH - No streaming transcription results returned from API")
        
        print("TEST SPEECH - Test completed")
        
    except Exception as e:
        print(f"TEST SPEECH - Error during test: {e}")
        import traceback
        traceback.print_exc()

@app.route('/test_direct_mic')
def test_direct_mic():
    """
    Test endpoint that launches a direct microphone stream for testing
    This mimics the user's working example directly for maximum compatibility
    """
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Cannot test speech recognition."
    
    # Run the test in a separate thread
    thread = threading.Thread(target=run_direct_mic_test)
    thread.daemon = True
    thread.start()
    
    return """
    <h1>Direct Microphone Test Started</h1>
    <p>A direct microphone test has been started on the server. This test bypasses the regular streaming pipeline
    and directly connects to the microphone like the user's working example.</p>
    <p>Check the server logs for transcription results. This test will run for 30 seconds.</p>
    <p><a href="/speech_status">Check Speech Recognition Status</a></p>
    """

def run_direct_mic_test():
    """Run a direct microphone test using the approach from the user's working example"""
    try:
        print("Starting direct microphone test - this mimics the user's working example")
        
        # Create speech client
        client = speech.SpeechClient()
        
        # Configure recognition
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="phone_call",
            audio_channel_count=CHANNELS,
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )
        
        # Create microphone stream class (similar to user's example)
        class DirectMicrophoneStream:
            """Opens a recording stream as a generator yielding the audio chunks."""
            def __init__(self, rate, chunk):
                self._rate = rate
                self._chunk = chunk
                self._buff = queue.Queue()
                self.closed = True

            def __enter__(self):
                self._audio_interface = pyaudio.PyAudio()
                self._audio_stream = self._audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=self._rate,
                    input=True,
                    frames_per_buffer=self._chunk,
                    stream_callback=self._fill_buffer,
                )
                self.closed = False
                return self

            def __exit__(self, type, value, traceback):
                self._audio_stream.stop_stream()
                self._audio_stream.close()
                self.closed = True
                self._buff.put(None)
                self._audio_interface.terminate()

            def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
                """Continuously collect data from the audio stream into the buffer."""
                self._buff.put(in_data)
                return None, pyaudio.paContinue

            def generator(self):
                while not self.closed:
                    chunk = self._buff.get()
                    if chunk is None:
                        return
                    data = [chunk]

                    # Now consume whatever other data's still buffered.
                    while True:
                        try:
                            chunk = self._buff.get(block=False)
                            if chunk is None:
                                return
                            data.append(chunk)
                        except queue.Empty:
                            break

                    yield b''.join(data)
        
        # Simple response handler
        def handle_responses(responses):
            for response in responses:
                if not response.results:
                    continue
                
                result = response.results[0]
                if not result.alternatives:
                    continue
                
                transcript = result.alternatives[0].transcript
                
                if result.is_final:
                    print(f"DIRECT TEST - Final transcript: '{transcript}'")
                else:
                    print(f"DIRECT TEST - Interim transcript: '{transcript}'")
        
        # Start direct mic stream test
        print("DIRECT TEST - Opening microphone stream")
        with DirectMicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            
            # Create streaming recognize requests
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            
            # Start streaming recognition
            print("DIRECT TEST - Starting streaming recognition")
            responses = client.streaming_recognize(streaming_config, requests)
            
            # Set up timeout - run for 30 seconds maximum
            def stop_after(stream_obj, seconds):
                print(f"DIRECT TEST - Will run for {seconds} seconds")
                time.sleep(seconds)
                print("DIRECT TEST - Stopping stream now")
                stream_obj.closed = True
            
            # Start timeout thread
            timeout_thread = threading.Thread(target=stop_after, args=(stream, 30))
            timeout_thread.daemon = True
            timeout_thread.start()
            
            # Process responses
            handle_responses(responses)
            
        print("DIRECT TEST - Test completed")
        
    except Exception as e:
        print(f"DIRECT TEST - Error during test: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html',)

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

@socketio.on('connect')
def on_connect():
    print(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def on_disconnect():
    print(f"Client disconnected: {request.sid}")

@app.route('/speech_status')
def speech_status():
    """Endpoint to check the status of speech recognition"""
    
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Speech recognition is disabled."
    
    if 'streaming_recognizer' not in globals() or streaming_recognizer is None:
        return "Speech recognition has not been initialized."
    
    # Calculate health metrics
    current_time = time.time()
    time_since_last_response = current_time - streaming_recognizer.last_response_time
    time_since_last_audio = current_time - streaming_recognizer.last_audio_time
    
    # Format status info
    status_info = {
        "active": streaming_active,
        "last_transcript": streaming_recognizer.last_transcript,
        "time_since_last_response": f"{time_since_last_response:.1f} seconds",
        "time_since_last_audio": f"{time_since_last_audio:.1f} seconds",
        "silence_duration": f"{streaming_recognizer.silence_duration:.1f} seconds",
        "reconnect_count": streaming_recognizer.reconnect_count,
        "stream_age": f"{current_time - streaming_recognizer.stream_start_time:.1f} seconds",
        "stream_max_age": f"{streaming_recognizer.max_stream_duration} seconds",
        "vad_enabled": streaming_recognizer.vad_enabled,
        "silence_threshold": streaming_recognizer.silence_threshold,
    }
    
    # Determine overall health
    if not streaming_active:
        health = "OFFLINE"
    elif time_since_last_response > 120:
        health = "CRITICAL - No responses received in over 2 minutes"
    elif time_since_last_response > 60:
        health = "WARNING - No responses received in over 1 minute"
    elif streaming_recognizer.reconnect_count > 5:
        health = "WARNING - Multiple reconnection attempts"
    else:
        health = "OK"
    
    status_info["health"] = health
    
    # Format a nice HTML response
    html = f"""
    <html>
    <head>
        <title>Speech Recognition Status</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .ok {{ background-color: #dff0d8; color: #3c763d; }}
            .warning {{ background-color: #fcf8e3; color: #8a6d3b; }}
            .critical {{ background-color: #f2dede; color: #a94442; }}
            .offline {{ background-color: #f5f5f5; color: #777; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .controls {{ margin-top: 20px; }}
            button {{ padding: 8px 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }}
            button:hover {{ background-color: #45a049; }}
        </style>
    </head>
    <body>
        <h1>Speech Recognition Status</h1>
        
        <div class="status {health.split(' ')[0].lower()}">
            <strong>Status:</strong> {health}
        </div>
        
        <h2>Current Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Active</td><td>{status_info['active']}</td></tr>
            <tr><td>VAD Enabled</td><td>{status_info['vad_enabled']}</td></tr>
            <tr><td>Silence Threshold</td><td>{status_info['silence_threshold']}</td></tr>
            <tr><td>Stream Age</td><td>{status_info['stream_age']}</td></tr>
            <tr><td>Stream Max Age</td><td>{status_info['stream_max_age']}</td></tr>
            <tr><td>Reconnect Count</td><td>{status_info['reconnect_count']}</td></tr>
        </table>
        
        <h2>Runtime Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Time Since Last Response</td><td>{status_info['time_since_last_response']}</td></tr>
            <tr><td>Time Since Last Audio</td><td>{status_info['time_since_last_audio']}</td></tr>
            <tr><td>Current Silence Duration</td><td>{status_info['silence_duration']}</td></tr>
        </table>
        
        <h2>Last Transcript</h2>
        <div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
            {status_info['last_transcript'] if status_info['last_transcript'] else "<i>No transcripts yet</i>"}
        </div>
        
        <div class="controls">
            <a href="/restart_speech"><button>Restart Speech Recognition</button></a>
            <a href="/"><button style="background-color: #2196F3;">Back to Home</button></a>
        </div>
    </body>
    </html>
    """
    
    return html

@app.route('/restart_speech')
def restart_speech():
    """Endpoint to manually restart speech recognition"""
    
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Cannot restart speech recognition."
    
    try:
        global streaming_recognizer, streaming_active
        
        # Stop current recognizer if it exists
        if 'streaming_recognizer' in globals() and streaming_recognizer is not None:
            print("Stopping current streaming recognizer")
            streaming_recognizer.stop()
            time.sleep(1)  # Give it time to stop
        
        # Start a new recognizer
        print("Starting new streaming recognizer")
        streaming_active = True
        start_streaming_recognition()
        
        return """
        <html>
        <head>
            <meta http-equiv="refresh" content="2;url=/speech_status" />
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; text-align: center; }
            </style>
        </head>
        <body>
            <h1>Speech Recognition Restarted</h1>
            <p>Redirecting to status page...</p>
            <p><a href="/speech_status">Click here if not redirected</a></p>
        </body>
        </html>
        """
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error restarting speech recognition: {str(e)}<br><pre>{error_trace}</pre>"

@app.route('/test_fixed_audio')
def test_fixed_audio():
    """
    Test speech recognition with a fixed audio sample
    This helps verify if the Google Speech API is working correctly
    """
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Cannot test speech recognition."
    
    try:
        # Run the test in a separate thread
        thread = threading.Thread(target=run_fixed_audio_test)
        thread.daemon = True
        thread.start()
        
        return """
        <h1>Fixed Audio Test Started</h1>
        <p>A test using a generated speech sample has been started.</p>
        <p>This test will send a simple sine wave audio sample to the Google Speech API 
        to verify if the API connection and authentication are working properly.</p>
        <p>Check the server logs for results.</p>
        <p><a href="/speech_status">Check Speech Recognition Status</a></p>
        """
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error starting fixed audio test: {str(e)}<br><pre>{error_trace}</pre>"

def run_fixed_audio_test():
    """
    Run a test with fixed audio data to verify API functionality
    This uses a simple generated tone that should be recognized as speech
    """
    try:
        print("Starting fixed audio sample test")
        
        # Create speech client
        client = speech.SpeechClient()
        
        # Configure recognition
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="phone_call",
            audio_channel_count=1,
        )
        
        # Generate a simple audio sample (sine wave)
        duration_seconds = 3
        sample_count = int(RATE * duration_seconds)
        
        # Create a sine wave with varying frequency to simulate speech
        t = np.linspace(0, duration_seconds, sample_count, False)
        
        # Create syllable-like pattern (basic speech simulation)
        # Start with 200Hz base frequency
        frequencies = 200 + 100 * np.sin(2 * np.pi * 2 * t)  # Vary between ~100-300Hz
        
        # Generate sine wave with varying frequency
        audio_data = 0.5 * np.sin(2 * np.pi * frequencies * t)
        
        # Apply amplitude modulation to simulate syllables
        syllable_rate = 4  # 4 syllables per second
        amplitude_mod = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)
        audio_data = audio_data * amplitude_mod
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        
        print(f"FIXED TEST - Created {duration_seconds}s test audio sample")
        
        # Create audio content
        audio = speech.RecognitionAudio(content=audio_bytes)
        
        # Send synchronous request
        print("FIXED TEST - Sending synchronous recognition request")
        response = client.recognize(config=config, audio=audio)
        
        # Process response
        if response.results:
            for result in response.results:
                for alternative in result.alternatives:
                    print(f"FIXED TEST - Transcript: '{alternative.transcript}'")
                    print(f"FIXED TEST - Confidence: {alternative.confidence}")
            print("FIXED TEST - Successfully received transcription from API")
        else:
            print("FIXED TEST - No transcription results returned from API")
            
        # Try streaming recognition as well
        print("FIXED TEST - Now testing streaming recognition")
        
        # Configure streaming recognition
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="en-US",
                enable_automatic_punctuation=True,
                model="phone_call",
                audio_channel_count=CHANNELS,
            ),
            interim_results=True,
        )
        
        # Split audio into chunks to simulate streaming
        chunk_duration_ms = 100  # 100ms chunks
        samples_per_chunk = int(RATE * chunk_duration_ms / 1000)
        chunks = [audio_int16[i:i+samples_per_chunk] for i in range(0, len(audio_int16), samples_per_chunk)]
        
        # Create generator for chunks
        def audio_chunk_generator():
            for chunk in chunks:
                yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
        
        # Start streaming recognition
        print(f"FIXED TEST - Sending {len(chunks)} streaming chunks")
        streaming_responses = client.streaming_recognize(streaming_config, audio_chunk_generator())
        
        # Process streaming responses
        any_results = False
        for response in streaming_responses:
            if response.results:
                any_results = True
                for result in response.results:
                    for alternative in result.alternatives:
                        result_type = "Final" if result.is_final else "Interim"
                        print(f"FIXED TEST - {result_type} transcript: '{alternative.transcript}'")
        
        if not any_results:
            print("FIXED TEST - No streaming transcription results returned from API")
        
        print("FIXED TEST - Test completed")
        
    except Exception as e:
        print(f"FIXED TEST - Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the SoundWatch server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0 to allow connections from anywhere)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    args = parser.parse_args()
    
    # Start streaming speech recognition if Google Cloud is enabled
    if GOOGLE_CLOUD_ENABLED:
        print("Starting streaming speech recognition...")
        streaming_recognizer = start_streaming_recognition()
    
    print(f"Starting server on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)
