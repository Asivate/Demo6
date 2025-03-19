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

# Buffer for speech recognition (holds audio data for processing)
speech_buffer = []
speech_buffer_lock = threading.Lock()
speech_processing_queue = queue.Queue()

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
            model="default",
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
        self.thread = None
    
    def add_audio(self, audio_data):
        """Add audio data to the processing queue"""
        if self.is_active:
            # Convert float32 [-1.0, 1.0] to int16
            int16_data = (audio_data * 32768).astype(np.int16)
            self.audio_queue.put(int16_data.tobytes())
    
    def stop(self):
        """Stop the streaming recognition"""
        self.is_active = False
        self.audio_queue.put(None)  # Signal to stop
    
    def audio_generator(self):
        """Generate audio chunks from the queue"""
        while self.is_active:
            chunk = self.audio_queue.get()
            if chunk is None:
                return
            
            data = [chunk]
            
            # Get any additional chunks that might be in the queue
            while True:
                try:
                    chunk = self.audio_queue.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            
            yield b''.join(data)
    
    def process_responses(self, responses):
        """Process streaming responses from the API"""
        try:
            for response in responses:
                if not self.is_active:
                    break
                
                if not response.results:
                    continue
                
                result = response.results[0]  # Get the first result
                
                if not result.alternatives:
                    continue
                
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence if hasattr(result.alternatives[0], 'confidence') else 0.0
                
                if result.is_final:
                    self.last_response_time = time.time()
                    self.last_transcript = transcript
                    print(f"Final transcript: '{transcript}' (confidence: {confidence})")
                    
                    # Skip empty transcripts
                    if not transcript.strip():
                        continue
                    
                    # Process for sentiment analysis
                    try:
                        # Create document for sentiment analysis
                        document = language_v1.Document(
                            content=transcript,
                            type_=language_v1.Document.Type.PLAIN_TEXT
                        )
                        
                        # Analyze sentiment
                        sentiment_response = language_client.analyze_sentiment(document=document)
                        sentiment = sentiment_response.document_sentiment
                        
                        # Map to emoji and emotion
                        emoji = get_sentiment_emoji(sentiment.score)
                        emotion = get_sentiment_emotion(sentiment.score)
                        
                        print(f"Sentiment analysis: score={sentiment.score}, emotion={emotion}, emoji={emoji}")
                        
                        # Create timestamp
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        
                        # Create transcript entry
                        transcript_entry = {
                            "transcription": transcript,
                            "confidence": float(confidence),
                            "emotion": emotion,
                            "emoji": emoji,
                            "sentiment_score": float(sentiment.score),
                            "sentiment_magnitude": float(sentiment.magnitude),
                            "timestamp": timestamp
                        }
                        
                        # Send to all connected clients
                        socketio.emit('audio_label', {
                            'label': 'Speech',
                            'accuracy': '1.0',
                            'emoji': emoji,
                            'emotion': emotion,
                            'transcription': transcript,
                            'confidence': confidence,
                            'sentiment_score': sentiment.score,
                            'sentiment_magnitude': sentiment.magnitude,
                            'timestamp': timestamp
                        })
                        
                        # Also emit transcript update event
                        socketio.emit('transcript_update', transcript_entry)
                        
                    except Exception as e:
                        print(f"Error processing sentiment for transcript: {e}")
                else:
                    # Interim result
                    print(f"Interim transcript: '{transcript}'")
                    
        except Exception as e:
            print(f"Error processing streaming responses: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the streaming recognition in a continuous loop"""
        retry_count = 0
        max_retries = 10
        
        while self.is_active and retry_count < max_retries:
            try:
                print("Starting streaming recognition session")
                
                # Create the generator for audio chunks
                audio_generator = self.audio_generator()
                
                # Create streaming recognize requests
                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator
                )
                
                # Start streaming recognition
                responses = self.client.streaming_recognize(self.streaming_config, requests)
                
                # Process responses
                self.process_responses(responses)
                
                # If we get here, the stream ended naturally
                print("Streaming recognition session ended normally")
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                print(f"Streaming recognition error (retry {retry_count}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                
                # Implement progressive backoff
                wait_time = min(5 * retry_count, 30)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
        print("Streaming recognition stopped")
    
    def start(self):
        """Start the streaming recognition in a background thread"""
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        return self.thread

def start_streaming_recognition():
    """Start the streaming recognition thread using the simplified implementation"""
    global streaming_recognizer, streaming_active
    
    if not GOOGLE_CLOUD_ENABLED:
        print("Google Cloud services not enabled. Cannot start streaming recognition.")
        return
    
    streaming_active = True
    
    # Use the simpler streaming recognizer implementation
    streaming_recognizer = SimpleStreamingRecognizer()
    
    # Print information about the environment
    print("Starting streaming speech recognition with the following settings:")
    print(f"- Sample rate: {RATE} Hz")
    print(f"- Channels: {CHANNELS}")
    print(f"- Google credentials: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    
    # Start in a background thread
    streaming_recognizer.start()
    
    print("Streaming recognition thread started")
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
                        socketio.emit('audio_label',
                                   {'label': str(homesounds.to_human_labels[active_context[m]]),
                                    'accuracy': str(context_prediction[m]),
                                    'db': str(db)},
                                   room=request.sid)
                        print("Prediction: %s (%0.2f)" % (
                            homesounds.to_human_labels[active_context[m]], context_prediction[m]))
                    else:
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
    data = str(json_data['data'])
    data = data[1:-1]
    global graph, model, sess, streaming_recognizer
    np_wav = np.fromstring(data, dtype=np.int16, sep=',') / \
        32768.0  # Convert to [-1.0, +1.0]
    
    # Compute RMS and convert to dB
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)
    
    # Log basic audio metrics
    print(f'Audio received - length: {len(np_wav)}, RMS: {rms:.4f}, dB: {db:.1f}')
    
    # Process for speech recognition
    if GOOGLE_CLOUD_ENABLED and streaming_active and 'streaming_recognizer' in globals() and streaming_recognizer is not None:
        # Send audio to the streaming recognizer
        # No need for extensive preprocessing now, our recognizer handles conversion
        streaming_recognizer.add_audio(np_wav)
    
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
                    socketio.emit('audio_label',
                                {
                                    'label': str(homesounds.to_human_labels[active_context[m]]),
                                    'accuracy': str(context_prediction[m]),
                                    'db': str(db)
                                },
                                room=request.sid)
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
            # Add more configuration options for better results
            enable_automatic_punctuation=True,
            model="default",  # Use "video" for better handling of media content or "phone_call" for phone conversations
            use_enhanced=True,  # Use enhanced model
            # Add these parameters to improve speech recognition in noisy environments
            audio_channel_count=CHANNELS,
            enable_separate_recognition_per_channel=False,
            # More aggressive noise settings
            # Try with different speech contexts if needed
            speech_contexts=[speech.SpeechContext(
                phrases=["sound", "watch", "notification", "alarm", "alert", "doorbell", "knock", "phone"]
            )]
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
            "transcription": transcript,
            "confidence": float(confidence),
            "emotion": emotion,
            "emoji": emoji,
            "sentiment_score": float(sentiment.score),
            "sentiment_magnitude": float(sentiment.magnitude),
            "timestamp": timestamp
        }
        
        # Send to connected clients
        socketio.emit('audio_label', {
            'label': 'Speech',
            'accuracy': '1.0',
            'db': str(db),
            'emoji': emoji,
            'emotion': emotion,
            'transcription': transcript,
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
        
        # Try the speech recognition with this test audio
        audio_bytes = audio_int16.tobytes()
        audio_obj = speech.RecognitionAudio(content=audio_bytes)
        
        # Create config with more debugging options
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="default",
            # Add more options for debugging
            enable_word_time_offsets=True,
            enable_word_confidence=True,
            audio_channel_count=CHANNELS,
            profanity_filter=False,
            # Try different models to see if any work better
            use_enhanced=True
        )
        
        # Call the API
        print("Testing Speech API with generated audio...")
        response = speech_client.recognize(config=config, audio=audio_obj)
        
        # Generate a plot of the audio waveform for visualization
        plt.figure(figsize=(10, 4))
        plt.plot(t, audio)
        plt.title("Test Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Save plot to a base64 string to embed in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Create result HTML
        html = "<h1>Speech Recognition Test</h1>"
        html += "<p>Testing Google Cloud Speech-to-Text API with generated test audio</p>"
        
        # Add the audio waveform
        html += "<h2>Test Audio Waveform:</h2>"
        html += f'<img src="data:image/png;base64,{plot_data}" alt="Test Audio Waveform" />'
        
        # Add the API response
        html += "<h2>API Response:</h2>"
        html += f"<pre>{str(response)}</pre>"
        
        # Format any results
        if hasattr(response, 'results') and response.results:
            html += "<h2>Transcription Results:</h2><ul>"
            for i, result in enumerate(response.results):
                html += f"<li>Result {i+1}:"
                if result.alternatives:
                    for j, alt in enumerate(result.alternatives):
                        html += f"<br>Alternative {j+1}: '{alt.transcript}' (confidence: {alt.confidence})"
                else:
                    html += "<br>No alternatives found in this result"
                html += "</li>"
            html += "</ul>"
        else:
            html += "<p>No transcription results returned by API</p>"
        
        # Add instructions for next steps
        html += "<h2>Next Steps:</h2>"
        html += "<p>If you see no results above, that indicates an issue with the Speech-to-Text API recognition.</p>"
        html += "<p>Possible reasons:</p>"
        html += "<ul>"
        html += "<li>The generated test audio doesn't contain actual speech (expected for synthetic audio)</li>"
        html += "<li>API credentials don't have access to the Speech-to-Text API</li>"
        html += "<li>There might be configuration issues with the API</li>"
        html += "</ul>"
        html += "<p>You can also try the <a href='/test_sentiment'>sentiment analysis test</a> to verify that part of the pipeline.</p>"
        
        return html
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error testing speech recognition: {str(e)}<br><pre>{error_trace}</pre>"

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
            model="default",
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

@app.route('/speech_status')
def speech_status():
    """
    Returns information about the current status of the speech recognition system
    Useful for debugging
    """
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Speech recognition unavailable."
    
    if 'streaming_recognizer' not in globals() or streaming_recognizer is None:
        return "No active streaming recognizer found."
    
    # Calculate timings
    now = time.time()
    last_audio_age = now - streaming_recognizer.last_audio_time
    last_response_age = now - streaming_recognizer.last_response_time
    
    status_html = "<h1>Speech Recognition Status</h1>"
    status_html += "<ul>"
    status_html += f"<li>Streaming active: {streaming_active}</li>"
    status_html += f"<li>Last audio received: {last_audio_age:.1f} seconds ago</li>"
    status_html += f"<li>Last transcript received: {last_response_age:.1f} seconds ago</li>"
    status_html += f"<li>Last transcript: '{streaming_recognizer.last_transcript}'</li>"
    status_html += "</ul>"
    
    status_html += "<h2>Troubleshooting</h2>"
    status_html += "<ul>"
    status_html += "<li>If 'Last audio received' is very recent but no transcriptions are appearing:</li>"
    status_html += "<ul><li>Check that your audio contains speech and is loud enough</li>"
    status_html += "<li>Check Google Cloud Speech API quota usage</li>"
    status_html += "<li>Try restarting the server</li></ul>"
    status_html += "</ul>"
    
    status_html += "<p><a href='/test_speech'>Run Speech Test</a> | <a href='/test_sentiment'>Run Sentiment Test</a></p>"
    
    return status_html

@app.route('/restart_speech')
def restart_speech_endpoint():
    """
    Endpoint to manually restart the speech recognition system
    Useful when issues occur
    """
    if not GOOGLE_CLOUD_ENABLED:
        return "Google Cloud services not enabled. Speech recognition unavailable."
    
    try:
        # Stop current recognizer if it exists
        if 'streaming_recognizer' in globals() and streaming_recognizer is not None:
            stop_streaming_recognition()
            time.sleep(2)  # Allow time for cleanup
        
        # Start a new recognizer
        new_recognizer = start_streaming_recognition()
        
        if new_recognizer:
            return "Speech recognition system restarted successfully. <a href='/speech_status'>Check status</a>"
        else:
            return "Failed to restart speech recognition system."
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
            model="default",
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
            config=config,
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
