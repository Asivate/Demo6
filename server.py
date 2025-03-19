from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context, jsonify, send_file
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
# Import the speech_processor module for speech recognition and sentiment analysis
from speech_processor import SpeechProcessor, categorize_sentiment
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import io
import logging
import threading

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

# Initialize the speech processor for speech recognition and sentiment analysis
speech_processor = SpeechProcessor()
speech_processor_lock = threading.Lock()

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


@socketio.on('audio_data')
def handle_source(json_data):
    data = str(json_data['data'])
    data = data[1:-1]
    global graph, model, sess
    np_wav = np.fromstring(data, dtype=np.int16, sep=',') / \
        32768.0  # Convert to [-1.0, +1.0]
    # Compute RMS and convert to dB
    print('Successfully convert to NP rep', np_wav)
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)
    print('Db...', db)
    
    # Skip processing if the audio is silent (very low RMS)
    if rms < SILENCE_RMS_THRESHOLD:
        print(f"Detected silent audio frame (RMS: {rms}, min={np_wav.min():.4f}, max={np_wav.max():.4f}). Skipping processing.")
        socketio.emit('audio_label',
                    {
                        'label': 'Silent',
                        'accuracy': '1.0',
                        'db': str(db)
                    },
                    room=request.sid)
        return
    
    # Additional check for near-zero audio frames that might not be caught by RMS threshold
    if np_wav.max() == 0.0 and np_wav.min() == 0.0:
        print(f"Detected empty audio frame with all zeros. Skipping processing.")
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
    
    # Make predictions
    print('Making prediction...')
    print(f'Audio data: shape={np_wav.shape}, min={np_wav.min():.4f}, max={np_wav.max():.4f}, rms={np.sqrt(np.mean(np_wav**2)):.4f}')
    x = waveform_to_examples(np_wav, RATE)
    print(f'Generated features: shape={x.shape}')
    
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
    x = x.reshape(x.shape[0], vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1)
    print('Successfully reshaped to', x.shape)
    
    # Store original audio for speech recognition
    original_audio = np.fromstring(data, dtype=np.int16, sep=',')
    
    # Process the audio for speech recognition in addition to sound classification
    process_audio_for_speech_recognition(original_audio, request.sid)
    
    predictions = []
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


def process_audio_for_speech_recognition(audio_data, sid):
    """
    Process audio data for speech recognition and send results to the client.
    
    Args:
        audio_data: Audio data as numpy array
        sid: Socket ID of the client
    """
    try:
        # Add the audio data to the speech processor
        speech_processor.add_audio_data(audio_data)
        
        # Get the latest results from the speech processor
        results = speech_processor.get_latest_results()
        
        # Only emit results if we have a transcript
        if results["transcript"]:
            # Get a human-readable sentiment category
            sentiment_score = results["sentiment"]["score"]
            sentiment_magnitude = results["sentiment"]["magnitude"]
            sentiment_category = categorize_sentiment(sentiment_score, sentiment_magnitude)
            
            # Send the transcript and sentiment analysis to the client
            socketio.emit('speech_transcript',
                        {
                            'transcript': results["transcript"],
                            'language': results["language"],
                            'sentiment_score': str(sentiment_score),
                            'sentiment_magnitude': str(sentiment_magnitude),
                            'sentiment_category': sentiment_category
                        },
                        room=sid)
            
            print(f"Speech transcript: {results['transcript']}")
            print(f"Sentiment: {sentiment_category} (score: {sentiment_score}, magnitude: {sentiment_magnitude})")
    except Exception as e:
        print(f"Error processing audio for speech recognition: {e}")


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count})


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.on('send_message')
def handle_source(json_data):
    print('got message from client' + str(json_data))
    socketio.emit('message', {'data': json_data}, room=request.sid)

@socketio.on('disconnect_request', namespace='/test')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('my_response',
         {'data': 'Disconnected!'},
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
def test_connect():
    # Start the speech recognition streaming when a client connects
    speech_processor.start_streaming()
    print('Client connected', request.sid)


@socketio.on('disconnect')
def test_disconnect():
    # No need to stop streaming here as we want it to continue for all clients
    print('Client disconnected', request.sid)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/start_speech_recognition', methods=['POST'])
def start_speech_recognition():
    """Start the speech recognition service."""
    with speech_processor_lock:
        speech_processor.start_streaming()
    return jsonify({"status": "success", "message": "Speech recognition started"})

@app.route('/stop_speech_recognition', methods=['POST'])
def stop_speech_recognition():
    """Stop the speech recognition service."""
    with speech_processor_lock:
        speech_processor.stop_streaming()
    return jsonify({"status": "success", "message": "Speech recognition stopped"})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process audio data for speech recognition and sentiment analysis.
    
    The audio data can be sent as raw bytes or as a base64 encoded string.
    """
    try:
        # Check if the request contains 'audio_data' in JSON
        if request.is_json:
            data = request.get_json()
            
            if 'audio_data_base64' in data:
                # Decode base64 audio data
                audio_bytes = base64.b64decode(data['audio_data_base64'])
            else:
                return jsonify({"status": "error", "message": "No audio data found in request"}), 400
        
        # Check if the request contains file upload
        elif 'audio_file' in request.files:
            file = request.files['audio_file']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No file selected"}), 400
            
            # Read file data
            audio_bytes = file.read()
        
        # Check if the request contains raw binary data
        elif request.data:
            audio_bytes = request.data
        
        else:
            return jsonify({"status": "error", "message": "No audio data found in request"}), 400
        
        # Process the audio data
        with speech_processor_lock:
            speech_processor.add_audio_data(audio_bytes)
        
        return jsonify({"status": "success", "message": "Audio data received and processing"})
    
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_speech_results', methods=['GET'])
def get_speech_results():
    """Get the latest speech recognition and sentiment analysis results."""
    with speech_processor_lock:
        results = speech_processor.get_latest_results()
    
    return jsonify({
        "status": "success",
        "results": results,
        "timestamp": time.time()
    })

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """
    Analyze sentiment for a provided text.
    
    Request JSON format:
    {
        "text": "Text to analyze for sentiment"
    }
    """
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400
        
        data = request.get_json()
        if 'text' not in data or not data['text']:
            return jsonify({"status": "error", "message": "Text is required"}), 400
        
        text = data['text']
        
        with speech_processor_lock:
            sentiment_result = speech_processor._analyze_sentiment(text)
        
        if sentiment_result:
            return jsonify({
                "status": "success",
                "text": text,
                "sentiment": sentiment_result,
                "timestamp": time.time()
            })
        else:
            return jsonify({"status": "error", "message": "Could not analyze sentiment"}), 500
    
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/clear_speech_buffer', methods=['POST'])
def clear_speech_buffer():
    """Clear the speech recognition buffer."""
    with speech_processor_lock:
        speech_processor.clear_buffer()
    return jsonify({"status": "success", "message": "Speech buffer cleared"})

if __name__ == '__main__':
    # Start with speech recognition enabled
    speech_processor.start_streaming()
    logger.info("Server started with speech recognition enabled")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)
