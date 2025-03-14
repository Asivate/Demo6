import numpy as np
# Apply NumPy patch for TensorFlow compatibility
if not hasattr(np, 'object'):
    np.object = object
    print("NumPy patch applied: Added np.object compatibility for TensorFlow")

import tensorflow as tf
from tensorflow import keras
from vggish_input import waveform_to_examples
import homesounds
import pyaudio
from pathlib import Path
import time
import argparse
import wget
from helpers import dbFS
import os

# Configure TensorFlow to use compatibility mode with TF 1.x code
tf.compat.v1.disable_eager_execution()

# contexts
context = homesounds.everything
active_context = homesounds.everything      # use this to change context -- see homesounds.py

# thresholds
PREDICTION_THRES = 0.5 # confidence
DBLEVEL_THRES = -30 # dB

# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0

###########################
# Check Microphone
###########################
print("=====")
print("1 / 2: Checking Microphones... ")
print("=====")

import microphones
desc, mics, indices = microphones.list_microphones()
if (len(mics) == 0):
    print("Error: No microphone found.")
    exit()

#############
# Read Command Line Args
#############
MICROPHONE_INDEX = indices[0]
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mic", help="Select which microphone / input device to use")
args = parser.parse_args()
try:
    if args.mic:
        MICROPHONE_INDEX = int(args.mic)
        print("User selected mic: %d" % MICROPHONE_INDEX)
    else:
        mic_in = input("Select microphone [%d]: " % MICROPHONE_INDEX).strip()
        if (mic_in!=''):
            MICROPHONE_INDEX = int(mic_in)
except:
    print("Invalid microphone")
    exit()

# Find description that matches the mic index
mic_desc = ""
for k in range(len(indices)):
    i = indices[k]
    if (i==MICROPHONE_INDEX):
        mic_desc = mics[k]
print("Using mic: %s" % mic_desc)

###########################
# Download model, if it doesn't exist
###########################
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
print("=====")
print("2 / 2: Checking model... ")
print("=====")
model_filename = "models/example_model.hdf5"

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(model_filename), exist_ok=True)

homesounds_model = Path(model_filename)
if (not homesounds_model.is_file()):
    print("Downloading example_model.hdf5 [867MB]: ")
    wget.download(MODEL_URL,MODEL_PATH)

##############################
# Load Deep Learning Model
##############################
print("Using deep learning model: %s" % (model_filename))
model = None

try:
    # Try to load the model with the new API
    model = keras.models.load_model(model_filename)
    print("Model loaded successfully with TensorFlow 2.x")
except Exception as e:
    print(f"Error loading model with standard method: {e}")
    # Fallback for older model formats
    try:
        model = tf.keras.models.load_model(model_filename, compile=False)
        print("Model loaded with compile=False option")
    except Exception as e2:
        print(f"Error with fallback method: {e2}")
        raise Exception("Could not load model with any method")

##############################
# Setup Audio Callback
##############################
def audio_samples(in_data, frame_count, time_info, status_flags):
    np_wav = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0 # Convert to [-1.0, +1.0]
    # Compute RMS and convert to dB
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)

    # Make predictions
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    
    if x.shape[0] != 0:
        x = x.reshape(len(x), 96, 64, 1)
        pred = model.predict(x)
        predictions.append(pred)

    for prediction in predictions:
        context_prediction = np.take(prediction[0], [homesounds.labels[x] for x in active_context])
        m = np.argmax(context_prediction)
        if (context_prediction[m] > PREDICTION_THRES and db > DBLEVEL_THRES):
            print("Prediction: %s (%0.2f)" % (homesounds.to_human_labels[active_context[m]], context_prediction[m]))
        
    return (in_data, pyaudio.paContinue)

##############################
# Launch Application
##############################
while(1):
    ##############################
    # Setup Audio
    ##############################
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_samples, input_device_index=MICROPHONE_INDEX)

    ##############################
    # Start Non-Blocking Stream
    ##############################
    print("# Live Prediction Using Microphone: %s" % (mic_desc))
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)
