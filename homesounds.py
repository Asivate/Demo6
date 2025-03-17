"""
Homesounds module for SoundWatch server.
Contains labels and mappings for sound classification.
"""

import os
import json
import logging
import numpy as np
from scipy import signal
import tensorflow as tf
from tensorflow.keras import layers
import time  # Add time module for timestamps
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# HomeSounds Label Definition
labels = {
    'dog-bark':0,
    'drill':1,
    'hazard-alarm':2,
    'phone-ring':3,
    'speech':4,
    'vacuum':5,
    'baby-cry':6,
    'chopping':7,
    'cough':8,
    'door':9,
    'water-running':10,
    'knock':11,
    'microwave':12,
    'shaver':13,
    'toothbrush':14,
    'blender':15,
    'dishwasher':16,
    'doorbell':17,
    'flush':18,
    'hair-dryer':19,
    'laugh':20,
    'snore':21,
    'typing':22,
    'hammer':23,
    'car-horn':24,
    'engine':25,
    'saw':26,
    'cat-meow':27,
    'alarm-clock':28,
    'cooking':29,
    # New sound labels
    'finger-snap':30,
    'hand-clap':31,
    'hand-sounds':32,
    'applause':33,
    'silence':34,
    'background':35,
    'music':36,
    'sound-effect':37,
    'electronic-sound':38,
    'notification':39,
    'male-conversation':40,
    'female-conversation':41,
    'conversation':42,
}

bathroom = ['water-running','flush','toothbrush','shaver','hair-dryer']
kitchen = ['water-running','chopping','cooking','microwave','blender','hazard-alarm','dishwasher','speech']
bedroom = ['alarm-clock','snore','cough','baby-cry','speech', 'music']
office = ['knock','typing','phone-ring','door','cough','speech', 'finger-snap', 'notification']
entrance = ['knock','door','doorbell','speech','laugh', 'finger-snap', 'hand-clap']
living = ['knock','door','doorbell','speech','laugh', 'music', 'applause', 'finger-snap']
workshop = ['hammer','saw','drill','vacuum','hazard-alarm','speech']
outdoor = ['dog-bark','cat-meow','engine','car-horn','speech','hazard-alarm', 'music']

everything = [
    'dog-bark', 'drill', 'hazard-alarm', 'phone-ring', 'speech', 
    'vacuum', 'baby-cry', 'chopping', 'cough', 'door', 
    'water-running', 'knock', 'microwave', 'shaver', 'toothbrush', 
    'blender', 'dishwasher', 'doorbell', 'flush', 'hair-dryer', 
    'laugh', 'snore', 'typing', 'hammer', 'car-horn', 
    'engine', 'saw', 'cat-meow', 'alarm-clock', 'cooking',
    # Add new sounds to everything context
    'finger-snap', 'hand-clap', 'hand-sounds', 'applause', 'silence',
    'background', 'music', 'sound-effect', 'electronic-sound', 'notification',
    'male-conversation', 'female-conversation', 'conversation'
]

context_mapping = {
    'kitchen': kitchen, 
    'bathroom': bathroom, 
    'bedroom': bedroom, 
    'office': office, 
    'entrance': entrance, 
    'workshop':workshop, 
    'outdoor':outdoor, 
    'everything': everything
}

# Sound class labels (index to label mapping)
to_human_labels = {
    0: "Alarm",
    1: "Baby crying",
    2: "Car horn",
    3: "Cat meowing",
    4: "Dishes",
    5: "Dog barking",
    6: "Door knock",
    7: "Doorbell",
    8: "Footsteps",
    9: "Glass breaking",
    10: "Keyboard typing",
    11: "Microwave",
    12: "Phone ringing",
    13: "Shower",
    14: "Sink water running",
    15: "Snoring",
    16: "Speech",
    17: "Toilet flush",
    18: "Vacuum cleaner",
    19: "Water boiling"
}

# Original HomeSounds mapping for comparison
original_to_human_labels = {
    'dog-bark': "Dog Barking",
    'drill': "Drill In-Use",
    'hazard-alarm': "Fire/Smoke Alarm",
    'phone-ring': "Phone Ringing",
    'speech': "Speech",
    'vacuum': "Vacuum In-Use",
    'baby-cry': "Baby Crying",
    'chopping': "Chopping",
    'cough': "Coughing",
    'door': "Door In-Use",
    'water-running': "Water Running",
    'knock': "Knocking",  # This is critical - the knock label is at index 11 in original
    'microwave': "Microwave",
    'shaver': "Shaver In-Use",
    'toothbrush': "Toothbrushing",
    'blender': "Blender In-Use",
    'dishwasher': "Dishwasher",
    'doorbell': "Doorbel In-Use",
    'flush': "Toilet Flushing",
    'hair-dryer': "Hair Dryer In-Use",
    'laugh': "Laughing",
    'snore': "Snoring",
    'typing': "Typing",
    'hammer': "Hammering",
    'car-horn': "Car Honking",
    'engine': "Vehicle Running",
    'saw': "Saw In-Use",
    'cat-meow': "Cat Meowing",
    'alarm-clock': "Alarm Clock",
    'cooking': "Utensils and Cutlery",  # This is likely being confused with knocking
}

# CORRECTED model index mapping for percussive sounds
KNOCK_IDX = 6  # Current index for Door knock in to_human_labels 
DISHES_IDX = 4  # Current index for Dishes in to_human_labels
GLASS_BREAKING_IDX = 9  # Current index for Glass breaking in to_human_labels

# Define percussive sounds that need special handling
percussive_sounds = ['Door knock', 'Glass breaking', 'Dishes']

# Add missing threshold constants
PREDICTION_THRES = 0.3  # Default prediction threshold for sound detection
DBLEVEL_THRES = -30  # Threshold for dB level to consider sound significant

# Define sound-specific thresholds
sound_specific_thresholds = {
    'Door knock': 0.15,   # Lower threshold for knock detection (from 0.3 to 0.15)
    'Dishes': 0.4,       # Keep dishes threshold higher to avoid confusion with knocking
    'door-bell': 0.4,
    'dog-bark': 0.4,
    'baby-cry': 0.3,
    'phone-ring': 0.35,
    'vacuum-cleaner': 0.35
    # Default threshold of 0.5 will be used for any sound not specified here
}

# Define function to get class names before it's used
def get_class_names():
    """
    Returns the list of class names used by the model output, properly mapped
    to human-readable labels.
    
    The model outputs a vector, and we need to know which
    index corresponds to which sound class.
    """
    # Create an array of class names in the order the model expects
    class_names = []
    for i in range(20):  # The model actually outputs 20 classes, not all of labels
        if i in to_human_labels:
            class_names.append(to_human_labels[i])
        else:
            class_names.append(f"Unknown-{i}")
    
    return class_names

# Map model index to actual sound class for specialized detection
class_names = get_class_names()
model_index_to_sound_class = {
    idx: name for idx, name in enumerate(class_names) 
    if name in percussive_sounds
}

# Add temporal smoothing for sound detection
class SoundDetectionHistory:
    def __init__(self, window_size=3, decay_factor=0.7):
        """Initialize with a window size for temporal smoothing
        
        Args:
            window_size: Number of recent predictions to consider
            decay_factor: How much to weight recent predictions vs older ones
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.history = {}  # Format: {sound_class: [(timestamp, confidence), ...]}
        self.last_detection = {}  # Format: {sound_class: timestamp}
    
    def add_prediction(self, predictions, timestamp=None):
        """Add new prediction to history
        
        Args:
            predictions: Dict mapping sound class to confidence
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Update history for each predicted class
        for sound_class, confidence in predictions.items():
            if sound_class not in self.history:
                self.history[sound_class] = []
                
            # Add new prediction
            self.history[sound_class].append((timestamp, confidence))
            
            # Keep only the most recent predictions
            if len(self.history[sound_class]) > self.window_size:
                self.history[sound_class] = self.history[sound_class][-self.window_size:]
    
    def get_smoothed_confidence(self, sound_class):
        """Get temporally smoothed confidence for a sound class
        
        Args:
            sound_class: The sound class to get confidence for
            
        Returns:
            Weighted average confidence over the temporal window
        """
        if sound_class not in self.history or not self.history[sound_class]:
            return 0.0
            
        entries = self.history[sound_class]
        total_weight = 0
        weighted_sum = 0
        
        # Calculate exponentially weighted average
        # More recent predictions have higher weight
        for i, (_, confidence) in enumerate(entries):
            # Position from oldest (0) to newest (n)
            position = i
            weight = self.decay_factor ** (len(entries) - position - 1)
            weighted_sum += confidence * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        # Add minimum confidence decay
        min_confidence = 0.01  # Prevent complete decay
        return max(weighted_sum, min_confidence)

    def check_for_percussive_sound(self, sound_class, current_confidence, db_level, min_time_between=1.0):  # Changed from 1.5s to 1.0s
        """Special handling for percussive sounds like knocking
        
        Args:
            sound_class: The sound class to check
            current_confidence: Current confidence level
            db_level: Current audio level in dB
            min_time_between: Minimum time between detections in seconds
            
        Returns:
            Adjusted confidence level
        """
        if sound_class not in percussive_sounds:
            return current_confidence
            
        current_time = time.time()
        
        # Check if we've detected this sound recently
        last_time = self.last_detection.get(sound_class, 0)
        time_since_last = current_time - last_time
        
        # For percussive sounds, we enhance confidence if:
        # 1. The sound hasn't been detected recently (avoid duplicate detections)
        # 2. The dB level is sufficiently high (indicating a sharp sound)
        # 3. There's at least some base confidence
        min_confidence = sound_specific_thresholds.get(sound_class, PREDICTION_THRES) * 0.5  # Only need 50% of threshold for boosting
        
        # Adjust for knocking with special handling
        if sound_class == 'Door knock' and current_confidence > min_confidence:  # Lower activation threshold
            # Check for dB spike which is common with knocks
            if db_level > -20 and time_since_last > min_time_between:  # More sensitive dB threshold (was -15)
                # Boost confidence for knock detection - higher multiplier
                adjusted_confidence = min(current_confidence * 2.5, 0.95)  # Boost by 2.5x (was 1.5x)
                # Record this detection time
                self.last_detection[sound_class] = current_time
                logger.info(f"Enhanced Door knock detection: {current_confidence:.4f} → {adjusted_confidence:.4f}")
                return adjusted_confidence
            
        # Special handling for Dishes class, which often activates during knocking
        if sound_class == 'Dishes' and current_confidence > min_confidence * 0.8:
            # If it's a sharp sound with high dB, it might be a knock misclassified as dishes
            if db_level > -15 and time_since_last > min_time_between:
                # Check if we have 'knock' data in history to ensure we're not creating false positives
                if 'Door knock' in self.history and len(self.history['Door knock']) > 0:
                    knock_confidence = self.get_smoothed_confidence('Door knock')
                    # If there's any knock confidence at all, transfer some confidence from dishes to knock
                    if knock_confidence > 0.05:  # Lowered from 0.1
                        # Transfer 40% of dishes confidence to Door knock (was just noting this relationship)
                        knock_boost = current_confidence * 0.4
                        logger.info(f"Transferring confidence from Dishes to Door knock: {knock_boost:.4f}")
                        
                        # Update Door knock confidence
                        transfer_time = time.time()
                        if 'Door knock' not in self.history:
                            self.history['Door knock'] = []
                        self.history['Door knock'].append((transfer_time, knock_boost + knock_confidence))
                        
                        # Record this as a Door knock detection
                        self.last_detection['Door knock'] = current_time
                        
                        # Return lower confidence for dishes
                        return current_confidence * 0.6
        
        return current_confidence

# Create a global instance for use in the server
detection_history = SoundDetectionHistory(window_size=3, decay_factor=0.7)

# Add a new method to detect abrupt changes in audio levels that might indicate knocking
def detect_percussive_event(audio_data, sample_rate=16000, threshold=0.25):  # Lower threshold from 0.4 to 0.25
    """
    Detect percussive events like knocking in audio data
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        threshold: Detection threshold (0-1)
        
    Returns:
        True if percussive event detected, False otherwise
    """
    try:
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_norm = audio_data / np.max(np.abs(audio_data))
        else:
            return False
            
        # Calculate short-term energy
        frame_length = int(0.01 * sample_rate)  # 10ms frames for better detection (was 15ms)
        hop_length = int(0.005 * sample_rate)   # 5ms hop (was 7.5ms)
        
        # Ensure we have enough data
        if len(audio_norm) < frame_length:
            return False
            
        # Calculate energy in each frame
        energy = []
        for i in range(0, len(audio_norm) - frame_length, hop_length):
            frame = audio_norm[i:i+frame_length]
            energy.append(np.sum(frame**2) / frame_length)
            
        # No frames calculated
        if not energy:
            return False
            
        # Convert to numpy array
        energy = np.array(energy)
        
        # Calculate the derivative of energy to detect sudden changes
        energy_diff = np.diff(energy)
        
        # Check for characteristic "double peak" pattern common in knocks
        if len(energy_diff) > 10:
            # Find peaks using peak detection
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(energy_diff, height=threshold*0.4, distance=3)  # More sensitive peak detection
            
            # If we found more than one peak within a short time
            if len(peaks) >= 2:
                # Check if they're spaced appropriately for a knock
                peak_spacing = np.diff(peaks)
                # Look for peaks between 4-25 frames apart (wider range to catch more variations)
                if any((peak_spacing >= 4) & (peak_spacing <= 25)):
                    logger.info(f"Knock pattern detected: {len(peaks)} peaks, max diff={np.max(energy_diff):.4f}")
                    return True
        
        # If max energy difference exceeds threshold, it's likely a percussive sound
        if np.max(energy_diff) > threshold:
            logger.info(f"Percussive event detected: {np.max(energy_diff):.4f}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error in percussive event detection: {e}")
        return False

def process_predictions(preds, context, db_level, history_length=5):
    """Stateful prediction processing with temporal analysis"""
    global PREDICTION_HISTORY
    
    # Dynamic threshold adjustment
    thresholds = {
        k: max(v['min'], v['base'] * (db_level/70)) 
        for k, v in SOUND_THRESHOLDS.items()
    }
    
    # Context weighting
    weighted = {
        label: prob * CONTEXT_WEIGHTS[context]['weights'].get(label, 1)
        for label, prob in preds.items()
    }
    
    # Temporal smoothing
    current_time = time.time()
    PREDICTION_HISTORY.setdefault(context, []).append({
        'time': current_time,
        'predictions': weighted,
        'db': db_level
    })
    
    # Remove old entries
    PREDICTION_HISTORY[context] = [
        entry for entry in PREDICTION_HISTORY[context]
        if current_time - entry['time'] < history_length
    ]
    
    # Calculate moving averages
    temporal_predictions = defaultdict(list)
    for entry in PREDICTION_HISTORY[context]:
        for label, prob in entry['predictions'].items():
            temporal_predictions[label].append(prob)
    
    averaged = {
        label: np.mean(probs) * (1 + 0.2*np.std(probs))
        for label, probs in temporal_predictions.items()
    }
    
    # Apply final thresholds
    valid = [
        (label, prob) 
        for label, prob in averaged.items() 
        if prob >= thresholds.get(label, 0.5)
    ]
    
    return sorted(valid, key=lambda x: x[1], reverse=True)

# Update data preparation pipeline
def create_optimized_dataset(dataset):
    """Enhanced data pipeline with augmentation"""
    augmentation = tf.keras.Sequential([
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
        layers.GaussianNoise(0.001)
    ])
    
    return dataset \
        .map(lambda x, y: (augmentation(x, training=True), y), 
             num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(10000, reshuffle_each_iteration=True) \
        .batch(32) \
        .prefetch(tf.data.AUTOTUNE) \
        .map(lambda x, y: (tf.image.per_image_standardization(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)

def apply_speech_correction(preds, db_level):
    speech_prob = preds.get('speech', 0)
    # Dynamic correction based on dB levels and presence of other sounds
    # Reduced correction factor from 0.3 to 0.25
    correction = 0.25 * (1 - np.exp(-db_level/60))
    corrected_prob = max(speech_prob - correction, 0)
    return {**preds, 'speech': corrected_prob}

# Add the compute_features function
def compute_features(audio_data, sample_rate=16000):
    """
    Compute spectrogram features from audio data for sound classification.
    Enhanced version for better percussive sound detection.
    
    Args:
        audio_data (numpy.ndarray): Audio samples as a numpy array
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Spectrogram features in shape (1, 96, 64, 1) for model input
    """
    try:
        logger.info(f"Computing features from audio data: shape={audio_data.shape}, sr={sample_rate}")
        
        # Check for percussive event
        is_percussive = detect_percussive_event(audio_data, sample_rate)
        
        # Pre-emphasis filter to enhance higher frequencies (better for percussive sounds)
        pre_emphasis = 0.95  # Moderate pre-emphasis coefficient (was 0.97)
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Normalize audio data to -1 to 1 range
        if np.max(np.abs(emphasized_audio)) > 0:
            emphasized_audio = emphasized_audio / np.max(np.abs(emphasized_audio))
        
        # Set spectrogram parameters
        n_fft = 512  # Shorter FFT window size for better time resolution (was 1024)
        hop_length = 256  # Shorter hop for better time resolution (was 512)
        n_mels = 64  # Number of Mel bands
        
        # For percussive sounds, we might want a slightly shorter window
        if is_percussive:
            logger.info("Using shorter window for percussive sound")
            n_fft = 256  # Even shorter window for better time resolution (was 512)
            hop_length = 128  # Shorter hop for better time resolution (was 256)
        
        # Compute spectrogram
        # Short-time Fourier transform
        f, t, Zxx = signal.stft(emphasized_audio, fs=sample_rate, nperseg=n_fft, 
                              noverlap=n_fft-hop_length, window='hann')
        
        # Convert to magnitude spectrogram
        spec = np.abs(Zxx)
        
        # Convert to mel scale
        # Approximate mel scale conversion by taking subset of FFT bins
        # (For proper mel conversion, a full mel filter bank should be used)
        max_freq_idx = int(n_mels * 2)  # Approximate upper frequency limit
        if max_freq_idx < spec.shape[0]:
            spec = spec[:max_freq_idx, :]
        
        # Resize to target shape if needed
        target_time_steps = 96
        if spec.shape[1] > target_time_steps:
            # Take center section if too long
            start = (spec.shape[1] - target_time_steps) // 2
            spec = spec[:, start:start+target_time_steps]
        elif spec.shape[1] < target_time_steps:
            # Pad with zeros if too short
            padding = target_time_steps - spec.shape[1]
            padded_spec = np.zeros((spec.shape[0], target_time_steps))
            padded_spec[:, :spec.shape[1]] = spec
            spec = padded_spec
        
        # Resize frequency dimension if needed
        if spec.shape[0] != n_mels:
            # Simple resize by interpolation (not ideal but works for this purpose)
            from scipy.ndimage import zoom
            zoom_factor = (n_mels / spec.shape[0], 1)
            spec = zoom(spec, zoom_factor, order=1)
        
        # Convert to log scale (log mel spectrogram)
        epsilon = 1e-10  # Small value to avoid log(0)
        spec = np.log(spec + epsilon)
        
        # Normalize to 0-1 range
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + epsilon)
        
        # For percussive sounds, enhance the areas of rapid energy change
        if is_percussive:
            # Calculate time derivatives to emphasize rapid changes
            time_deriv = np.zeros_like(spec)
            time_deriv[:, 1:] = spec[:, 1:] - spec[:, :-1]
            
            # Blend with original, emphasizing transients
            spec = spec + 0.5 * np.abs(time_deriv)  # Higher enhancement factor (was 0.3)
            
            # Re-normalize
            spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + epsilon)
        
        # Reshape to model input format: (1, height, width, channels)
        features = spec.T.reshape(1, target_time_steps, n_mels, 1)
        
        logger.info(f"Computed features with shape {features.shape}")
        return features
        
    except Exception as e:
        logger.error(f"Error computing features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return empty features with expected shape
        return np.zeros((1, 96, 64, 1))
