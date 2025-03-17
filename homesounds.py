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

# Define indices for percussive sounds
KNOCK_IDX = 6      # Door knock index in the model output
DISHES_IDX = 4     # Dishes index in the model output
GLASS_BREAKING_IDX = 9  # Glass breaking index in the model output
CAR_HORN_IDX = 2   # Car horn index

# Define percussive sounds that need special handling
percussive_sounds_names = ['Door knock', 'Glass breaking', 'Dishes', 'Car horn']
percussive_sounds = [KNOCK_IDX, DISHES_IDX, GLASS_BREAKING_IDX, CAR_HORN_IDX]  # Indices for percussive sounds

# Add missing threshold constants
PREDICTION_THRES = 0.2  # Default prediction threshold for sound detection - lowered from 0.3 to 0.2
DBLEVEL_THRES = -30  # Threshold for dB level to consider sound significant

# Define sound-specific thresholds - adjusted based on observed confidence levels
sound_specific_thresholds = {
    'Door knock': 0.1,    # Lowered from 0.15 to 0.1 to better detect knocks
    'Dishes': 0.2,        # Lowered from 0.4 to 0.2 but still higher than knock
    'Glass breaking': 0.1, # Lowered to better detect breaking glass
    'door-bell': 0.2,     # Lowered from 0.4
    'dog-bark': 0.2,      # Lowered from 0.4
    'baby-cry': 0.15,     # Lowered from 0.3
    'phone-ring': 0.2,    # Lowered from 0.35
    'vacuum-cleaner': 0.2, # Lowered from 0.35
    'Car horn': 0.1       # Added based on logs showing low confidence for car horn
    # Default threshold of 0.2 will be used for any sound not specified here
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
    if name in percussive_sounds_names
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
        if sound_class not in percussive_sounds_names:
            return current_confidence
            
        current_time = time.time()
        
        # Check if we've detected this sound recently
        last_time = self.last_detection.get(sound_class, 0)
        time_since_last = current_time - last_time
        
        # For percussive sounds, we enhance confidence if:
        # 1. The sound hasn't been detected recently (avoid duplicate detections)
        # 2. The dB level is sufficiently high (indicating a sharp sound)
        # 3. There's at least some base confidence
        min_confidence = sound_specific_thresholds.get(sound_class, PREDICTION_THRES) * 0.3  # Only need 30% of threshold for boosting (was 50%)
        
        # Adjust for knocking with special handling
        if sound_class == 'Door knock' and current_confidence > min_confidence:  # Lower activation threshold
            # Check for dB spike which is common with knocks
            if db_level > -25 and time_since_last > min_time_between:  # More sensitive dB threshold (was -20)
                # Boost confidence for knock detection - higher multiplier
                adjusted_confidence = min(current_confidence * 3.5, 0.95)  # Boost by 3.5x (was 2.5x)
                # Record this detection time
                self.last_detection[sound_class] = current_time
                logger.info(f"Enhanced Door knock detection: {current_confidence:.4f} → {adjusted_confidence:.4f}")
                return adjusted_confidence
            
        # Special handling for Dishes class, which often activates during knocking
        if sound_class == 'Dishes' and current_confidence > min_confidence * 0.8:
            # If it's a sharp sound with high dB, it might be a knock misclassified as dishes
            if db_level > -20 and time_since_last > min_time_between:  # More sensitive dB threshold (was -15)
                # Check if we have 'knock' data in history to ensure we're not creating false positives
                if 'Door knock' in self.history and len(self.history['Door knock']) > 0:
                    knock_confidence = self.get_smoothed_confidence('Door knock')
                    # If there's any knock confidence at all, transfer some confidence from dishes to knock
                    if knock_confidence > 0.02:  # Lowered from 0.05
                        # Transfer 60% of dishes confidence to Door knock (was 40%)
                        knock_boost = current_confidence * 0.6
                        logger.info(f"Transferring confidence from Dishes to Door knock: {knock_boost:.4f}")
                        
                        # Update Door knock confidence
                        transfer_time = time.time()
                        if 'Door knock' not in self.history:
                            self.history['Door knock'] = []
                        self.history['Door knock'].append((transfer_time, knock_boost + knock_confidence))
                        
                        # Record this as a Door knock detection
                        self.last_detection['Door knock'] = current_time
                        
                        # Return lower confidence for dishes
                        return current_confidence * 0.4
        
        # Special handling for Car horn - added based on logs
        if sound_class == 'Car horn' and current_confidence > min_confidence:
            if db_level > -20 and time_since_last > min_time_between:
                # Boost confidence for car horn detection
                adjusted_confidence = min(current_confidence * 3.0, 0.95)  # Boost by 3x
                # Record this detection time
                self.last_detection[sound_class] = current_time
                logger.info(f"Enhanced Car horn detection: {current_confidence:.4f} → {adjusted_confidence:.4f}")
                return adjusted_confidence
        
        return current_confidence

# Create a global instance for use in the server
detection_history = SoundDetectionHistory(window_size=3, decay_factor=0.7)

# Add a new method to detect abrupt changes in audio levels that might indicate knocking
def detect_percussive_event(audio_data, sample_rate, threshold=0.2, min_gap=0.1, max_gap=0.8):
    """
    Detect percussive events in audio data, with improved detection for knocking patterns.
    
    Args:
        audio_data (numpy.ndarray): Audio samples as a numpy array
        sample_rate (int): Sample rate of the audio in Hz
        threshold (float): Threshold for detecting peaks (lower = more sensitive)
        min_gap (float): Minimum gap between peaks in seconds
        max_gap (float): Maximum gap between peaks in seconds to consider as part of same event
        
    Returns:
        bool: True if a percussive event is detected, False otherwise
    """
    try:
        logger.info(f"Checking for percussive event in audio shape={audio_data.shape}")
        
        # Calculate frame size based on 20ms windows
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        
        # Calculate energy in each frame
        num_frames = len(audio_data) // frame_size
        if num_frames < 3:  # Need at least 3 frames for meaningful analysis
            return False
            
        frame_energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio_data))
            frame = audio_data[start:end]
            frame_energy[i] = np.sum(frame**2) / len(frame)
        
        # Normalize energy
        if np.max(frame_energy) > 0:
            frame_energy = frame_energy / np.max(frame_energy)
        
        # Find peaks in energy
        # A peak is defined as a frame with energy higher than threshold and higher than adjacent frames
        peaks = []
        for i in range(1, len(frame_energy) - 1):
            if (frame_energy[i] > threshold and 
                frame_energy[i] > frame_energy[i-1] and 
                frame_energy[i] > frame_energy[i+1]):
                peaks.append(i)
        
        if len(peaks) == 0:
            return False
            
        # Calculate peak times in seconds
        peak_times = [p * frame_size / sample_rate for p in peaks]
        
        # Check for double knocks or triple knocks (common door knock patterns)
        # This is looking for 2 or 3 peaks within a short time window with appropriate gaps
        min_gap_frames = int(min_gap * sample_rate / frame_size)
        max_gap_frames = int(max_gap * sample_rate / frame_size)
        
        # Check for pairs of peaks with appropriate spacing
        for i in range(len(peaks) - 1):
            gap = peaks[i+1] - peaks[i]
            if min_gap_frames <= gap <= max_gap_frames:
                logger.info(f"Detected double-knock pattern with {gap * frame_size / sample_rate:.3f}s gap")
                return True
                
        # Check for triple knocks (three peaks with appropriate spacing)
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                gap1 = peaks[i+1] - peaks[i]
                gap2 = peaks[i+2] - peaks[i+1]
                # Both gaps should be within reasonable range
                if (min_gap_frames <= gap1 <= max_gap_frames and 
                    min_gap_frames <= gap2 <= max_gap_frames):
                    logger.info(f"Detected triple-knock pattern with gaps {gap1 * frame_size / sample_rate:.3f}s and {gap2 * frame_size / sample_rate:.3f}s")
                    return True
        
        # Check for one very strong peak (could be a single loud knock)
        strong_peak_threshold = 0.8
        if any(frame_energy > strong_peak_threshold):
            logger.info(f"Detected single strong percussive event with peak energy {np.max(frame_energy):.3f}")
            return True
            
        # Calculate rate of energy change (first derivative)
        energy_change = np.diff(frame_energy)
        
        # Check for sudden increases in energy (sharp attacks)
        sharp_attack_threshold = 0.4  # Lower threshold for more sensitivity
        if any(energy_change > sharp_attack_threshold):
            logger.info(f"Detected sharp attack with max energy change {np.max(energy_change):.3f}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error detecting percussive event: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        is_percussive = detect_percussive_event(audio_data, sample_rate, threshold=0.2)  # Lower threshold for more sensitivity
        
        # Pre-emphasis filter to enhance higher frequencies (better for percussive sounds)
        pre_emphasis = 0.92  # Lower pre-emphasis coefficient for clearer sound (was 0.95)
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Normalize audio data to -1 to 1 range
        if np.max(np.abs(emphasized_audio)) > 0:
            emphasized_audio = emphasized_audio / np.max(np.abs(emphasized_audio))
        
        # Set spectrogram parameters
        n_fft = 400  # Shorter FFT window size for better time resolution (was 512)
        hop_length = 200  # Shorter hop for better time resolution (was 256)
        n_mels = 64  # Number of Mel bands
        
        # For percussive sounds, we might want a slightly shorter window
        if is_percussive:
            logger.info("Using shorter window for percussive sound")
            n_fft = 200  # Even shorter window for better time resolution (was 256)
            hop_length = 100  # Shorter hop for better time resolution (was 128)
        
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
            spec = spec + 0.7 * np.abs(time_deriv)  # Higher enhancement factor (was 0.5)
            
            # Re-normalize
            spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + epsilon)
            
            # Add frequency derivative enhancement for better capturing transients
            freq_deriv = np.zeros_like(spec)
            freq_deriv[1:, :] = spec[1:, :] - spec[:-1, :]
            
            # Add this to the spectrogram too
            spec = spec + 0.4 * np.abs(freq_deriv)
            
            # Re-normalize again
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
