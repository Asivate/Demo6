"""
Vosk Speech-to-Text Module for SoundWatch

This module provides offline speech recognition functionality using Vosk.
It serves as an alternative to Google Cloud Speech-to-Text when offline operation is needed.
"""
import os
import json
import logging
import numpy as np
from threading import Lock
import traceback
from pathlib import Path
import wget
import zipfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Vosk (will be installed via requirements.txt)
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    VOSK_AVAILABLE = True
    logger.info("Vosk speech recognition is available")
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("Vosk speech recognition is not available")

# Define model URLs for different languages
VOSK_MODEL_URLS = {
    # Small model: ~40MB, fast but less accurate
    "en-us-small": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    # Medium model: ~100MB, balanced performance
    "en-us-medium": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
    # Large model: ~1.6GB, slower but more accurate, better for complex audio
    "en-us-large": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.21.zip"
}

# Define model download lock to prevent multiple simultaneous downloads
model_download_lock = Lock()

# Global Vosk model and recognizer
_vosk_model = None
_vosk_recognizer = None

def initialize_vosk(sample_rate=16000):
    """
    Initialize Vosk speech recognition.
    
    Args:
        sample_rate: Audio sample rate in Hz
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global _vosk_model, _vosk_recognizer, VOSK_AVAILABLE
    
    if not VOSK_AVAILABLE:
        logger.warning("Vosk is not available, cannot initialize")
        return False
    
    try:
        if _vosk_model is None:
            logger.info("Initializing Vosk model...")
            # Use small model for faster inference
            _vosk_model = Model("models/vosk-model-small-en-us-0.15")
            logger.info("Vosk model initialized successfully")
        
        if _vosk_recognizer is None:
            logger.info(f"Initializing Vosk recognizer with sample rate {sample_rate} Hz...")
            _vosk_recognizer = KaldiRecognizer(_vosk_model, sample_rate)
            _vosk_recognizer.SetWords(True)
            logger.info("Vosk recognizer initialized successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing Vosk: {str(e)}")
        VOSK_AVAILABLE = False
        return False

def transcribe_with_vosk(audio_data, sample_rate=16000):
    """
    Transcribe audio data using Vosk speech recognition.
    
    Args:
        audio_data: Numpy array of audio samples
        sample_rate: Audio sample rate in Hz
        
    Returns:
        dict: Transcription result with 'text' key,
              or None if transcription failed
    """
    global _vosk_model, _vosk_recognizer, VOSK_AVAILABLE
    
    if not VOSK_AVAILABLE:
        logger.warning("Vosk is not available, cannot transcribe")
        return None
    
    try:
        # Initialize Vosk if not already initialized
        if _vosk_model is None or _vosk_recognizer is None:
            if not initialize_vosk(sample_rate):
                logger.error("Failed to initialize Vosk")
                return None
        
        # Convert audio data to bytes
        if isinstance(audio_data, np.ndarray):
            # Convert to int16 and then to bytes
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_data
        
        # Process audio data
        if _vosk_recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(_vosk_recognizer.Result())
            text = result.get('text', '')
            
            if text:
                logger.info(f"Vosk transcription: {text}")
                return {'text': text, 'engine': 'vosk'}
            else:
                logger.debug("Vosk returned empty transcription")
                return None
        else:
            # No speech detected
            logger.debug("Vosk did not detect speech")
            return None
    
    except Exception as e:
        logger.error(f"Error transcribing with Vosk: {str(e)}")
        return None

# Define filter_hallucinations function similar to the one in speech_to_text.py
def filter_hallucinations(text):
    """
    Filter out common hallucination patterns from transcribed text.
    
    Args:
        text: The transcribed text to filter
        
    Returns:
        Filtered text with hallucination patterns removed
    """
    if not text:
        return text
        
    # Check for repetitive patterns (a common hallucination issue)
    words = text.split()
    if len(words) >= 6:
        # Check for exact repetition of 3+ word phrases
        for i in range(len(words) - 5):
            phrase1 = " ".join(words[i:i+3])
            for j in range(i+3, len(words) - 2):
                phrase2 = " ".join(words[j:j+3])
                if phrase1.lower() == phrase2.lower():
                    # Found repetition, keep only the first occurrence
                    return " ".join(words[:j])
    
    # Remove common hallucination phrases
    common_hallucinations = [
        "thank you for watching",
        "thanks for watching", 
        "please subscribe",
        "like and subscribe",
        "click the link",
        "click below",
        "check out",
        "follow me",
        "my channel",
        "my website",
        "my social"
    ]
    
    for phrase in common_hallucinations:
        if phrase in text.lower():
            text = text.lower().replace(phrase, "")
    
    # Clean up any double spaces and capitalize first letter
    text = " ".join(text.split())
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    return text

class VoskSpeechToText:
    """
    Singleton class for Vosk speech recognition.
    """
    _instance = None
    _lock = Lock()
    _is_initialized = False
    _max_retries = 3
    _initialization_in_progress = False
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VoskSpeechToText, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not self._is_initialized and not self._initialization_in_progress:
            self._initialize()
    
    def _initialize(self):
        """
        Initialize the Vosk speech recognition system.
        """
        # Set flag to prevent multiple simultaneous initializations
        VoskSpeechToText._initialization_in_progress = True
        
        try:
            if not VOSK_AVAILABLE:
                logger.error("Vosk is not available. Please install it with: pip install vosk")
                VoskSpeechToText._initialization_in_progress = False
                return
            
            # Set log level to reduce verbosity
            SetLogLevel(-1)
            
            # Define model directory
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Define model path - using large model for better accuracy
            model_name = "vosk-model-en-us-0.21"
            model_path = os.path.join(model_dir, model_name)
            
            # Check if model exists, if not download it
            if not os.path.exists(model_path) or not os.listdir(model_path):
                with model_download_lock:
                    # Check again in case another thread downloaded it while we were waiting
                    if not os.path.exists(model_path) or not os.listdir(model_path):
                        logger.info(f"Downloading Vosk large model {model_name}...")
                        model_url = VOSK_MODEL_URLS["en-us-large"]
                        zip_path = os.path.join(model_dir, f"{model_name}.zip")
                        
                        # Download the model with retry logic
                        retry_count = 0
                        download_success = False
                        
                        while retry_count < self._max_retries and not download_success:
                            try:
                                if os.path.exists(zip_path):
                                    os.remove(zip_path)  # Remove existing zip if download was interrupted
                                
                                # Download with progress reporting
                                wget.download(model_url, zip_path)
                                logger.info(f"\nDownloaded model to {zip_path}")
                                download_success = True
                            except Exception as e:
                                retry_count += 1
                                logger.error(f"Download attempt {retry_count} failed: {str(e)}")
                                if retry_count < self._max_retries:
                                    logger.info(f"Retrying in 3 seconds...")
                                    time.sleep(3)
                                else:
                                    logger.error(f"Failed to download model after {self._max_retries} attempts")
                                    raise
                        
                        # Check if download directory exists, create if needed
                        if not os.path.exists(model_path):
                            os.makedirs(model_path, exist_ok=True)
                        
                        # Extract the model
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                # Get the top-level directory from the zip file
                                top_dir = zip_ref.namelist()[0].split('/')[0] if '/' in zip_ref.namelist()[0] else None
                                
                                # Extract files
                                zip_ref.extractall(model_dir)
                                logger.info(f"Extracted model to {model_dir}")
                                
                                # If extracted to a subdirectory, move files to model_path
                                if top_dir and top_dir != model_name:
                                    extracted_dir = os.path.join(model_dir, top_dir)
                                    if os.path.exists(extracted_dir) and os.path.isdir(extracted_dir):
                                        # Rename directory to match expected name
                                        if os.path.exists(model_path) and model_path != extracted_dir:
                                            import shutil
                                            shutil.rmtree(model_path, ignore_errors=True)
                                        os.rename(extracted_dir, model_path)
                                        logger.info(f"Renamed directory {extracted_dir} to {model_path}")
                        except Exception as e:
                            logger.error(f"Error extracting model: {str(e)}")
                            raise
                        
                        # Remove the zip file to save space
                        try:
                            os.remove(zip_path)
                            logger.info(f"Removed zip file {zip_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove zip file: {str(e)}")
            
            # Verify the model path exists and has content
            if not os.path.exists(model_path) or not os.listdir(model_path):
                raise FileNotFoundError(f"Model path {model_path} is empty or does not exist after download attempt")
            
            # Load the model with retry logic
            retry_count = 0
            while retry_count < self._max_retries:
                try:
                    logger.info(f"Loading Vosk large model from {model_path}...")
                    logger.info(f"Note: This may take a while as the large model is approximately 1.6GB in size")
                    self.model = Model(model_path)
                    logger.info("Vosk large model loaded successfully - this should provide better transcription accuracy")
                    break
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error loading model (attempt {retry_count}): {str(e)}")
                    if retry_count >= self._max_retries:
                        raise
                    time.sleep(2)  # Wait before retrying
            
            # Create a recognizer with default sample rate
            self.sample_rate = 16000
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            # Enable words with timestamps
            self.recognizer.SetWords(True)
            
            # Mark as initialized
            VoskSpeechToText._is_initialized = True
            logger.info("Vosk speech recognition initialized successfully with large model (0.21)")
            logger.info("Note: Large model provides better accuracy but requires more memory and processing time")
            
        except Exception as e:
            error_msg = f"Error initializing Vosk speech recognition: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
        finally:
            # Reset initialization flag regardless of success/failure
            VoskSpeechToText._initialization_in_progress = False
    
    def preprocess_audio(self, audio_data, sample_rate=16000):
        """
        Preprocess audio data to improve speech recognition quality.
        
        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio (default: 16000)
            
        Returns:
            numpy.ndarray: The preprocessed audio data
        """
        if len(audio_data) == 0:
            return audio_data
            
        # Convert to proper data type
        processed_audio = audio_data.astype(np.float32)
        
        # Calculate RMS value to check audio level
        rms = np.sqrt(np.mean(processed_audio**2))
        
        # Boost low volume audio for better recognition
        if rms < 0.05:  # If audio is very quiet
            # Calculate boost factor (more boost for quieter audio)
            boost_factor = min(0.1 / rms if rms > 0 else 10, 5)  # Cap at 5x boost
            processed_audio = processed_audio * boost_factor
            logger.info(f"Boosted quiet audio by factor of {boost_factor:.2f}")
        
        # Normalize audio to 16-bit range for Vosk
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 32767
        
        # Convert to int16 for Vosk
        return processed_audio.astype(np.int16)

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data to text using Vosk.
        
        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio (default: 16000)
            
        Returns:
            str: The transcribed text
        """
        if not VOSK_AVAILABLE:
            logger.error("Vosk is not available. Please install it with: pip install vosk")
            return ""
            
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data provided for transcription")
            return ""
        
        # Initialize if not already done
        if not VoskSpeechToText._is_initialized:
            logger.info("Vosk not initialized, initializing now...")
            self._initialize()
            if not VoskSpeechToText._is_initialized:
                logger.error("Failed to initialize Vosk")
                return ""
        
        try:
            # Preprocess audio data
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            
            # Check if sample rate matches the recognizer's sample rate
            if sample_rate != self.sample_rate:
                logger.info(f"Recreating recognizer with sample rate {sample_rate}")
                self.sample_rate = sample_rate
                self.recognizer = KaldiRecognizer(self.model, sample_rate)
                self.recognizer.SetWords(True)
            
            # Convert audio data to bytes
            audio_bytes = processed_audio.tobytes()
            
            # Process audio data
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                logger.info(f"Vosk transcription: '{text}'")
                
                # Filter hallucinations
                filtered_text = filter_hallucinations(text)
                if filtered_text != text:
                    logger.info(f"Filtered hallucination: '{text}' -> '{filtered_text}'")
                
                return filtered_text
            else:
                # Get partial result
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                logger.debug(f"Vosk partial transcription: '{partial_text}'")
                return ""
                
        except Exception as e:
            logger.error(f"Error in Vosk transcription: {str(e)}", exc_info=True)
            # Reset recognizer on error to avoid state corruption
            try:
                self.reset()
            except:
                pass
            return ""
    
    def reset(self):
        """
        Reset the recognizer to clear any buffered audio.
        """
        if not VOSK_AVAILABLE or not self._is_initialized:
            return
            
        try:
            # Recreate the recognizer with the same sample rate
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            logger.info("Vosk recognizer reset")
        except Exception as e:
            logger.error(f"Error resetting Vosk recognizer: {str(e)}", exc_info=True)

# Singleton instance for easy access
def get_vosk_transcriber():
    """
    Get the singleton instance of VoskSpeechToText.
    
    Returns:
        VoskSpeechToText: The singleton instance
    """
    return VoskSpeechToText() 