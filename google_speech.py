"""
Google Cloud Speech-to-Text Module for SoundWatch

This module provides speech recognition functionality using Google Cloud Speech-to-Text API.
It supports both synchronous and streaming recognition for better transcription quality.
"""
import os
import io
import time
import queue
import threading
import logging
import numpy as np
from google.cloud import speech
from threading import Lock
import traceback
import re
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    ]
    
    for phrase in common_hallucinations:
        if phrase in text.lower():
            # Remove the hallucination phrase
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub("", text)
    
    # Clean up any resulting double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class GoogleSpeechToText:
    _instance = None
    _lock = Lock()
    _is_initialized = False
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GoogleSpeechToText, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not self._is_initialized:
            self._initialize()
    
    def _initialize(self):
        """
        Initialize the Google Cloud Speech-to-Text client.
        """
        try:
            # Check if environment variable is already set
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ and os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
                logger.info(f"Using Google Cloud credentials from environment variable: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
            else:
                # Try to find credentials in these locations
                possible_locations = [
                    # Check server directory first
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "asivate-452914-5c12101797af.json"),
                    # Check home directory on Linux
                    "/home/hirwa0250/asivate-452914-5c12101797af.json",
                    # Check parent directory of server (for Sonarity-server structure)
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asivate-452914-5c12101797af.json")
                ]
                
                # Find the first existing file
                credentials_path = None
                for path in possible_locations:
                    if os.path.exists(path):
                        credentials_path = path
                        break
                
                if not credentials_path:
                    logger.error(f"Google Cloud credentials file not found in any of these locations: {possible_locations}")
                    logger.error("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable or place credentials file in one of the above locations")
                    raise FileNotFoundError(f"Google Cloud credentials file not found")
                
                # Set environment variable for authentication
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                logger.info(f"Using Google Cloud credentials from file: {credentials_path}")
            
            # Initialize the client
            self.client = speech.SpeechClient()
            logger.info("Google Cloud Speech-to-Text client initialized successfully")
            
            # Create recognition config
            self.config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                model="command_and_search",  # Better for short phrases than "video"
                use_enhanced=True,
                profanity_filter=False,
                enable_automatic_punctuation=True,
                # Increase alternatives to get better results
                max_alternatives=3,
                # Enable speech adaptation to improve recognition of specific contexts
                speech_contexts=[speech.SpeechContext(
                    phrases=[
                        "okay", "not okay", "I am not okay", "I'm not okay", 
                        "I am", "help", "emergency", "alert",
                        "I am tired", "so tired", "feeling tired", "I'm tired",
                        "I am so tired", "I'm so tired", "I am very tired",
                        "happy", "sad", "angry", "confused", "scared", "tired", "sleepy",
                        "good", "bad", "fine", "great", "terrible", "awful",
                        "I feel", "I'm feeling", "I am feeling"
                    ],
                    boost=25.0  # Increased boost for better detection
                )],
                # Enable word-level confidence
                enable_word_confidence=True,
                # Enable word time offsets for timing
                enable_word_time_offsets=True
            )
            
            # Create a streaming config as well
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=self.config,
                interim_results=False
            )
            
            # Mark as initialized
            GoogleSpeechToText._is_initialized = True
            
        except Exception as e:
            error_msg = f"Error initializing Google Speech-to-Text client: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
    
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
            boost_factor = min(0.1 / rms if rms > 0 else 10, 10)  # Cap at 10x boost
            processed_audio = processed_audio * boost_factor
            logger.info(f"Boosted quiet audio by factor of {boost_factor:.2f}")
        
        # Apply noise reduction if signal is noisy
        # This is a simple high-pass filter to reduce low-frequency noise
        if sample_rate > 1000:  # Only apply if we have enough frequency resolution
            try:
                # High-pass filter to reduce background noise (cutoff at 80Hz)
                b, a = signal.butter(4, 80/(sample_rate/2), 'highpass')
                processed_audio = signal.filtfilt(b, a, processed_audio)
                logger.info("Applied high-pass filter for noise reduction")
            except Exception as e:
                logger.warning(f"Could not apply filter: {str(e)}")
        
        # Normalize audio to 16-bit range for LINEAR16 encoding
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 32767
        
        # Convert to int16 for Google Speech API
        return processed_audio.astype(np.int16)

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data to text using Google Cloud Speech-to-Text API.
        
        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio (default: 16000)
            
        Returns:
            str: The transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data provided for transcription")
            return ""
        
        try:
            # Preprocess audio data
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            
            # Convert audio data to appropriate format
            audio_content = processed_audio.tobytes()
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_content)
            
            # Update config to match current sample rate
            if sample_rate != self.config.sample_rate_hertz:
                self.config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    language_code="en-US",
                    model="command_and_search",  # Better for short phrases than "video"
                    use_enhanced=True,
                    profanity_filter=False,
                    enable_automatic_punctuation=True,
                    max_alternatives=3,
                    speech_contexts=[speech.SpeechContext(
                        phrases=[
                            "okay", "not okay", "I am not okay", "I'm not okay", 
                            "I am", "help", "emergency", "alert",
                            "I am tired", "so tired", "feeling tired", "I'm tired",
                            "I am so tired", "I'm so tired", "I am very tired",
                            "happy", "sad", "angry", "confused", "scared", "tired", "sleepy",
                            "good", "bad", "fine", "great", "terrible", "awful",
                            "I feel", "I'm feeling", "I am feeling"
                        ],
                        boost=25.0  # Increased boost for better detection
                    )],
                    enable_word_confidence=True,
                    enable_word_time_offsets=True
                )
            
            # Detect speech using Google Cloud Speech-to-Text
            logger.info("Sending audio to Google Cloud Speech-to-Text API...")
            response = self.client.recognize(config=self.config, audio=audio)
            
            # Process the response
            transcription = ""
            for result in response.results:
                transcription += result.alternatives[0].transcript
            
            # Apply post-processing to remove possible hallucinations
            transcription = filter_hallucinations(transcription)
            
            logger.info(f"Google transcription result: '{transcription}'")
            return transcription
            
        except Exception as e:
            logger.error(f"Error in Google Cloud Speech-to-Text transcription: {str(e)}")
            traceback.print_exc()
            return ""

# Function for use in the main server code
def transcribe_with_google(audio_data, sample_rate=16000):
    """
    Wrapper function for easier integration with server code.
    
    Args:
        audio_data: Audio data to transcribe
        sample_rate: Sample rate of the audio
        
    Returns:
        The transcribed text
    """
    speech_processor = GoogleSpeechToText()
    return speech_processor.transcribe(audio_data, sample_rate) 