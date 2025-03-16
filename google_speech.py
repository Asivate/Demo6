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
try:
    from google.cloud import speech
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    print("WARNING: Google Cloud Speech API not available. Install with: pip install google-cloud-speech")
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
    hallucination_patterns = [
        r"\bthanks for watching\b",
        r"\bsubscribe\b",
        r"\blike and subscribe\b",
        r"\bclick the link\b",
        r"\bclick below\b",
        r"\bcheck out\b",
        r"\bfollow me\b",
        r"\bmy channel\b",
        r"\bmy website\b",
        r"\bmy social\b",
    ]
    
    for pattern in hallucination_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Clean up any double spaces
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
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "asivate-452914-9778a9b91269.json"),
                    # Check home directory on Linux
                    "/home/hirwa0250/asivate-452914-9778a9b91269.json",
                    # Check parent directory of server (for Sonarity-server structure)
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asivate-452914-9778a9b91269.json")
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
                model="phone_call",  # Changed from "command_and_search" to "phone_call" for better handling of noisy audio
                use_enhanced=True,
                profanity_filter=False,
                enable_automatic_punctuation=True,
                # Increase alternatives to get better results
                max_alternatives=5,  # Increased from 3 to 5 to get more alternatives
                # Enhanced speech adaptation with much broader context
                speech_contexts=[speech.SpeechContext(
                    phrases=[
                        # Common conversational phrases
                        "okay", "not okay", "I am not okay", "I'm not okay", 
                        "I am", "help", "emergency", "alert",
                        "I am tired", "so tired", "feeling tired", "I'm tired",
                        "I am so tired", "I'm so tired", "I am very tired",
                        "happy", "sad", "angry", "confused", "scared", "tired", "sleepy",
                        "good", "bad", "fine", "great", "terrible", "awful",
                        "I feel", "I'm feeling", "I am feeling",
                        # Additional common phrases for better recognition
                        "hello", "hi", "hey", "yes", "no", "please", "thank you",
                        "what", "where", "when", "who", "why", "how",
                        "can you", "I need", "I want", "I would like",
                        "help me", "listen", "understand", "hear", "talk", "speak",
                        "morning", "afternoon", "evening", "night", "today", "tomorrow",
                        "the", "a", "an", "is", "are", "was", "were", "be", "been"
                    ],
                    boost=20.0  # Adjusted for optimal recognition
                )],
                # Enable word-level confidence
                enable_word_confidence=True,
                # Enable word time offsets for timing
                enable_word_time_offsets=True,
                # Enable speaker diarization for better speech detection
                diarization_config=speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=1,
                    max_speaker_count=1,
                ),
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
        logger.info(f"Original audio RMS: {rms:.6f}")
        
        # Enhance speech with spectral subtraction for noise reduction
        try:
            # Apply a pre-emphasis filter to boost high frequencies
            # This helps with speech clarity as human speech has more information in higher frequencies
            pre_emphasis = 0.97
            emphasized_audio = np.append(processed_audio[0], processed_audio[1:] - pre_emphasis * processed_audio[:-1])
            
            # Apply a gentle high-pass filter (40Hz instead of 80Hz) to reduce only very low frequency noise
            # This preserves more of the speech content
            try:
                b, a = signal.butter(2, 40/(sample_rate/2), 'highpass')  # Lower order (2 instead of 4) and lower cutoff (40Hz)
                filtered_audio = signal.filtfilt(b, a, emphasized_audio)
                logger.info("Applied gentle high-pass filter (40Hz) for noise reduction")
                processed_audio = filtered_audio
            except Exception as e:
                logger.warning(f"Error applying high-pass filter: {str(e)}")
                processed_audio = emphasized_audio
                
            # Apply a low-pass filter to remove high-frequency noise above speech range
            try:
                # Properly calculate the normalized frequency (must be between 0 and 1)
                cutoff_hz = 8000  # Speech typically is below 8kHz
                nyquist = sample_rate / 2
                normalized_freq = cutoff_hz / nyquist
                
                # Ensure we stay within valid range (0 < Wn < 1)
                normalized_freq = min(0.99, max(0.01, normalized_freq))
                
                b, a = signal.butter(3, normalized_freq, 'lowpass')  # Keep frequencies up to 8000Hz (speech range)
                filtered_audio = signal.filtfilt(b, a, processed_audio)
                logger.info(f"Applied low-pass filter ({cutoff_hz}Hz) to focus on speech frequencies")
                processed_audio = filtered_audio
            except Exception as e:
                logger.warning(f"Error applying low-pass filter: {str(e)}")
        except Exception as e:
            logger.warning(f"Error in spectral processing: {str(e)}")
        
        # Boost low volume audio for better recognition with a more aggressive approach
        if rms < 0.05:  # If audio is very quiet
            # Calculate boost factor (more boost for quieter audio)
            boost_factor = min(0.15 / rms if rms > 0 else 15, 8)  # Cap at 8x boost (increased from 5x)
            processed_audio = processed_audio * boost_factor
            logger.info(f"Boosted quiet audio by factor of {boost_factor:.2f}")
        
        # Apply normalization to ensure consistent volume
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.9 * 32767  # Scale to 90% of max to prevent clipping
        
        # Calculate new RMS after processing
        new_rms = np.sqrt(np.mean(np.square(processed_audio)))
        logger.info(f"Processed audio RMS: {new_rms:.6f} (gain: {new_rms/rms:.2f}x)")
        
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
                    model="phone_call",  # Changed from "command_and_search" to "phone_call" for better handling of noisy audio
                    use_enhanced=True,
                    profanity_filter=False,
                    enable_automatic_punctuation=True,
                    max_alternatives=5,  # Increased from 3 to 5
                    speech_contexts=[speech.SpeechContext(
                        phrases=[
                            # Common conversational phrases
                            "okay", "not okay", "I am not okay", "I'm not okay", 
                            "I am", "help", "emergency", "alert",
                            "I am tired", "so tired", "feeling tired", "I'm tired",
                            "I am so tired", "I'm so tired", "I am very tired",
                            "happy", "sad", "angry", "confused", "scared", "tired", "sleepy",
                            "good", "bad", "fine", "great", "terrible", "awful",
                            "I feel", "I'm feeling", "I am feeling",
                            # Additional common phrases for better recognition
                            "hello", "hi", "hey", "yes", "no", "please", "thank you",
                            "what", "where", "when", "who", "why", "how",
                            "can you", "I need", "I want", "I would like",
                            "help me", "listen", "understand", "hear", "talk", "speak"
                        ],
                        boost=20.0  # Adjusted for optimal recognition
                    )],
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    # Enable speaker diarization for better speech detection
                    diarization_config=speech.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=1,
                        max_speaker_count=1,
                    ),
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
def transcribe_with_google(audio_data, sample_rate=16000, **options):
    """
    Transcribe audio using Google Cloud Speech-to-Text API.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio data
        **options: Additional options to pass to the RecognitionConfig
            - language_code: Language code (default: "en-US")
            - model: Model to use (default: "phone_call")
            - use_enhanced: Whether to use enhanced model (default: True)
            - enable_automatic_punctuation: Whether to enable automatic punctuation (default: True)
            - audio_channel_count: Number of audio channels (default: 1)
        
    Returns:
        Transcribed text
    """
    if not GOOGLE_SPEECH_AVAILABLE:
        logger.error("Google Cloud Speech API not available. Install with: pip install google-cloud-speech")
        return ""
    
    try:
        # Log audio properties for debugging
        duration = len(audio_data) / sample_rate
        logger.info(f"Transcribing {duration:.2f} seconds of audio (samples: {len(audio_data)})")
        
        # Calculate audio energy to detect silence
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.001:  # Very quiet audio
            logger.info(f"Audio is nearly silent (RMS: {rms:.6f}), skipping transcription")
            return ""
            
        # Boost quiet audio if needed
        if rms < 0.05:
            boost_factor = min(0.1 / max(rms, 0.001), 10.0)
            audio_data = audio_data * boost_factor
            logger.info(f"Boosted quiet audio by factor of {boost_factor:.2f}")
        
        # Ensure audio is in the correct format (16-bit PCM)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create a BytesIO object to hold the audio data
        byte_io = io.BytesIO()
        
        # Write the audio data to the BytesIO object
        byte_io.write(audio_data.tobytes())
        byte_io.seek(0)
        
        # Create a speech client
        client = speech.SpeechClient()
        
        # Extract options with defaults
        language_code = options.get("language_code", "en-US")
        model = options.get("model", "phone_call")  # Changed from "command_and_search" to "phone_call"
        use_enhanced = options.get("use_enhanced", True)
        enable_automatic_punctuation = options.get("enable_automatic_punctuation", True)
        audio_channel_count = options.get("audio_channel_count", 1)
        
        # Configure the audio settings
        audio = speech.RecognitionAudio(content=byte_io.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            model=model,
            enable_automatic_punctuation=enable_automatic_punctuation,
            use_enhanced=use_enhanced,
            audio_channel_count=audio_channel_count,
            # Enhanced speech adaptation with broader context
            speech_contexts=[
                speech.SpeechContext(
                    phrases=["fire alarm", "doorbell", "phone", "baby", "crying", "water", "running", 
                             "dog", "barking", "cat", "meowing", "door", "knocking", "microwave", 
                             "vacuum", "drill", "alarm", "clock", "cough", "snore", "typing", 
                             "blender", "dishwasher", "flush", "toilet", "hair dryer", "shaver", 
                             "toothbrush", "cooking", "chopping", "cutting", "hammer", "saw", 
                             "car horn", "engine", "hazard", "finger snap", "hand clap", "applause",
                             # Common speech phrases for improved accuracy
                             "hello", "hi", "hey", "okay", "yes", "no", "please", "thank you",
                             "I need help", "help me", "emergency", "call", "message", "text",
                             "I don't feel well", "I'm not feeling well", "I feel sick", "I'm sick",
                             "I'm fine", "I am okay", "I'm okay", "I need assistance",
                             # Additional common words and phrases
                             "what", "where", "when", "who", "why", "how",
                             "can you", "I need", "I want", "I would like", 
                             "listen", "understand", "hear", "talk", "speak",
                             "morning", "afternoon", "evening", "night", "today", "tomorrow",
                             "the", "a", "an", "is", "are", "was", "were", "be", "been"],
                    boost=20.0  # Increased boost for better detection
                )
            ],
            # Improved configuration for better transcription
            enable_word_confidence=True,
            enable_word_time_offsets=True,
            # Lower the threshold for acceptable confidence
            enable_separate_recognition_per_channel=False,
            diarization_config=speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=1,
                max_speaker_count=1,
            ),
        )
        
        # Log the request configuration
        logger.info(f"Speech recognition request: {duration:.2f}s audio, model={model}, enhanced={use_enhanced}")
        
        # Detect speech in the audio
        logger.info("Sending audio to Google Cloud Speech-to-Text API...")
        response = client.recognize(config=config, audio=audio)
        
        # Process the response
        transcript = ""
        confidence = 0.0
        
        # Extract all results and their confidence scores
        all_results = []
        for result in response.results:
            if result.alternatives:
                alt = result.alternatives[0]
                all_results.append((alt.transcript, alt.confidence if hasattr(alt, 'confidence') else 0.0))
                
        # Sort by confidence
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Log all detected alternatives for debugging
        for idx, (text, conf) in enumerate(all_results):
            logger.info(f"Alternative {idx+1}: '{text}' (confidence: {conf:.2f})")
            
        # Use the highest confidence result
        if all_results:
            transcript, confidence = all_results[0]
            
        # If confidence is very low, be cautious about returning results
        if confidence < 0.2 and transcript:
            logger.warning(f"Low confidence transcription ({confidence:.2f}): '{transcript}'")
            
            # For very low confidence, don't return potentially incorrect transcriptions
            if confidence < 0.08:
                logger.info("Confidence too low, discarding transcription")
                return ""
        
        # Filter out hallucinations
        transcript = filter_hallucinations(transcript)
        
        logger.info(f"Final transcription: '{transcript}' (confidence: {confidence:.2f})")
        return transcript
    
    except Exception as e:
        logger.error(f"Error in Google Speech transcription: {str(e)}")
        traceback.print_exc()
        return ""

class GoogleSpeechToText:
    """
    Google Cloud Speech-to-Text client for transcribing audio.
    """
    
    def __init__(self, sample_rate=16000):
        """
        Initialize the Google Cloud Speech-to-Text client.
        
        Args:
            sample_rate: Sample rate of the audio data
        """
        if not GOOGLE_SPEECH_AVAILABLE:
            logger.error("Google Cloud Speech API not available. Install with: pip install google-cloud-speech")
            raise ImportError("Google Cloud Speech API not available")
        
        self.sample_rate = sample_rate
        self.client = speech.SpeechClient()
        
    def transcribe(self, audio_data):
        """
        Transcribe audio using Google Cloud Speech-to-Text API.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Transcribed text
        """
        return transcribe_with_google(audio_data, self.sample_rate) 