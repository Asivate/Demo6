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
    logging.warning("WARNING: Google Cloud Speech API not available. Install with: pip install google-cloud-speech")
from threading import Lock
import traceback
import re
from scipy import signal
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced credential handling function
def setup_google_credentials():
    """Set up Google Cloud credentials for the Speech API"""
    # Check if environment variable already exists
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    if creds_path and os.path.exists(creds_path):
        logger.info(f"Using Google Cloud credentials from environment variable: {creds_path}")
        return True
        
    # Potential locations to check for credential files
    possible_locations = [
        # Home directory with exact filename
        os.path.expanduser("~/asivate-452914-9778a9b91269.json"),
        # Server directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asivate-452914-9778a9b91269.json"),
        # Parent directory of server
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asivate-452914-9778a9b91269.json"),
        # Look for any .json files in home directory that might be credential files
        *[os.path.join(os.path.expanduser("~"), f) for f in os.listdir(os.path.expanduser("~")) 
          if f.endswith('.json') and os.path.isfile(os.path.join(os.path.expanduser("~"), f))]
    ]
    
    # Find and validate credential files
    for path in possible_locations:
        if os.path.exists(path):
            logger.info(f"Found potential credentials file: {path}")
            try:
                # Validate that it's actually a credentials file
                with open(path, 'r') as f:
                    cred_data = json.load(f)
                    if 'type' in cred_data and cred_data['type'] == 'service_account':
                        # Set the environment variable
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
                        logger.info(f"✅ Successfully set GOOGLE_APPLICATION_CREDENTIALS to {path}")
                        
                        # Print details about the service account being used
                        logger.info(f"Using service account: {cred_data.get('client_email')}")
                        logger.info(f"Project ID: {cred_data.get('project_id')}")
                        
                        return True
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"File {path} is not a valid credentials file: {e}")
                continue
    
    logger.error("No valid Google Cloud credentials file found!")
    logger.error("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable manually")
    logger.error("Example: export GOOGLE_APPLICATION_CREDENTIALS=~/your-credentials-file.json")
    return False

# Call the setup function at module load time
CREDENTIALS_SETUP = setup_google_credentials()

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
            
            # Verify credentials with a minimal test request to check authentication
            try:
                logger.info("Verifying Google Cloud credentials with test request...")
                # Create a minimal audio content (silence) for testing
                test_audio = speech.RecognitionAudio(content=b"\0" * 1600)  # 0.1 seconds of silence
                test_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                )
                # Make a minimal request to verify API access
                self.client.recognize(config=test_config, audio=test_audio)
                logger.info("✅ Google Cloud credentials verified successfully")
            except Exception as e:
                logger.error(f"❌ Google Cloud credentials verification failed: {str(e)}")
                logger.error("This will cause transcription to fail - please check your credentials")
                # We don't raise an exception here to allow the system to continue running,
                # but transcription will likely fail later
            
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
                # Enhanced speech adaptation with a more focused context
                speech_contexts=[speech.SpeechContext(
                    phrases=[
                        # Common conversational phrases - minimal set to avoid biasing too much
                        "okay", "help", "emergency", "alert",
                        "tired", "happy", "sad", "angry", "confused", "scared",
                        "good", "bad", "fine", "not okay", "I need help",
                        # Important function words that help with sentence structure
                        "I am", "I'm", "the", "a", "an", "is", "are", "was", "were",
                        "what", "where", "when", "who", "why", "how"
                    ],
                    boost=5.0  # Reduced from 20.0 to 5.0 to minimize hallucinations
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
                b, a = signal.butter(3, 8000/(sample_rate/2), 'lowpass')  # Keep frequencies up to 8000Hz (speech range)
                filtered_audio = signal.filtfilt(b, a, processed_audio)
                logger.info("Applied low-pass filter (8000Hz) to focus on speech frequencies")
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
def transcribe_with_google(audio_data, sample_rate=16000):
    """
    Transcribe audio using Google Cloud Speech-to-Text API.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        Transcription text or empty string if transcription failed
    """
    try:
        # Validate and log audio data info
        if audio_data is None:
            logger.error("No audio data provided for transcription")
            return ""
            
        if not isinstance(audio_data, np.ndarray):
            try:
                logger.warning(f"Converting non-numpy audio data of type {type(audio_data)} to numpy array")
                audio_data = np.array(audio_data, dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed to convert audio data to numpy array: {e}")
                return ""
        
        # Log audio data statistics
        try:
            logger.info(f"transcribe_with_google: audio_data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            if len(audio_data) > 0:
                logger.info(f"audio stats: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean={np.mean(audio_data):.4f}, non-zero={np.count_nonzero(audio_data)}")
                # Check if audio has actual content
                if np.count_nonzero(audio_data) < len(audio_data) * 0.01:  # Less than 1% non-zero
                    logger.warning("Audio data appears to be mostly silent (< 1% non-zero values)")
        except Exception as e:
            logger.error(f"Error analyzing audio data: {e}")
        
        if len(audio_data) < 1000:  # Too short for meaningful transcription
            logger.warning(f"Audio data too short for transcription: {len(audio_data)} samples")
            return ""
            
        # Initialize Google Speech client
        try:
            from google.cloud import speech
        except ImportError:
            logger.error("Google Cloud Speech API not available. Install with: pip install google-cloud-speech")
            return ""
            
        # Ensure credentials are set up
        if not CREDENTIALS_SETUP:
            logger.info("Attempting to set up credentials again...")
            setup_google_credentials()
            
        # Check if credentials are available
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            logger.error("Google Cloud credentials not set. Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
            logger.error("Transcription will likely fail without valid credentials.")
        else:
            # Log details about the credentials file being used
            creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            logger.info(f"Using credentials file: {creds_path}")
            if os.path.exists(creds_path):
                logger.info(f"✅ Credentials file exists")
                try:
                    with open(creds_path, 'r') as f:
                        cred_data = json.load(f)
                        logger.info(f"Service account: {cred_data.get('client_email')}")
                        logger.info(f"Project ID: {cred_data.get('project_id')}")
                except Exception as e:
                    logger.error(f"Error reading credentials file: {e}")
            else:
                logger.error(f"❌ Credentials file does not exist at {creds_path}")
        
        # Attempt to create a client
        try:
            client = speech.SpeechClient()
            logger.info("Successfully created Speech client")
        except Exception as e:
            logger.error(f"Failed to initialize Google Speech client: {e}")
            return ""
        
        # Convert audio to the correct format for Google Speech API
        try:
            # Ensure audio is normalized between -1 and 1
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                
            # Convert to int16 format that Google Speech API expects (scale to full range)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            logger.info(f"Converted audio to int16, length in bytes: {len(audio_bytes)}")
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return ""
        
        # Create RecognitionAudio object
        audio = speech.RecognitionAudio(content=audio_bytes)
        
        # Create RecognitionConfig with enhanced settings
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            model="phone_call",  # Good for noisy audio
            max_alternatives=3,
            enable_automatic_punctuation=True,
            use_enhanced=True,
        )
        
        # Perform the transcription
        logger.info("Sending audio to Google Cloud Speech API...")
        try:
            response = client.recognize(config=config, audio=audio)
            
            # Process the response
            if not response.results:
                logger.warning("No transcription results returned from Google Speech API")
                return ""
            
            # Get the most likely transcription
            transcription = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            logger.info(f"Google Speech API transcription result: '{transcription}' (confidence: {confidence:.2f})")
            
            # Return the transcription
            return transcription
        except Exception as e:
            logger.error(f"Error in Google Speech API recognize call: {e}")
            if "403" in str(e) and "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(e):
                logger.error("Permission denied: Your service account doesn't have the right permissions")
                logger.error("You need to give your service account the 'Speech API user' role")
                logger.error("Go to the Google Cloud Console -> IAM -> Edit service account -> Add 'Speech API user' role")
            return ""
        
    except Exception as e:
        # Format detailed error information for easier debugging
        error_message = f"Google Cloud Speech API error: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        # Return empty string on error
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

# New streaming transcription implementation
class AudioStream:
    """A generator class that yields audio chunks for streaming recognition."""
    def __init__(self):
        self.chunks = []
        self.closed = False
        self._lock = threading.Lock()
    
    def add_chunk(self, chunk):
        """Add an audio chunk to the stream."""
        with self._lock:
            self.chunks.append(chunk)
    
    def close(self):
        """Close the stream."""
        self.closed = True
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Yield the next audio chunk."""
        while not self.closed and not self.chunks:
            time.sleep(0.1)  # Wait for more audio data
        
        with self._lock:
            if self.chunks:
                return self.chunks.pop(0)
            elif self.closed:
                raise StopIteration
            else:
                return None

def streaming_transcribe_with_google(audio_stream, sample_rate=16000):
    """
    Transcribe streaming audio using Google Cloud Speech-to-Text API.
    
    Args:
        audio_stream: A generator yielding audio chunks
        sample_rate: Sample rate of the audio
        
    Returns:
        Generator yielding transcription results
    """
    try:
        if not GOOGLE_SPEECH_AVAILABLE:
            logger.error("Google Cloud Speech API not available. Install with: pip install google-cloud-speech")
            return
            
        # Ensure credentials are set up
        if not CREDENTIALS_SETUP:
            logger.warning("Attempting to set up credentials again...")
            setup_google_credentials()
            
        # Create the Speech client
        client = speech.SpeechClient()
        
        # Create RecognitionConfig for streaming
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            model="phone_call",  # Good for noisy audio
            enable_automatic_punctuation=True,
            use_enhanced=True,
        )
        
        # Create StreamingRecognitionConfig
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True  # Get interim results as speech is processed
        )
        
        # Generator to convert audio chunks to streaming requests
        def request_generator():
            # First request must contain only the config
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            
            # Subsequent requests contain audio data
            for chunk in audio_stream:
                if chunk is None:
                    continue
                    
                if isinstance(chunk, np.ndarray):
                    # Ensure audio is normalized between -1 and 1
                    if np.max(np.abs(chunk)) > 0:
                        chunk = chunk / np.max(np.abs(chunk))
                        
                    # Convert to int16 format that Google Speech API expects
                    chunk_int16 = (chunk * 32767).astype(np.int16)
                    chunk_bytes = chunk_int16.tobytes()
                elif isinstance(chunk, bytes):
                    chunk_bytes = chunk
                else:
                    # Skip invalid chunks
                    logger.warning(f"Skipping invalid chunk type: {type(chunk)}")
                    continue
                
                if chunk_bytes:
                    yield speech.StreamingRecognizeRequest(audio_content=chunk_bytes)
        
        # Start streaming recognition
        logger.info("Starting streaming transcription...")
        streaming_responses = client.streaming_recognize(request_generator())
        
        # Process streaming responses
        num_chars_printed = 0
        current_transcript = ""
        
        for response in streaming_responses:
            if not response.results:
                continue
                
            # The result might be for intermediate or final results
            result = response.results[0]
            if not result.alternatives:
                continue
                
            # Display the transcription
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence if result.is_final else 0
            
            # If the result is final, return it
            if result.is_final:
                logger.info(f"Final transcript: '{transcript}' (confidence: {confidence:.2f})")
                current_transcript = ""
                yield {
                    'transcript': transcript,
                    'is_final': True,
                    'confidence': confidence
                }
            else:
                # For interim results, just log them
                if transcript != current_transcript:
                    logger.debug(f"Interim transcript: '{transcript}'")
                    current_transcript = transcript
                    yield {
                        'transcript': transcript, 
                        'is_final': False,
                        'confidence': 0.0
                    }
                    
    except Exception as e:
        logger.error(f"Error in streaming transcription: {e}")
        logger.error(traceback.format_exc())
        if "403" in str(e) and "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(e):
            logger.error("Permission denied: Your service account doesn't have the right permissions")
            logger.error("You need to give your service account the 'Speech API user' role")
            logger.error("Go to the Google Cloud Console -> IAM -> Edit service account -> Add 'Speech API user' role")
        return

# Global audio stream for streaming recognition
streaming_audio = AudioStream()

# Function to add audio to the streaming transcription
def add_to_streaming_transcription(audio_data):
    """
    Add audio data to the streaming transcription.
    
    Args:
        audio_data: Audio data as numpy array
    """
    global streaming_audio
    if audio_data is not None and len(audio_data) > 0:
        streaming_audio.add_chunk(audio_data)

# Function to close the streaming transcription
def close_streaming_transcription():
    """Close the streaming transcription."""
    global streaming_audio
    streaming_audio.close() 