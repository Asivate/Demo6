import os
import time
import numpy as np
from google.cloud import speech
from google.cloud import language_v1 as language
import logging
import threading
import queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def categorize_sentiment(score, magnitude):
    """
    Categorize sentiment based on score and magnitude
    
    Args:
        score (float): Sentiment score (-1.0 to 1.0)
        magnitude (float): Sentiment magnitude (0.0+)
    
    Returns:
        str: Human-readable sentiment category
    """
    if score >= 0.7:
        return "very positive"
    elif score >= 0.3:
        return "positive"
    elif score >= 0.1:
        return "slightly positive"
    elif score <= -0.7:
        return "very negative"
    elif score <= -0.3:
        return "negative" 
    elif score <= -0.1:
        return "slightly negative"
    elif magnitude >= 0.6:  # Mixed sentiment with significant magnitude
        return "mixed"
    else:
        return "neutral"

class SpeechProcessor:
    """
    Handles real-time speech recognition using Google Cloud Speech-to-Text API
    with sentiment analysis using Google Cloud Natural Language API.
    """
    
    def __init__(self):
        """Initialize the speech processor with Google Cloud clients."""
        # Ensure the environment variable is set
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Speech recognition may fail.")
        
        # Initialize the Google Cloud Speech client
        self._speech_client = speech.SpeechClient()
        
        # Initialize the Google Cloud Language client for sentiment analysis
        self._language_client = language.LanguageServiceClient()
        
        # Configuration for streaming recognition
        self._config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
        )
        
        self._streaming_config = speech.StreamingRecognitionConfig(
            config=self._config,
            interim_results=True,
        )
        
        # Buffer for audio data
        self._audio_queue = queue.Queue()
        
        # Latest transcription results
        self._latest_results = []
        self._latest_transcript = ""
        self._latest_sentiment = None
        self._buffer = bytearray()
        self._language_code = "en-US"
        
        # Flag to control streaming thread
        self._is_streaming = False
        self._streaming_thread = None

    def start_streaming(self):
        """Start the streaming recognition thread."""
        if self._is_streaming:
            return
            
        self._is_streaming = True
        self._streaming_thread = threading.Thread(target=self._stream_recognition_thread)
        self._streaming_thread.daemon = True
        self._streaming_thread.start()
        logger.info("Speech recognition streaming started")
    
    def stop_streaming(self):
        """Stop the streaming recognition thread."""
        self._is_streaming = False
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)
            logger.info("Speech recognition streaming stopped")
        self._latest_results = []
        self._latest_transcript = ""
        self._latest_sentiment = None
    
    def add_audio_data(self, audio_data, sample_rate=16000):
        """
        Add audio data to the processing queue.
        
        Args:
            audio_data (bytes): Raw audio data
            sample_rate (int): Sample rate of the audio data
        """
        if self._is_streaming:
            # Buffer the audio data for recognition
            if isinstance(audio_data, bytes):
                self._audio_queue.put(audio_data)
            else:
                logger.warning(f"Invalid audio data type: {type(audio_data)}")
    
    def _stream_recognition_thread(self):
        """Thread function for streaming recognition."""
        
        def request_generator():
            # First request contains the configuration
            yield speech.StreamingRecognizeRequest(streaming_config=self._streaming_config)
            
            # Subsequent requests contain audio data
            while self._is_streaming:
                try:
                    # Get audio data from the queue with a timeout
                    chunk = self._audio_queue.get(block=True, timeout=0.5)
                    if chunk:
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    # No audio data available, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in request generator: {e}")
                    break
        
        while self._is_streaming:
            try:
                # Start streaming recognition - provide both config and requests parameters
                responses = self._speech_client.streaming_recognize(
                    config=self._streaming_config,
                    requests=request_generator()
                )
                
                # Process streaming responses
                for response in responses:
                    if not self._is_streaming:
                        break
                    
                    if not response.results:
                        continue
                    
                    for result in response.results:
                        if not result.alternatives:
                            continue
                        
                        transcript = result.alternatives[0].transcript
                        
                        # Process final results (including sentiment analysis)
                        if result.is_final:
                            self._latest_transcript = transcript
                            logger.info(f"Final transcript: {transcript}")
                            
                            # Perform sentiment analysis
                            sentiment_result = self._analyze_sentiment(transcript)
                            if sentiment_result:
                                self._latest_results.append({
                                    'transcript': transcript,
                                    'language': self._language_code[:2],  # Extract language code (e.g., 'en' from 'en-US')
                                    'sentiment_score': sentiment_result['score'],
                                    'sentiment_magnitude': sentiment_result['magnitude'],
                                    'sentiment_category': sentiment_result['category'],
                                    'timestamp': time.time()
                                })
                                # Keep only the latest 10 results
                                if len(self._latest_results) > 10:
                                    self._latest_results = self._latest_results[-10:]
                        else:
                            # Interim results
                            logger.debug(f"Interim transcript: {transcript}")
            
            except Exception as e:
                logger.error(f"Error in speech recognition stream: {e}")
                # Sleep briefly before attempting to restart
                time.sleep(2)
                
                # Clear the audio queue
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
    
    def _analyze_sentiment(self, text):
        """
        Analyze sentiment in the provided text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results with score, magnitude, and category
        """
        if not text or len(text.strip()) == 0:
            return None
        
        try:
            # Create a document object
            document = language.Document(
                content=text,
                type_=language.Document.Type.PLAIN_TEXT,
                language_code=self._language_code[:2]  # 'en' from 'en-US'
            )
            
            # Analyze sentiment
            response = self._language_client.analyze_sentiment(
                document=document
            )
            
            sentiment = response.document_sentiment
            score = sentiment.score
            magnitude = sentiment.magnitude
            
            # Categorize sentiment
            category = categorize_sentiment(score, magnitude)
            
            logger.info(f"Sentiment analysis - Score: {score}, Magnitude: {magnitude}, Category: {category}")
            
            return {
                'score': score,
                'magnitude': magnitude,
                'category': category
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'score': 0.0,
                'magnitude': 0.0,
                'category': 'neutral'
            }
    
    def get_latest_results(self):
        """
        Get the latest recognition results.
        
        Returns:
            list: List of dictionaries containing transcript and sentiment information
        """
        return self._latest_results
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        self._buffer = bytearray()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break 