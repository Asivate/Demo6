import os
import time
import threading
import queue
import numpy as np
from google.cloud import speech
from google.cloud import language_v1
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('speech_processor')

class SpeechProcessor:
    """
    Handles continuous speech recognition and sentiment analysis.
    This class manages a buffer of audio data and processes it for both
    speech recognition and sentiment analysis.
    """
    
    def __init__(self):
        # Check if Google credentials are set
        if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        
        # Initialize Google Cloud clients
        self.speech_client = speech.SpeechClient()
        self.language_client = language_v1.LanguageServiceClient()
        
        # Set up speech recognition config
        self.speech_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.speech_config,
            interim_results=True
        )
        
        # Audio buffer and processing state
        self.audio_buffer = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Keep track of the latest transcription and sentiment
        self.latest_transcript = ""
        self.latest_sentiment = None
        
    def start_processing(self):
        """Start the processing thread for continuous speech recognition."""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Speech processing started")
        
    def stop_processing(self):
        """Stop the processing thread."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        logger.info("Speech processing stopped")
        
    def add_audio_data(self, audio_data):
        """
        Add audio data to the buffer for processing.
        
        Args:
            audio_data (bytes): Raw audio data to process
        """
        self.audio_buffer.put(audio_data)
        
    def _process_audio_stream(self):
        """
        Continuously process audio data from the buffer for speech recognition.
        This runs in a separate thread.
        """
        def audio_generator():
            while self.is_processing:
                # Use a timeout to allow checking is_processing condition
                try:
                    chunk = self.audio_buffer.get(block=True, timeout=0.1)
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    continue
        
        while self.is_processing:
            try:
                responses = self.speech_client.streaming_recognize(
                    config=self.streaming_config,
                    requests=audio_generator()
                )
                
                # Process streaming responses
                self._handle_responses(responses)
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")
                # Brief pause before retrying
                time.sleep(1)
    
    def _handle_responses(self, responses):
        """
        Process streaming recognition responses.
        
        Args:
            responses: The stream of responses from the speech recognition service.
        """
        for response in responses:
            if not response.results:
                continue
                
            # Get the most recent result
            result = response.results[0]
            
            if not result.alternatives:
                continue
                
            transcript = result.alternatives[0].transcript
            
            # If this is a final result, analyze sentiment
            if result.is_final:
                self.latest_transcript = transcript
                sentiment_data = self._analyze_sentiment(transcript)
                self.latest_sentiment = sentiment_data
                
                # Log the results
                logger.info(f"Transcript: {transcript}")
                logger.info(f"Sentiment: {sentiment_data}")
                
                # Return a dictionary with both transcript and sentiment
                result_data = {
                    'transcript': transcript,
                    'sentiment': sentiment_data
                }
                
                # This is where we'd emit the result to connected clients
                # In the server.py file, we'll handle the emitting
                return result_data
    
    def _analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: A dictionary containing sentiment analysis results
        """
        try:
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            sentiment = self.language_client.analyze_sentiment(document=document).document_sentiment
            
            # Determine sentiment category based on score
            category = "neutral"
            if sentiment.score >= 0.25:
                category = "positive"
            elif sentiment.score <= -0.25:
                category = "negative"
            
            # Return structured sentiment data
            return {
                'score': sentiment.score,
                'magnitude': sentiment.magnitude,
                'category': category
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'score': 0,
                'magnitude': 0,
                'category': 'error'
            }
    
    def get_latest_results(self):
        """
        Get the latest transcript and sentiment analysis results.
        
        Returns:
            dict: Dictionary containing the latest transcript and sentiment analysis
        """
        return {
            'transcript': self.latest_transcript,
            'sentiment': self.latest_sentiment
        } 