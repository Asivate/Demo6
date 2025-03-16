"""
Continuous Sentiment Analysis Module for SoundWatch

This module provides a separate service for continuous sentiment analysis of speech.
It operates independently of the main transcription process, allowing for:
1. Real-time sentiment analysis of any detected speech
2. Conversation history tracking
3. Independent notification capabilities
"""
import time
import logging
import threading
import queue
import numpy as np
import json
from datetime import datetime
from threading import Lock
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Google Speech API components
try:
    from google_speech import transcribe_with_google, GoogleSpeechToText
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    logger.warning("Google Speech API not available. Install google-cloud-speech package.")

# Try to import Vosk as fallback
try:
    from vosk_speech import transcribe_with_vosk, VOSK_AVAILABLE
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("Vosk speech recognition not available")

# Import sentiment analysis
from sentiment_analyzer import analyze_sentiment

class ContinuousSentimentAnalyzer:
    """
    Service for continuous sentiment analysis of speech audio.
    This runs as a separate thread and processes audio chunks independently.
    """
    
    def __init__(self, socketio, sample_rate=16000, history_size=50):
        """
        Initialize the continuous sentiment analyzer.
        
        Args:
            socketio: Flask-SocketIO instance for sending notifications
            sample_rate: Audio sample rate (default: 16000)
            history_size: Maximum number of conversation items to store
        """
        self.socketio = socketio
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        self.lock = Lock()
        self.audio_queue = queue.Queue()
        
        # For conversation history
        self.history_size = history_size
        self.conversation_history = deque(maxlen=history_size)
        
        # Speech recognition preference
        self.use_google_speech = GOOGLE_SPEECH_AVAILABLE
        
        logger.info("Continuous Sentiment Analyzer initialized")
    
    def start(self):
        """Start the sentiment analysis service."""
        with self.lock:
            if self.running:
                logger.warning("Sentiment analysis service already running")
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._process_audio_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Sentiment analysis service started")
    
    def stop(self):
        """Stop the sentiment analysis service."""
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            # Add None to queue to signal shutdown
            self.audio_queue.put(None)
            
            if self.thread:
                self.thread.join(timeout=5.0)
                logger.info("Sentiment analysis service stopped")
    
    def add_audio(self, audio_data):
        """
        Add audio data to the processing queue.
        
        Args:
            audio_data: Audio data as numpy array
        """
        if self.running and audio_data is not None and len(audio_data) > 0:
            self.audio_queue.put(audio_data)
    
    def _process_audio_loop(self):
        """Main processing loop that runs in a separate thread."""
        buffer = []
        last_process_time = time.time()
        silence_threshold = 0.01  # Adjust based on your audio normalization
        
        while self.running:
            try:
                # Get audio data with timeout to allow for checking self.running
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Check for shutdown signal
                if audio_data is None:
                    break
                
                # Add to buffer
                buffer.append(audio_data)
                
                # Process if buffer is large enough or enough time has passed
                current_time = time.time()
                buffer_duration = sum(len(chunk) for chunk in buffer) / self.sample_rate
                
                # Process if we have 2+ seconds of audio or 3+ seconds since last process
                if buffer_duration >= 2.0 or (current_time - last_process_time >= 3.0 and buffer):
                    # Concatenate audio chunks
                    concat_audio = np.concatenate(buffer)
                    
                    # Check if audio contains speech (basic energy-based check)
                    audio_energy = np.sqrt(np.mean(np.square(concat_audio)))
                    
                    if audio_energy > silence_threshold:
                        # Process the audio for sentiment
                        self._analyze_sentiment(concat_audio)
                    
                    # Reset buffer and timer
                    buffer = []
                    last_process_time = current_time
            
            except queue.Empty:
                # Queue timeout - check if we should process current buffer
                if buffer and time.time() - last_process_time >= 3.0:
                    concat_audio = np.concatenate(buffer)
                    audio_energy = np.sqrt(np.mean(np.square(concat_audio)))
                    
                    if audio_energy > silence_threshold:
                        self._analyze_sentiment(concat_audio)
                    
                    buffer = []
                    last_process_time = time.time()
            
            except Exception as e:
                logger.error(f"Error in sentiment analysis thread: {str(e)}", exc_info=True)
                buffer = []
                last_process_time = time.time()
    
    def _analyze_sentiment(self, audio_data):
        """
        Analyze sentiment from audio data.
        
        Args:
            audio_data: Audio data as numpy array
        """
        try:
            # Transcribe audio
            transcription = ""
            transcription_engine = "none"
            error_info = None
            
            if self.use_google_speech:
                try:
                    transcription, error_info = transcribe_with_google(audio_data, self.sample_rate)
                    if transcription:
                        transcription_engine = "google"
                except Exception as e:
                    logger.warning(f"Google speech recognition failed: {str(e)}")
                    error_info = {"error": str(e)}
            
            # Try Vosk fallback if Google failed or isn't available
            if not transcription and VOSK_AVAILABLE:
                try:
                    transcription = transcribe_with_vosk(audio_data, self.sample_rate)
                    if transcription:
                        transcription_engine = "vosk"
                except Exception as e:
                    logger.warning(f"Vosk speech recognition failed: {str(e)}")
            
            # If we have a transcription, analyze sentiment
            if transcription:
                # Analyze sentiment
                sentiment_result = analyze_sentiment(transcription)
                
                if sentiment_result:
                    # Create conversation item
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conversation_item = {
                        "text": transcription,
                        "sentiment": sentiment_result,
                        "timestamp": timestamp,
                        "engine": transcription_engine
                    }
                    
                    # Add to history
                    self.conversation_history.append(conversation_item)
                    
                    # Emit notification
                    self._emit_sentiment_notification(conversation_item)
                    
                    logger.info(f"Sentiment analyzed: {sentiment_result['category']} - '{transcription[:30]}...' if len(transcription) > 30 else transcription")
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
            return False
    
    def _emit_sentiment_notification(self, conversation_item):
        """
        Send sentiment notification to connected clients.
        
        Args:
            conversation_item: The conversation item with transcription and sentiment
        """
        try:
            notification_data = {
                'type': 'sentiment_analysis',
                'transcription': conversation_item['text'],
                'sentiment': conversation_item['sentiment'],
                'timestamp': conversation_item['timestamp'],
                'engine': conversation_item['engine']
            }
            
            self.socketio.emit('sentiment_analysis', notification_data)
            logger.debug(f"Sentiment notification emitted: {conversation_item['sentiment']['category']}")
        
        except Exception as e:
            logger.error(f"Error emitting sentiment notification: {str(e)}", exc_info=True)
    
    def get_conversation_history(self, limit=None):
        """
        Get conversation history.
        
        Args:
            limit: Maximum number of items to return (None for all)
            
        Returns:
            List of conversation items (newest first)
        """
        with self.lock:
            history = list(self.conversation_history)
            history.reverse()  # Newest first
            
            if limit is not None and limit > 0:
                return history[:limit]
            return history

# Global instance to be used by server
sentiment_analyzer = None

def initialize_sentiment_analyzer(socketio, sample_rate=16000):
    """
    Initialize the global sentiment analyzer instance.
    
    Args:
        socketio: Flask-SocketIO instance
        sample_rate: Audio sample rate
        
    Returns:
        The initialized sentiment analyzer
    """
    global sentiment_analyzer
    
    if sentiment_analyzer is None:
        sentiment_analyzer = ContinuousSentimentAnalyzer(socketio, sample_rate)
    
    return sentiment_analyzer

def get_sentiment_analyzer():
    """Get the global sentiment analyzer instance."""
    return sentiment_analyzer 