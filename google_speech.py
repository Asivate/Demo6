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

def transcribe_with_google(audio_data, sample_rate=16000):
    """
    Transcribe audio using Google Cloud Speech-to-Text API.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio data
        
    Returns:
        Transcribed text
    """
    if not GOOGLE_SPEECH_AVAILABLE:
        logger.error("Google Cloud Speech API not available. Install with: pip install google-cloud-speech")
        return ""
    
    try:
        # Ensure audio is in the correct format (16-bit PCM)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create a BytesIO object to hold the audio data
        byte_io = io.BytesIO()
        
        # Write the audio data to the BytesIO object
        byte_io.write(audio_data.tobytes())
        byte_io.seek(0)
        
        # Create a speech client
        client = speech.SpeechClient()
        
        # Configure the audio settings
        audio = speech.RecognitionAudio(content=byte_io.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            model="default",
            enable_automatic_punctuation=True,
            use_enhanced=True,
            # Add speech contexts to improve recognition of specific phrases
            speech_contexts=[
                speech.SpeechContext(
                    phrases=["fire alarm", "doorbell", "phone", "baby", "crying", "water", "running", 
                             "dog", "barking", "cat", "meowing", "door", "knocking", "microwave", 
                             "vacuum", "drill", "alarm", "clock", "cough", "snore", "typing", 
                             "blender", "dishwasher", "flush", "toilet", "hair dryer", "shaver", 
                             "toothbrush", "cooking", "chopping", "cutting", "hammer", "saw", 
                             "car horn", "engine", "hazard", "finger snap", "hand clap", "applause"],
                    boost=10.0
                )
            ]
        )
        
        # Detect speech in the audio
        logger.info("Sending audio to Google Cloud Speech-to-Text API...")
        response = client.recognize(config=config, audio=audio)
        
        # Process the response
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        # Filter out hallucinations
        transcript = filter_hallucinations(transcript)
        
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