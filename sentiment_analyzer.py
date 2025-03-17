"""
Sentiment Analysis Module for SoundWatch

This module provides sentiment analysis for transcribed speech,
classifying emotions and providing emojis for visualization.
"""
from transformers import pipeline
import re
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the sentiment analysis model (lazy loading)
_sentiment_model = None

# Define emotion categories and mappings
EMOTION_GROUPS = {
    "Happy": ["joy", "love", "admiration", "approval", "caring", "excitement", "amusement", "gratitude", "optimism", "pride", "relief"],
    "Neutral": ["neutral", "realization", "curiosity"],
    "Surprised": ["surprise", "confusion", "nervousness"],
    "Unpleasant": ["sadness", "fear", "anger", "disgust", "disappointment", "embarrassment", "grief", "remorse", "annoyance", "disapproval"]
}

# Emoji mappings for each category
CATEGORY_EMOJIS = {
    "Happy": "😄",
    "Neutral": "😀",
    "Surprised": "😮",
    "Unpleasant": "😔"
}

# Threshold for low confidence results
CONFIDENCE_THRESHOLD = 0.3

def _get_sentiment_model():
    """
    Lazy loading of the sentiment analysis model.
    
    Returns:
        The sentiment analysis pipeline
    """
    global _sentiment_model
    if _sentiment_model is None:
        try:
            _sentiment_model = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            traceback.print_exc()
    return _sentiment_model

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    return text

def analyze_sentiment(text):
    """
    Analyze the sentiment of the transcribed text.
    
    Args:
        text: Transcribed text
        
    Returns:
        Dictionary with sentiment analysis results
    """
    if not text:
        logger.warning("No text provided for sentiment analysis")
        return None
    
    # Get the sentiment model
    sentiment_model = _get_sentiment_model()
    if not sentiment_model:
        logger.error("Sentiment model not available")
        return None
    
    try:
        # Clean the text
        cleaned_text = preprocess_text(text)
        if not cleaned_text:
            logger.warning("Text was empty after preprocessing")
            return None
        
        # Get sentiment predictions with all scores
        sentiment_results = sentiment_model(cleaned_text)[0]
        
        # Find the emotion with highest confidence
        highest_emotion = max(sentiment_results, key=lambda x: x['score'])
        emotion_label = highest_emotion['label']
        confidence = highest_emotion['score']
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Low confidence sentiment ({confidence:.4f}) for: '{text}'")
            # Default to neutral for low confidence
            emotion_category = "Neutral"
            emotion_label = "neutral"
            confidence = max(confidence, 0.5)  # Set a minimum confidence
        else:
            # Map emotion to broader category
            emotion_category = "Neutral"  # Default
            for category, emotions in EMOTION_GROUPS.items():
                if emotion_label in emotions:
                    emotion_category = category
                    break
        
        # Get corresponding emoji
        emoji = CATEGORY_EMOJIS.get(emotion_category, "😀")
        
        logger.info(f"Sentiment analysis: {emotion_category} ({confidence:.4f}) - {emotion_label}")
        
        return {
            "category": emotion_category,
            "original_emotion": emotion_label,
            "confidence": confidence,
            "emoji": emoji
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        traceback.print_exc()
        return None 