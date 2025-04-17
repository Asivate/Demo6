#!/usr/bin/env python3
"""
Standalone script to test Google Cloud sentiment analysis functionality
without running the full server.
"""
import os
import sys
import time

# Check if Google Cloud libraries are installed
try:
    from google.cloud import language_v1
    from google.cloud import speech
except ImportError:
    print("Google Cloud libraries not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "google-cloud-speech==2.0.1", 
                          "google-cloud-language==2.0.0"])
    from google.cloud import language_v1
    from google.cloud import speech

def get_sentiment_emoji(score):
    """Convert sentiment score to appropriate emoji"""
    if score > 0.7:
        return "😄"
    elif score > 0.3:
        return "🙂"
    elif score > -0.3:
        return "😐"
    elif score > -0.7:
        return "😕"
    else:
        return "😢"
        
def get_sentiment_emotion(score):
    """Convert sentiment score to emotion label"""
    if score > 0.7:
        return "Very Positive"
    elif score > 0.3:
        return "Positive"
    elif score > -0.3:
        return "Neutral"
    elif score > -0.7:
        return "Negative"
    else:
        return "Very Negative"

def check_credentials():
    """Check if Google Cloud credentials are properly configured"""
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not cred_path:
        print("⚠️ GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print("Sentiment analysis will not work without valid credentials.")
        return False
    
    if not os.path.exists(cred_path):
        print(f"⚠️ Credentials file does not exist at: {cred_path}")
        return False
    
    if not os.access(cred_path, os.R_OK):
        print(f"⚠️ Credentials file exists but is not readable: {cred_path}")
        return False
    
    print(f"✓ Credentials found at: {cred_path}")
    return True

def test_sentiment_analysis():
    """Test the sentiment analysis functionality"""
    if not check_credentials():
        print("\nTo set credentials:")
        print("export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-credentials.json")
        user_input = input("\nDo you want to continue without credentials? (y/n): ")
        if user_input.lower() not in ['y', 'yes']:
            return
    
    try:
        # Initialize the client
        print("\n📝 Initializing Google Cloud Language client...")
        client = language_v1.LanguageServiceClient()
        print("✓ Client initialized successfully")
        
        # Test phrases with different sentiments
        test_phrases = [
            "I am absolutely delighted with how well this is working!",
            "This is terrible, I don't like it at all.",
            "The weather seems fine today."
        ]
        
        print("\n🧪 Testing sentiment analysis on sample phrases:")
        
        for phrase in test_phrases:
            print(f"\n📣 Phrase: \"{phrase}\"")
            
            # Process with Natural Language API for sentiment
            document = language_v1.Document(
                content=phrase,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            sentiment = client.analyze_sentiment(document=document).document_sentiment
            emoji = get_sentiment_emoji(sentiment.score)
            emotion = get_sentiment_emotion(sentiment.score)
            
            print(f"📊 Sentiment score: {sentiment.score:.2f}")
            print(f"📏 Magnitude: {sentiment.magnitude:.2f}")
            print(f"😀 Emoji: {emoji}")
            print(f"🎭 Emotion: {emotion}")
        
        print("\n✅ Sentiment analysis is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_speech_recognition():
    """Test speech recognition functionality if audio file is available"""
    if not check_credentials():
        print("Skipping speech recognition test due to missing credentials.")
        return False
    
    # Check if we have a test audio file
    test_files = ["test_audio.wav", "sample.wav", "speech_sample.wav"]
    audio_file = None
    
    for file in test_files:
        if os.path.exists(file):
            audio_file = file
            break
    
    if not audio_file:
        print("\n⚠️ No test audio file found. Skipping speech recognition test.")
        print("To test speech recognition, place a WAV file named 'test_audio.wav' in this directory.")
        return False
    
    try:
        print(f"\n🎤 Testing speech recognition with file: {audio_file}")
        
        # Initialize speech client
        speech_client = speech.SpeechClient()
        
        # Read the audio file
        with open(audio_file, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True
        )
        
        print("🔍 Sending audio to Google Speech-to-Text API...")
        response = speech_client.recognize(config=config, audio=audio)
        
        if not response.results:
            print("❌ No speech detected in the audio file.")
            return False
        
        for i, result in enumerate(response.results):
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence
            
            print(f"\n📋 Result {i+1}:")
            print(f"📝 Transcript: \"{transcript}\"")
            print(f"🎯 Confidence: {confidence:.4f}")
            
            # Now analyze the sentiment of the transcript
            document = language_v1.Document(
                content=transcript,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            sentiment = speech_client.analyze_sentiment(document=document).document_sentiment
            emoji = get_sentiment_emoji(sentiment.score)
            emotion = get_sentiment_emotion(sentiment.score)
            
            print(f"📊 Sentiment score: {sentiment.score:.2f}")
            print(f"📏 Magnitude: {sentiment.magnitude:.2f}")
            print(f"😀 Emoji: {emoji}")
            print(f"🎭 Emotion: {emotion}")
        
        print("\n✅ Speech recognition and sentiment analysis pipeline is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing speech recognition: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 SoundWatch Sentiment Analysis Test")
    print("=" * 60)
    
    # Test sentiment analysis
    sentiment_works = test_sentiment_analysis()
    
    # Only test speech if sentiment worked
    if sentiment_works:
        print("\n" + "=" * 60)
        speech_works = test_speech_recognition()
    else:
        speech_works = False
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"🎭 Sentiment Analysis: {'✅ Working' if sentiment_works else '❌ Not working'}")
    print(f"🎤 Speech Recognition: {'✅ Working' if speech_works else '❌ Not tested or not working'}")
    print("=" * 60)
    
    if not sentiment_works:
        print("\n⚠️ Sentiment analysis is not working. Please check:")
        print("1. Your Google Cloud credentials are set correctly")
        print("2. The Natural Language API is enabled in your Google Cloud project")
        print("3. Your project has billing enabled")
    
    if sentiment_works and not speech_works:
        print("\n⚠️ Speech recognition is not working. Please check:")
        print("1. The Speech-to-Text API is enabled in your Google Cloud project")
        print("2. You have a valid audio file for testing")
        print("3. The audio file contains speech in the expected language (English)") 