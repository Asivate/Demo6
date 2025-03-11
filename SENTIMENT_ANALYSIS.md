# Speech Sentiment Analysis for SoundWatch

This document explains the enhanced speech sentiment analysis feature for SoundWatch.

## Overview

SoundWatch has been enhanced with an improved sentiment analysis capability for detected speech. When the system detects speech, it now:

1. Buffers audio to collect more context
2. Applies advanced audio preprocessing for better speech clarity
3. Transcribes the speech to text with higher accuracy
4. Filters out poor quality or too-short transcriptions
5. Analyzes the emotional content of valid transcriptions
6. Categorizes the sentiment into one of four categories
7. Displays a notification with an appropriate emoji

## Sentiment Categories

Speech is categorized into four main sentiment types, each with its own emoji:

| Sentiment   | Emoji | Description                                  |
|-------------|-------|----------------------------------------------|
| Happy       | 😊    | Positive emotions (joy, admiration, etc.)    |
| Neutral     | 😀    | Neutral emotions (no strong sentiment)       |
| Surprised   | 😲    | Surprise or excitement                       |
| Unpleasant  | 😣    | Negative emotions (anger, sadness, fear, etc.) |

## Technical Implementation

The enhanced sentiment analysis system consists of several improved components:

1. **Audio Buffer System**: Collects multiple audio chunks for better context
2. **Advanced Audio Preprocessing**: 
   - Improved bandpass filtering (250-3400Hz)
   - Noise floor reduction
   - Dynamic range compression
   - Mirror padding for short segments

3. **Speech-to-Text with Quality Filtering**:
   - Minimum audio quality checks
   - Extended transcription context (2+ seconds)
   - Improved Whisper model parameters
   - Filtering of common false positives ("you", "thank you", etc.)
   - Minimum meaningful word count requirements

4. **Sentiment Analysis**:
   - Uses the roberta-base-go_emotions model for 28 emotion detection
   - Emotion mapping to four main categories

## Configuration

Sentiment analysis has several configurable parameters in `server.py`:

```python
# Speech detection threshold (increased from 0.12 to 0.35 for better accuracy)
SPEECH_SENTIMENT_THRES = 0.35

# Audio context multiplier (increased from 1.5 to 2.5 seconds)
SPEECH_CHUNK_MULTIPLIER = 2.5

# Quality filters
MIN_TRANSCRIPTION_LENGTH = 3  # Minimum characters in transcription
MIN_MEANINGFUL_WORDS = 2  # Minimum number of meaningful words
```

## Usage

When speech is detected with sufficient confidence (>35%), the system will:

1. Buffer and process the audio for optimal quality
2. Attempt to transcribe the speech
3. If the transcription meets quality standards:
   - Analyze the sentiment
   - Send a notification to the watch with:
     - The sentiment category ("Speech Happy", "Speech Neutral", etc.)
     - An appropriate emoji
     - The transcribed text

## Best Practices for Users

For optimal speech recognition and sentiment analysis:

1. Speak clearly and at a normal-to-loud volume
2. Speak for at least 2-3 seconds continuously
3. Minimize background noise
4. Position the watch where its microphone can clearly capture your voice
5. Use complete sentences rather than single words

## Troubleshooting

If sentiment analysis isn't working as expected:

1. Check that speech is detected with >35% confidence
2. Ensure you're speaking clearly and long enough for valid transcription
3. Verify the server is running with all models properly loaded
4. Check the server logs for specific error messages or filtering reasons

## Models and Resources

- Speech recognition: [OpenAI Whisper small](https://huggingface.co/openai/whisper-small)
- Sentiment analysis: [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)

## Advanced Technical Details

The enhanced system implements several best practices for audio processing:

1. **Audio Buffering**: Maintains a rolling buffer of recent audio chunks, allowing the system to use more context for better transcription accuracy.

2. **Dynamic Range Compression**: Applies audio compression to make quieter speech parts more audible while preventing loud parts from distorting.

3. **Quality Filtering**: Not all detected speech can be reliably transcribed. The system now filters out:
   - Very short utterances
   - Common false positives ("you", "the", etc.)
   - Transcriptions with insufficient meaningful words

4. **Improved Audio Preprocessing**: Multiple stages of audio enhancement:
   - Stronger pre-emphasis (0.98) for consonant clarity
   - Wider bandpass filter (250-3400Hz vs 300-3000Hz)
   - Noise floor removal
   - Smart padding using reflection for better continuity 