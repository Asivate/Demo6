# Testing Speech Recognition and Sentiment Analysis

This guide outlines how to test the speech recognition and sentiment analysis features in SoundWatch.

## Prerequisites

Before testing, ensure you have:

1. Set up Google Cloud credentials (`google_cloud_credentials.json`)
2. Enabled the required APIs in your Google Cloud project:
   - Cloud Speech-to-Text API
   - Cloud Natural Language API
3. Started the server with the proper environment variables:
   ```
   source setup_google_cloud.sh
   python server.py
   ```

## Testing Process

### 1. Testing Server-Side Processing

1. Start the server with verbose logging:
   ```
   python server.py --debug
   ```

2. Monitor the console output for the following:
   - "Google Cloud clients initialized successfully" at startup
   - "Speech processing thread started" message
   - "Transcript: [text]" when speech is processed
   - "Sentiment analysis: score=[value], emotion=[label], emoji=[symbol]" for sentiment analysis results

### 2. Testing Watch Notifications

1. Ensure the watch is connected to the server
2. Speak near the watch microphone
3. Verify that you receive notifications with:
   - Speech label
   - Transcription text
   - Sentiment emoji
   - Emotion label

### 3. Testing Conversation History

1. Generate several speech transcriptions
2. Open the mobile app
3. Tap "View Conversation History"
4. Verify that your transcriptions appear in the list
5. Check that each entry shows:
   - Transcription text
   - Timestamp
   - Sentiment emoji and emotion
   - Sentiment score

## Troubleshooting

### If speech is not recognized:

1. Check the server logs for error messages
2. Verify that Google Cloud credentials are properly set
3. Ensure the audio is not too quiet (check dB levels in logs)
4. Ensure you're speaking in a supported language (default is English)

### If sentiment analysis is not working:

1. Check that transcription is successful
2. Verify that the Natural Language API is enabled
3. Try speaking with more emotionally charged language

### If conversation history is empty:

1. Verify that transcriptions are being processed
2. Check that the phone app is receiving the data from the watch
3. Check for any JSON parsing errors in the logs

## Example Phrases for Testing

Test the sentiment analysis with these phrases:

- Positive: "I'm having a wonderful day, everything is going great!"
- Negative: "This is terrible, I'm very disappointed with the results."
- Neutral: "The meeting is scheduled for Tuesday at 2 PM."
- Mixed: "The food was delicious but the service was really slow." 