# Setting Up Sentiment Analysis for SoundWatch

## Overview

SoundWatch's real-time sentiment analysis feature uses Google Cloud's Natural Language API and Speech-to-Text API to process spoken language and determine its emotional tone. This guide will help you set up the necessary credentials and run the server with sentiment analysis enabled.

## Prerequisites

- A Google Cloud Platform account (You can create one at [cloud.google.com](https://cloud.google.com/))
- A GCP project with the following APIs enabled:
  - Cloud Speech-to-Text API
  - Cloud Natural Language API
- Service account credentials with access to these APIs

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" or select an existing project
3. Make note of your Project ID

## Step 2: Enable Required APIs

1. In the Cloud Console, go to "APIs & Services" > "Library"
2. Search for and enable:
   - "Cloud Speech-to-Text API"
   - "Cloud Natural Language API"

## Step 3: Create a Service Account and Credentials

1. In the Cloud Console, go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Give your service account a name (e.g., "soundwatch-sentiment")
4. Assign the following roles:
   - "Cloud Speech-to-Text User"
   - "Cloud Natural Language User"
5. Click "Create Key" (JSON format)
6. Save the JSON file to a secure location

## Step 4: Set Up the Server

1. Copy your credentials JSON file to the server directory:
   ```bash
   cp /path/to/your-credentials.json server/google-credentials.json
   ```

2. Make the run script executable:
   ```bash
   chmod +x server/run_server.sh
   ```

3. Run the server using the script:
   ```bash
   cd server
   ./run_server.sh
   ```
   - If you saved your credentials with a different name, provide the path:
   ```bash
   ./run_server.sh /path/to/your-credentials.json
   ```

## Verifying Sentiment Analysis Setup

To verify that sentiment analysis is working properly:

1. Start the server with credentials properly set up
2. Visit http://localhost:8080/test_sentiment in your browser
   - This endpoint will test sentiment analysis on sample phrases
   - You should see sentiment scores, emojis, and emotion descriptions

## Troubleshooting

### Sentiment Analysis is Disabled

If you see "Google Cloud services not enabled. Sentiment analysis will be disabled":

1. Check that the credentials file exists and is readable
2. Verify the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set:
   ```bash
   echo $GOOGLE_APPLICATION_CREDENTIALS
   ```
3. Ensure your Google Cloud project has billing enabled
4. Check that the required APIs are enabled in your project

### Authorization Errors

If you see "Permission denied" or "Not authorized":

1. Verify that your service account has the correct roles
2. Check that your credentials haven't expired
3. Make sure your project has billing set up correctly

### API Quotas

Google Cloud APIs have quotas. If you exceed them:

1. Check your usage in the Google Cloud Console
2. Consider upgrading your account or requesting higher quotas

## Running on a Virtual Machine

When running on an Ubuntu VM:

1. Transfer your credentials file to the VM:
   ```bash
   scp /path/to/your-credentials.json username@your-vm-ip:~/
   ```

2. Connect to your VM:
   ```bash
   ssh username@your-vm-ip
   ```

3. Follow the setup steps above
4. Make sure the VM has internet access to connect to Google Cloud APIs

## Note on Environment Variables

For production environments, you might want to permanently set the credentials:

```bash
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-credentials.json"' >> ~/.bashrc
source ~/.bashrc
``` 