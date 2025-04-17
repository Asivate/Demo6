# Running SoundWatch Server on Ubuntu Virtual Machine

This README provides instructions for setting up and running the SoundWatch server with sentiment analysis on an Ubuntu virtual machine.

## Setup Instructions

1. **Transfer Files to VM**

   Transfer the server directory to your Ubuntu VM using scp, rsync, or any other file transfer method:
   
   ```bash
   # From your local machine
   scp -r /path/to/SoundWatch/server username@vm-ip-address:~/
   ```

2. **Set Up Google Cloud Credentials**

   To enable sentiment analysis, you need Google Cloud credentials:
   
   - Create a service account and download the JSON key file (see SENTIMENT_SETUP.md for details)
   - Transfer the JSON key file to your VM
   - Set the environment variable to point to your credentials:
   
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-credentials.json
   ```

3. **Make Scripts Executable**

   Once files are on the VM, make the scripts executable:
   
   ```bash
   chmod +x run_server.sh test_sentiment.py record_test_audio.py
   ```

## Running the Server

1. **Test Sentiment Analysis**

   Before running the full server, verify that sentiment analysis is working:
   
   ```bash
   ./test_sentiment.py
   ```
   
   This will test the sentiment analysis functionality with sample phrases.

2. **Record Test Audio (Optional)**

   If you want to test with your own speech:
   
   ```bash
   ./record_test_audio.py
   ```
   
   This will record audio from your microphone and save it to `test_audio.wav`.

3. **Start the Server**

   Use the run script to start the server with proper environment setup:
   
   ```bash
   ./run_server.sh /path/to/your-credentials.json
   ```
   
   If you don't provide a path to credentials, the script will look for a file named `google-credentials.json` in the current directory or your home directory.

## Verifying Everything Works

1. **Test Endpoint**

   Once the server is running, open a web browser and navigate to:
   
   ```
   http://<vm-ip-address>:8080/test_sentiment
   ```
   
   This will show test results for sentiment analysis.

2. **Connect from SoundWatch App**

   Update the server URL in the SoundWatch app to point to your VM's IP address:
   
   - Change `server/DEFAULT_SERVER` to `"http://<vm-ip-address>:8080"` in the app's Constants file
   - Change `server/WS_SERVER_URL` to `"ws://<vm-ip-address>:8080"` in the app's Constants file

## Troubleshooting

1. **Dependencies Issues**

   If you encounter dependency issues, install them manually:
   
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-venv python3-dev portaudio19-dev
   ```

2. **Port Accessibility**

   Make sure port 8080 is accessible on your VM:
   
   ```bash
   # Allow traffic on port 8080
   sudo ufw allow 8080
   ```

3. **Check Logs**

   Monitor the server logs for any issues:
   
   ```bash
   # In a separate terminal
   tail -f nohup.out
   ```

4. **Run in Background**

   To keep the server running after you log out:
   
   ```bash
   nohup ./run_server.sh > server.log 2>&1 &
   ```

## Additional Notes

- The sentiment analysis requires an internet connection to access Google Cloud APIs
- For consistent results, use a VM with at least 2GB of RAM and 1 CPU core
- If you're running on a public cloud VM, make sure your firewall rules allow traffic on port 8080 