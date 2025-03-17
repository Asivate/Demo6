# SoundWatch Server Cloud Deployment Guide

This guide walks you through deploying the SoundWatch server on a cloud virtual machine so it's accessible from anywhere.

## Prerequisites

- A cloud virtual machine (VM) with:
  - Linux operating system (Ubuntu recommended)
  - Python 3.6 or higher
  - Public IP address
  - Open port 8080 in the firewall

## Setup Steps

### 1. Upload Server Code

Transfer the `server` directory to your cloud VM using SCP, SFTP, or Git:

```bash
# Example using SCP
scp -r server/ username@your-vm-ip:~/soundwatch/
```

### 2. Install Dependencies

SSH into your VM and install the required dependencies:

```bash
ssh username@your-vm-ip

# Navigate to server directory
cd ~/soundwatch/server

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Model

Download the sound classification model:

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Download the model
wget -O models/example_model.hdf5 https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1
```

### 4. Running the Server

Run the server with the following command:

```bash
python server.py --host 0.0.0.0 --port 8080
```

For production deployment, you may want to use a process manager like Supervisor or PM2 to keep the server running:

```bash
# Example Supervisor configuration (/etc/supervisor/conf.d/soundwatch.conf)
[program:soundwatch]
command=/home/username/soundwatch/server/venv/bin/python /home/username/soundwatch/server/server.py --host 0.0.0.0 --port 8080
directory=/home/username/soundwatch/server
user=username
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
```

### 5. Configure Firewall

Make sure port 8080 is open in your VM's firewall:

```bash
# For Ubuntu with UFW
sudo ufw allow 8080/tcp

# For firewalld
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

### 6. Test the Server

Verify the server is accessible by opening a web browser and navigating to:

```
http://your-vm-ip:8080
```

You should see the SoundWatch server web interface.

## Updating Client Applications

Make sure your client applications (phone and watch) are configured to connect to your server's IP address. Update the following constants in your client code if needed:

```java
public static final String DEFAULT_SERVER = "http://your-vm-ip:8080";
public static final String WS_SERVER_URL = "ws://your-vm-ip:8080";
```

## Troubleshooting

1. **Server not accessible**: Check that the firewall on your VM allows incoming connections on port 8080.
2. **Connection refused**: Ensure the server is running and bound to 0.0.0.0 (not localhost).
3. **Model loading errors**: Verify the model file is downloaded correctly to the models directory.
4. **Dependencies issues**: Confirm all requirements are installed with `pip list`.

## Production Considerations

For production deployment, consider:

1. Using HTTPS with a proper SSL certificate
2. Setting up a reverse proxy like Nginx
3. Implementing proper authentication
4. Setting up log rotation
5. Monitoring server health 