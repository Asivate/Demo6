# Deploying Sonarity Server on Google Cloud Platform

This guide will walk you through setting up the Sonarity server on a Google Cloud VM instance.

## 1. Creating a VM Instance

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "Compute Engine" > "VM instances"
3. Click "Create Instance"
4. Configure your VM with the following specifications:
   - **Machine type**: At least e2-standard-4 (4 vCPUs, 16 GB memory) recommended for model processing
   - **Boot disk**: Ubuntu 20.04 LTS or newer, at least 30GB of disk space
   - **Firewall**: Allow HTTP/HTTPS traffic
   - **Network tags**: Add "http-server" and "https-server"

## 2. Setting Up Firewall Rules

Create a firewall rule to allow connections to your server port:

1. Go to "VPC Network" > "Firewall"
2. Click "Create Firewall Rule"
3. Name: `sonarity-server`
4. Direction of traffic: Ingress
5. Targets: Specified target tags
6. Target tags: `http-server`
7. Source filter: IP ranges
8. Source IP ranges: `0.0.0.0/0` (or restrict as needed)
9. Protocols and ports: Select "Specified protocols and ports"
   - TCP: `8080` (or your custom port)
10. Click "Create"

## 3. Connect to Your VM

1. In the VM instances list, click the "SSH" button next to your instance
2. Wait for the connection to establish and the terminal to open

## 4. Install Dependencies

```bash
# Update package list
sudo apt update

# Install Python and related tools
sudo apt install -y python3-pip python3-venv git

# Install audio processing dependencies
sudo apt install -y libsndfile1-dev ffmpeg portaudio19-dev python3-pyaudio
```

## 5. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Asivate/Sonarity-server.git
cd Sonarity-server
```

## 6. Run the Server

```bash
# Make the startup script executable
chmod +x start_server.sh

# Start the server
./start_server.sh
```

The server will be available at `http://<your-vm-external-ip>:8080`

## 7. Setting Up as a Systemd Service (Optional)

To make the server run automatically on boot and restart if it crashes:

1. Create a systemd service file:

```bash
sudo nano /etc/systemd/system/sonarity.service
```

2. Add the following configuration (replace paths as needed):

```
[Unit]
Description=Sonarity Audio Analysis Server
After=network.target

[Service]
User=your-username
WorkingDirectory=/path/to/Sonarity-server
ExecStart=/path/to/Sonarity-server/start_server.sh
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=sonarity

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:

```bash
sudo systemctl enable sonarity
sudo systemctl start sonarity
```

4. Check the status:

```bash
sudo systemctl status sonarity
```

## 8. Using a Domain Name (Optional)

To use a custom domain with your server:

1. Configure your domain's DNS to point to your VM's external IP
2. Set up Nginx as a reverse proxy:

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/sonarity
```

3. Add the following configuration:

```
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

4. Enable the site and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/sonarity /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

## 9. Monitoring and Logs

- View logs: `sudo journalctl -u sonarity`
- Monitor the process: `ps aux | grep python`
- Check ports in use: `sudo netstat -tuln | grep 8080`

## Troubleshooting

- **Server won't start**: Check Python and dependencies installation
- **Cannot connect to server**: Verify firewall rules and that the server is running
- **Models not loading**: Check if the models directory exists and if there's enough disk space
- **High CPU/memory usage**: Consider upgrading your VM instance type 