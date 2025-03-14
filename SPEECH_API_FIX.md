# Fixing Google Cloud Speech API Authentication Issues

This guide provides instructions for resolving the `ACCESS_TOKEN_SCOPE_INSUFFICIENT` error encountered with Google Cloud Speech API in SoundWatch.

## Problem

The error occurs because the service account does not have the proper permissions to access the Speech-to-Text API:

```
Error: 403 Request had insufficient authentication scopes.
[reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
domain: "googleapis.com"
metadata {
  key: "method"
  value: "google.cloud.speech.v1.Speech.Recognize"
}
metadata {
  key: "service"
  value: "speech.googleapis.com"
}
]
```

## Solution

### 1. Check Your Credentials Path

Ensure you're using the correct credentials file. Your system is currently looking for credentials at:
```
/home/hirwa0250/asivate-452914-5c12101797af.json
```

### 2. Update Service Account Permissions

To fix this issue, you need to update the permissions for your service account:

1. **Go to the Google Cloud Console**:
   - Visit https://console.cloud.google.com/
   - Make sure you're using the same project as specified in your credentials file

2. **Enable the Speech-to-Text API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Speech-to-Text API"
   - Click on it and ensure it's enabled (click "Enable" if it's not)

3. **Update Service Account Permissions**:
   - Navigate to "IAM & Admin" > "Service Accounts"
   - Find your service account (check the email in your credentials file)
   - Click the three dots menu (⋮) next to the service account
   - Click "Manage Access"
   - Click "GRANT ACCESS" and add these roles:
     - "Cloud Speech-to-Text User"
     - "Service Account Token Creator" (optional, but helpful)

4. **Alternative: Create a New Service Account**:
   If updating the existing service account doesn't work:
   
   - Navigate to "IAM & Admin" > "Service Accounts"
   - Click "CREATE SERVICE ACCOUNT"
   - Enter a name and description
   - Click "CREATE AND CONTINUE"
   - Add these roles:
     - "Cloud Speech-to-Text User"
     - "Service Account Token Creator" (optional)
   - Click "CONTINUE" and then "DONE"
   - Click on the new service account, go to the "KEYS" tab
   - Click "ADD KEY" > "Create new key"
   - Choose JSON format and click "CREATE"
   - A JSON file will be downloaded automatically
   - Copy this file to your VM at `/home/hirwa0250/asivate-452914-5c12101797af.json`

### 3. Check the Scope Configuration

Ensure your application is requesting the correct scopes:

- Required scope for Speech-to-Text API: `https://www.googleapis.com/auth/cloud-platform`
- This scope is typically added automatically when using the client libraries

### 4. Verify with the Check Script

Use the included `check_gcp_permissions.py` script to verify your credentials:

```bash
# Run from the server directory
python check_gcp_permissions.py
```

### 5. Run the Server with Proper Credentials

Use the updated start script to run the server with the correct credentials:

```bash
# Run from the server directory
bash start_with_speech_api.sh
```

## Troubleshooting

If you continue to experience issues:

1. Ensure the credentials file is in the correct location
2. Make sure the file permissions allow the server to read the file
3. Verify that the Speech-to-Text API is enabled in your project
4. Check that your project has billing enabled (required for using the Speech API)
5. Try creating a new credentials file through the Google Cloud Console 