/*
 * SoundWatch Fire Alarm Monitor for NodeMCU ESP8266
 * 
 * This program monitors a buzzer for fire alarm sounds and controls a light bulb.
 * When the buzzer is triggered (simulating a fire alarm), it sends an alert to the server,
 * which then commands the NodeMCU to turn off the light bulb (simulating turning off an appliance).
 * 
 * Hardware connections:
 * - Buzzer connected to D1
 * - Relay module for light bulb connected to D2
 * 
 * Server communication:
 * - Reports buzzer status to the server
 * - Receives commands from the server to control the light bulb
 */

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "RCA-OFFICE-Ghz";
const char* password = "RCA@2024";

// Server information
const char* serverUrl = "http://35.223.140.76:8080"; // Main server URL
const char* alertEndpoint = "/api/buzzer_alert";  // Endpoint to send buzzer alerts

// Pin definitions
const int BUZZER_PIN = D1;    // Buzzer input pin
const int LIGHT_PIN = D2;     // Relay control for light bulb

// Device identification
String deviceId = "nodemcu_fire_monitor";

// State variables
bool buzzerStatus = false;      // Current buzzer status (true = activated)
bool previousBuzzerStatus = false;  // Previous buzzer status for change detection
bool lightStatus = true;        // Light bulb status (true = on, false = off)
unsigned long lastStatusUpdate = 0;
const int STATUS_UPDATE_INTERVAL = 30000; // Status update every 30 seconds

// Variables for buzzer detection
const int BUZZER_THRESHOLD = 500;    // Analog threshold to detect buzzer activation
const int BUZZER_SAMPLES = 10;       // Number of samples to confirm buzzer activation
int buzzCounter = 0;                 // Counter for consecutive buzzer readings
const int DEBOUNCE_DELAY = 1000;     // Debounce delay in milliseconds
unsigned long lastDebounceTime = 0;  // Last time buzzer status changed

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Initialize pins
  pinMode(BUZZER_PIN, INPUT);
  pinMode(LIGHT_PIN, OUTPUT);
  
  // Turn on the light bulb by default
  digitalWrite(LIGHT_PIN, HIGH);
  lightStatus = true;
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Initial status update
  sendStatusUpdate();
}

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connection lost. Reconnecting...");
    WiFi.begin(ssid, password);
    delay(5000);
    return;
  }
  
  // Read buzzer status
  int buzzerValue = digitalRead(BUZZER_PIN);
  
  // Process buzzer reading with debounce
  if (buzzerValue == HIGH) {
    // Buzzer is active
    if (buzzCounter < BUZZER_SAMPLES) {
      buzzCounter++;
    }
  } else {
    // Buzzer is inactive
    if (buzzCounter > 0) {
      buzzCounter--;
    }
  }
  
  // Determine buzzer status based on counter
  buzzerStatus = (buzzCounter >= BUZZER_SAMPLES/2);
  
  // Check if buzzer status changed
  if (buzzerStatus != previousBuzzerStatus) {
    // Debounce
    if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
      lastDebounceTime = millis();
      previousBuzzerStatus = buzzerStatus;
      
      // Report buzzer status change to server
      sendStatusUpdate();
      
      // Print status change
      Serial.print("Buzzer status changed to: ");
      Serial.println(buzzerStatus ? "ACTIVE" : "INACTIVE");
      
      // If buzzer is activated, turn off the light immediately
      // (We'll also get a command from the server, but this is faster)
      if (buzzerStatus) {
        turnOffLight();
      }
    }
  }
  
  // Periodic status update
  if (millis() - lastStatusUpdate > STATUS_UPDATE_INTERVAL) {
    sendStatusUpdate();
  }
  
  // Check for commands from server
  checkServerCommands();
  
  delay(100); // Small delay to avoid excessive looping
}

// Send buzzer status update to server
void sendStatusUpdate() {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;
    
    // Construct the full URL
    String url = String(serverUrl) + String(alertEndpoint);
    
    // Start the HTTP request
    http.begin(client, url);
    http.addHeader("Content-Type", "application/json");
    
    // Create JSON payload
    StaticJsonDocument<200> doc;
    doc["device_id"] = deviceId;
    doc["buzzer_status"] = buzzerStatus;
    doc["light_status"] = lightStatus;
    doc["battery"] = 100;  // Placeholder for battery level
    
    String requestBody;
    serializeJson(doc, requestBody);
    
    // Send the request
    int httpResponseCode = http.POST(requestBody);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("HTTP Response code: " + String(httpResponseCode));
      Serial.println("Response: " + response);
      
      // Parse response to check for commands
      parseServerResponse(response);
    } else {
      Serial.println("Error sending request: " + String(httpResponseCode));
    }
    
    http.end();
    lastStatusUpdate = millis();
  }
}

// Check for commands from server
void checkServerCommands() {
  // This function could poll the server for commands
  // For now, we rely on commands returned from status updates
}

// Parse server response for commands
void parseServerResponse(String response) {
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, response);
  
  if (error) {
    Serial.print("JSON parsing failed: ");
    Serial.println(error.c_str());
    return;
  }
  
  // Check if there's a command
  if (doc.containsKey("command")) {
    String command = doc["command"].as<String>();
    Serial.print("Received command: ");
    Serial.println(command);
    
    if (command == "turn_off_light") {
      turnOffLight();
    } else if (command == "turn_on_light") {
      turnOnLight();
    }
  }
}

// Turn off the light bulb
void turnOffLight() {
  digitalWrite(LIGHT_PIN, LOW);
  lightStatus = false;
  Serial.println("Light turned OFF");
}

// Turn on the light bulb
void turnOnLight() {
  digitalWrite(LIGHT_PIN, HIGH);
  lightStatus = true;
  Serial.println("Light turned ON");
} 