/**
 * Coconut GPS IoT Device - ESP32 Firmware
 * =========================================
 * ESP32 + NEO-M8N GPS + OLED Display + Button
 *
 * Features:
 * - GPS location capture with averaging for accuracy
 * - OLED display showing real-time GPS data
 * - Button press to save location to backend
 * - WiFi connection via phone hotspot
 * - HTTP POST to Node.js backend API
 *
 * Wiring:
 * NEO-M8N GPS:  VCC→3.3V, GND→GND, TX→GPIO16(RX2), RX→GPIO17(TX2)
 * OLED (I2C):   VCC→3.3V, GND→GND, SDA→GPIO21, SCL→GPIO22
 * Button:       One leg→GPIO4, Other leg→GND
 *
 * Libraries needed (install via Arduino Library Manager):
 * 1. TinyGPSPlus by Mikal Hart
 * 2. Adafruit SSD1306
 * 3. Adafruit GFX Library
 * 4. ArduinoJson by Benoit Blanchon
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include <TinyGPSPlus.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>
#include "config.h"

// ============================================
// Objects
// ============================================
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);  // UART2
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);

// ============================================
// State Variables
// ============================================
enum DeviceState {
  STATE_BOOT,
  STATE_WIFI_CONNECTING,
  STATE_GPS_SEARCHING,
  STATE_GPS_FIXING,
  STATE_READY,
  STATE_AVERAGING,
  STATE_SENDING,
  STATE_SUCCESS,
  STATE_ERROR
};

DeviceState currentState = STATE_BOOT;

// GPS data
double currentLat = 0;
double currentLng = 0;
double currentAlt = 0;
float currentHdop = 99.9;
int currentSatellites = 0;
float currentAccuracy = 99.9;  // estimated meters

// Averaging
double avgLat = 0;
double avgLng = 0;
int avgCount = 0;
bool isAveraging = false;
unsigned long lastAvgReading = 0;

// Button
volatile bool buttonPressed = false;
unsigned long lastButtonPress = 0;

// Status
bool wifiConnected = false;
bool serverFound = false;
String apiHost = "";
int totalSaved = 0;
String lastStatus = "";

// ============================================
// Button Interrupt
// ============================================
void IRAM_ATTR buttonISR() {
  unsigned long now = millis();
  if (now - lastButtonPress > BUTTON_DEBOUNCE_MS) {
    buttonPressed = true;
    lastButtonPress = now;
  }
}

// ============================================
// Setup
// ============================================
void setup() {
  Serial.begin(115200);
  Serial.println("\n🌴 Coconut GPS Device Starting...");

  // LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Button with internal pull-up
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

  // Initialize OLED
  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("OLED init failed!");
  } else {
    showBootScreen();
  }

  // Initialize GPS
  gpsSerial.begin(GPS_BAUD_RATE, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("GPS initialized on UART2");

  // Connect WiFi
  connectWiFi();
}

// ============================================
// Main Loop
// ============================================
void loop() {
  // Read GPS data
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Update GPS values
  if (gps.location.isValid()) {
    currentLat = gps.location.lat();
    currentLng = gps.location.lng();
    currentSatellites = gps.satellites.value();
    currentHdop = gps.hdop.hdop();
    currentAccuracy = currentHdop * 2.5;  // rough HDOP to meters

    if (gps.altitude.isValid()) {
      currentAlt = gps.altitude.meters();
    }

    // Determine state
    if (currentSatellites < MIN_SATELLITES || currentHdop > MAX_HDOP) {
      currentState = STATE_GPS_FIXING;
    } else if (!isAveraging) {
      currentState = STATE_READY;
    }
  } else {
    currentState = STATE_GPS_SEARCHING;
  }

  // Handle averaging
  if (isAveraging) {
    currentState = STATE_AVERAGING;
    collectAverageReading();
  }

  // Handle button press
  if (buttonPressed) {
    buttonPressed = false;
    handleButtonPress();
  }

  // Check WiFi
  if (WiFi.status() != WL_CONNECTED) {
    wifiConnected = false;
  }

  // Update display
  updateDisplay();

  // Blink LED based on state
  blinkLED();

  delay(100);
}

// ============================================
// WiFi Connection
// ============================================
void connectWiFi() {
  currentState = STATE_WIFI_CONNECTING;
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Connecting WiFi...");
  display.println(WIFI_SSID);
  display.display();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    display.print(".");
    display.display();
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println("\nWiFi Connected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());

    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("WiFi Connected!");
    display.print("IP: ");
    display.println(WiFi.localIP());
    display.println("\nFinding server...");
    display.display();

    // Auto-discover backend server
    discoverServer();
  } else {
    Serial.println("\nWiFi Failed! Will retry...");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("WiFi FAILED!");
    display.println("Check hotspot");
    display.println("and config.h");
    display.display();
    delay(3000);
  }
}

// ============================================
// Auto-discover backend server via UDP broadcast
// ============================================
void discoverServer() {
  Serial.println("Discovering backend server...");

  WiFiUDP udp;
  udp.begin(4210);  // any local port

  // Send broadcast
  IPAddress broadcastIP = WiFi.localIP();
  broadcastIP[3] = 255;  // make broadcast (x.x.x.255)

  for (int attempt = 0; attempt < 10; attempt++) {
    udp.beginPacket(broadcastIP, DISCOVER_PORT);
    udp.print(DISCOVER_MSG);
    udp.endPacket();

    Serial.printf("  Discovery attempt %d/10 (broadcast: %s)\n",
                  attempt + 1, broadcastIP.toString().c_str());

    // Wait for response
    unsigned long start = millis();
    while (millis() - start < 1000) {
      int packetSize = udp.parsePacket();
      if (packetSize > 0) {
        char buf[64];
        int len = udp.read(buf, sizeof(buf) - 1);
        buf[len] = 0;

        String response = String(buf);
        // Expected: "COCONUT_SERVER:10.44.73.192:5000"
        if (response.startsWith("COCONUT_SERVER:")) {
          response.remove(0, 15);  // remove prefix
          int colonPos = response.indexOf(':');
          if (colonPos > 0) {
            apiHost = response.substring(0, colonPos);
          } else {
            apiHost = response;
          }
          serverFound = true;

          Serial.printf("  SERVER FOUND: %s\n", apiHost.c_str());

          display.clearDisplay();
          display.setCursor(0, 0);
          display.println("Server found!");
          display.print("IP: ");
          display.println(apiHost);
          display.println("\nWaiting for GPS...");
          display.display();
          delay(1500);

          udp.stop();
          return;
        }
      }
      delay(50);
    }
  }

  Serial.println("  Server NOT found! Will retry later.");
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Server not found!");
  display.println("Backend running?");
  display.display();
  delay(2000);

  udp.stop();
}

// ============================================
// Button Handler
// ============================================
void handleButtonPress() {
  if (currentState == STATE_GPS_SEARCHING) {
    lastStatus = "No GPS fix!";
    return;
  }

  if (currentState == STATE_AVERAGING) {
    // Cancel averaging
    isAveraging = false;
    avgCount = 0;
    lastStatus = "Cancelled";
    return;
  }

  if (currentState == STATE_SENDING) {
    return;  // Already sending
  }

  // Start averaging GPS readings for better accuracy
  Serial.println("\n--- Starting GPS capture ---");
  Serial.printf("  Sats: %d, HDOP: %.1f\n", currentSatellites, currentHdop);
  isAveraging = true;
  avgLat = 0;
  avgLng = 0;
  avgCount = 0;
  lastAvgReading = 0;  // take first reading immediately
}

// ============================================
// GPS Averaging — 1 reading per second, HDOP filtered
// ============================================
void collectAverageReading() {
  if (!gps.location.isValid()) return;

  // Only take 1 reading per second
  if (millis() - lastAvgReading < GPS_INTERVAL_MS) return;
  lastAvgReading = millis();

  // Skip bad HDOP readings
  if (currentHdop > MAX_HDOP) {
    Serial.printf("  SKIP: hdop:%.1f > %.1f\n", currentHdop, MAX_HDOP);
    return;
  }

  if (currentSatellites >= MIN_SATELLITES) {
    avgLat += currentLat;
    avgLng += currentLng;
    avgCount++;

    Serial.printf("  Reading %d/%d: %.6f, %.6f (sats:%d, hdop:%.1f)\n",
                  avgCount, GPS_READINGS, currentLat, currentLng,
                  currentSatellites, currentHdop);

    // Flash LED
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);
  }

  if (avgCount >= GPS_READINGS) {
    double finalLat = avgLat / avgCount;
    double finalLng = avgLng / avgCount;
    float finalAccuracy = currentHdop * 2.0;

    isAveraging = false;

    Serial.printf("\n===== RESULT =====\n");
    Serial.printf("Averaged: %.6f, %.6f (~%.1fm)\n", finalLat, finalLng, finalAccuracy);
    Serial.printf("==================\n\n");

    sendLocation(finalLat, finalLng, finalAccuracy);
  }
}

// ============================================
// Send Location to Backend
// ============================================
void sendLocation(double lat, double lng, float accuracy) {
  currentState = STATE_SENDING;

  // Check WiFi
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected! Reconnecting...");
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) {
      currentState = STATE_ERROR;
      lastStatus = "No WiFi!";
      return;
    }
  }

  // Build JSON payload
  StaticJsonDocument<512> doc;
  doc["latitude"] = lat;
  doc["longitude"] = lng;
  doc["accuracy"] = accuracy;
  doc["satellites"] = currentSatellites;
  doc["hdop"] = currentHdop;
  doc["altitude"] = currentAlt;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Auto-discover server if not found yet
  if (!serverFound) {
    discoverServer();
    if (!serverFound) {
      currentState = STATE_ERROR;
      lastStatus = "No server!";
      return;
    }
  }

  // Build URL
  String url = "http://";
  url += apiHost;
  url += ":";
  url += API_PORT;
  url += API_ENDPOINT;

  Serial.print("Sending to: ");
  Serial.println(url);
  Serial.print("Payload: ");
  Serial.println(jsonPayload);

  // HTTP POST
  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-Id", DEVICE_ID);
  http.addHeader("X-Device-Key", DEVICE_KEY);

  int httpCode = http.POST(jsonPayload);
  String response = http.getString();

  Serial.printf("Response code: %d\n", httpCode);
  Serial.println("Response: " + response);

  if (httpCode == 200 || httpCode == 201) {
    currentState = STATE_SUCCESS;
    totalSaved++;

    // Parse response for tree label
    StaticJsonDocument<512> resDoc;
    deserializeJson(resDoc, response);
    const char* label = resDoc["data"]["label"] | "saved";

    lastStatus = String("OK! ") + label;
    Serial.println("Location saved successfully!");

    // Flash LED
    for (int i = 0; i < 5; i++) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  } else {
    currentState = STATE_ERROR;

    // Parse error message
    StaticJsonDocument<256> errDoc;
    deserializeJson(errDoc, response);
    const char* errMsg = errDoc["message"] | "Send failed";

    lastStatus = String("ERR: ") + errMsg;
    Serial.println("Send failed: " + String(errMsg));
  }

  http.end();

  // Return to ready state after delay
  delay(2000);
  if (gps.location.isValid() && currentSatellites >= MIN_SATELLITES) {
    currentState = STATE_READY;
  }
}

// ============================================
// OLED Display Update
// ============================================
void updateDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Top bar - status
  display.setTextSize(1);
  display.setCursor(0, 0);

  // WiFi + Server status
  display.print(serverFound ? "SRV" : (wifiConnected ? "W:OK" : "W:--"));

  // Satellites
  display.setCursor(40, 0);
  display.printf("S:%d", currentSatellites);

  // HDOP
  display.setCursor(72, 0);
  display.printf("H:%.1f", currentHdop);

  // Saved count
  display.setCursor(105, 0);
  display.printf("#%d", totalSaved);

  // Separator line
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  switch (currentState) {
    case STATE_BOOT:
    case STATE_WIFI_CONNECTING:
      display.setTextSize(1);
      display.setCursor(10, 20);
      display.println("Connecting WiFi...");
      display.setCursor(10, 35);
      display.println(WIFI_SSID);
      break;

    case STATE_GPS_SEARCHING:
      display.setTextSize(1);
      display.setCursor(10, 15);
      display.println("Searching GPS...");
      display.setCursor(10, 30);
      display.println("Go outside!");
      display.setCursor(10, 45);
      display.println("Clear sky needed");
      break;

    case STATE_GPS_FIXING:
      display.setTextSize(1);
      display.setCursor(0, 13);
      display.printf("Lat: %.6f", currentLat);
      display.setCursor(0, 23);
      display.printf("Lng: %.6f", currentLng);
      display.setCursor(0, 35);
      display.printf("Fixing... S:%d H:%.1f", currentSatellites, currentHdop);
      display.setCursor(0, 47);
      display.printf("Need: S>=%d H<=%.1f", MIN_SATELLITES, MAX_HDOP);
      display.setCursor(0, 57);
      display.print("Wait for better fix");
      break;

    case STATE_READY:
      display.setTextSize(1);
      display.setCursor(0, 13);
      display.printf("Lat: %.6f", currentLat);
      display.setCursor(0, 23);
      display.printf("Lng: %.6f", currentLng);
      display.setCursor(0, 35);
      display.printf("Acc: ~%.1fm", currentAccuracy);
      display.setCursor(72, 35);
      display.printf("Alt:%.0f", currentAlt);

      // Ready indicator
      display.setTextSize(1);
      display.setCursor(0, 50);
      display.println("[BTN] Save Location");

      if (lastStatus.length() > 0) {
        display.setCursor(0, 57);
        display.print(lastStatus.substring(0, 21));
      }
      break;

    case STATE_AVERAGING: {
      display.setTextSize(1);
      display.setCursor(0, 13);
      display.printf("Lat: %.6f", currentLat);
      display.setCursor(0, 23);
      display.printf("Lng: %.6f", currentLng);
      display.setCursor(0, 38);
      display.printf("Averaging: %d/%d", avgCount, GPS_READINGS);

      // Progress bar
      int barWidth = map(avgCount, 0, GPS_READINGS, 0, 120);
      display.drawRect(4, 50, 120, 10, SSD1306_WHITE);
      display.fillRect(4, 50, barWidth, 10, SSD1306_WHITE);
      break;
    }

    case STATE_SENDING:
      display.setTextSize(1);
      display.setCursor(20, 25);
      display.println("Sending...");
      display.setCursor(15, 40);
      display.println("Please wait");
      break;

    case STATE_SUCCESS:
      display.setTextSize(1);
      display.setCursor(20, 15);
      display.println("SAVED!");
      display.setCursor(0, 30);
      display.printf("Lat: %.6f", currentLat);
      display.setCursor(0, 40);
      display.printf("Lng: %.6f", currentLng);
      display.setCursor(0, 55);
      display.print(lastStatus.substring(0, 21));
      break;

    case STATE_ERROR:
      display.setTextSize(1);
      display.setCursor(20, 20);
      display.println("ERROR!");
      display.setCursor(0, 35);
      display.print(lastStatus.substring(0, 21));
      display.setCursor(0, 50);
      display.println("[BTN] Try again");
      break;
  }

  display.display();
}

// ============================================
// LED Blink Pattern
// ============================================
void blinkLED() {
  static unsigned long lastBlink = 0;
  static bool ledState = false;

  unsigned long interval;
  switch (currentState) {
    case STATE_WIFI_CONNECTING: interval = 100; break;  // Fast blink
    case STATE_GPS_SEARCHING:   interval = 500; break;  // Slow blink
    case STATE_GPS_FIXING:      interval = 250; break;  // Medium blink
    case STATE_READY:           interval = 2000; break; // Slow pulse
    case STATE_AVERAGING:       interval = 150; break;  // Fast pulse
    case STATE_SENDING:         interval = 50; break;   // Very fast
    default:                    interval = 1000; break;
  }

  if (millis() - lastBlink >= interval) {
    lastBlink = millis();
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }
}

// ============================================
// Boot Screen
// ============================================
void showBootScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(15, 5);
  display.println("COCONUT GPS");
  display.setCursor(20, 18);
  display.println("IoT Device");

  display.drawLine(0, 30, 128, 30, SSD1306_WHITE);

  display.setCursor(10, 35);
  display.println("Health Monitor");
  display.setCursor(10, 48);
  display.printf("ID: %s", DEVICE_ID);

  display.display();
  delay(2000);
}
