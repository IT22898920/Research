/**
 * Coconut GPS Live Tracker + Stable Tree Capture
 * ================================================
 * TWO modes:
 *   1. LIVE MODE: Sends GPS every 3 sec → app map shows real-time position
 *   2. CAPTURE MODE: Button press → stable capture → saves tree to backend
 *
 * Wiring:
 *   GPS: VCC→5V(VIN), GND→GND, TX→GPIO16, RX→GPIO17
 *   OLED: VCC→3.3V, GND→GND, SDA→GPIO21, SCL→GPIO22
 *   Button: GPIO4 → GND
 *
 * Libraries (Arduino Library Manager):
 *   1. TinyGPSPlus
 *   2. Adafruit SSD1306
 *   3. Adafruit GFX Library
 *   4. ArduinoJson
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TinyGPSPlus.h>
#include <ArduinoJson.h>

// ==============================================
// CONFIGURATION — EDIT THESE BEFORE UPLOADING!
// ==============================================

// WiFi (Android Phone Hotspot)
const char* WIFI_SSID     = "LL";
const char* WIFI_PASSWORD = "87654321";

// Backend API — laptop IP on hotspot network
const char* API_HOST  = "10.199.219.192";
const int   API_PORT  = 5000;

// Device Auth
const char* DEVICE_ID  = "ESP32_001";
const char* DEVICE_KEY = "2de60b58e801fb571fbcd5818738d41e1901302d17f6fdf9365bab8b18d8d325";

// Pins
#define GPS_RX_PIN    16
#define GPS_TX_PIN    17
#define GPS_BAUD      9600
#define OLED_SDA      21
#define OLED_SCL      22
#define BUTTON_PIN    4
#define LED_PIN       2

// Live tracking
#define LIVE_SEND_INTERVAL  3000    // send every 3 seconds

// Stable capture
#define CAPTURE_READINGS    15      // median of 15
#define MAX_HDOP            1.5     // reject bad readings
#define STABILITY_WINDOW    10      // check last 10 readings
#define STABILITY_THRESHOLD 3.0     // meters — must be within this
#define MIN_STABLE_COUNT    5       // need 5 stable in a row

// ==============================================
// END OF CONFIGURATION
// ==============================================

// Objects
Adafruit_SSD1306 display(128, 64, &Wire, -1);
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

// State
enum Mode { MODE_LIVE, MODE_STABILIZING, MODE_CAPTURING };
Mode currentMode = MODE_LIVE;

// Button
volatile bool buttonPressed = false;
unsigned long lastDebounce = 0;

// GPS
double currentLat = 0, currentLng = 0;
float currentHdop = 99.9;
int currentSats = 0;
float currentAlt = 0;
bool gpsValid = false;

// Live tracking
unsigned long lastLiveSend = 0;
bool wifiConnected = false;
int liveSendCount = 0;
bool liveEnabled = true;

// Stability tracking
double recentLats[STABILITY_WINDOW];
double recentLngs[STABILITY_WINDOW];
int recentIndex = 0, recentCount = 0;
int stableCounter = 0;
bool isStable = false;
float currentSpread = 99.9;

// Capture
double capLats[CAPTURE_READINGS];
double capLngs[CAPTURE_READINGS];
int captureCount = 0;
unsigned long lastCaptureTime = 0;

// Results
double resultLat = 0, resultLng = 0;
int savedTrees = 0;
String lastStatus = "";

// ============================================
// Button Interrupt
// ============================================
void IRAM_ATTR onButton() {
  if (millis() - lastDebounce > 500) {
    buttonPressed = true;
    lastDebounce = millis();
  }
}

// ============================================
// Enable GPS Stationary Mode (UBX command)
// Tells NEO-M8N "I'm not moving" → better accuracy
// ============================================
void enableStationaryMode() {
  uint8_t cmd[] = {
    0xB5, 0x62, 0x06, 0x24, 0x24, 0x00,
    0xFF, 0xFF, 0x02, 0x03,
    0x00, 0x00, 0x00, 0x00,
    0x10, 0x27, 0x00, 0x00,
    0x05, 0x00, 0xFA, 0x00, 0xFA, 0x00,
    0x64, 0x00, 0x2C, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00
  };
  uint8_t ck_a = 0, ck_b = 0;
  for (int i = 2; i < sizeof(cmd) - 2; i++) {
    ck_a += cmd[i]; ck_b += ck_a;
  }
  cmd[sizeof(cmd) - 2] = ck_a;
  cmd[sizeof(cmd) - 1] = ck_b;
  gpsSerial.write(cmd, sizeof(cmd));
  delay(250);
  Serial.println("[GPS] Stationary mode enabled");
}

// ============================================
// Distance between two GPS points (meters)
// ============================================
float distMeters(double lat1, double lng1, double lat2, double lng2) {
  double dLat = (lat2 - lat1) * 111000.0;
  double dLng = (lng2 - lng1) * 111000.0 * cos(lat1 * PI / 180.0);
  return sqrt(dLat * dLat + dLng * dLng);
}

// ============================================
// Median calculation
// ============================================
double medianOf(double arr[], int n) {
  double sorted[CAPTURE_READINGS];
  for (int i = 0; i < n; i++) sorted[i] = arr[i];
  for (int i = 0; i < n - 1; i++)
    for (int j = 0; j < n - i - 1; j++)
      if (sorted[j] > sorted[j + 1]) {
        double t = sorted[j]; sorted[j] = sorted[j + 1]; sorted[j + 1] = t;
      }
  return (n % 2) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
}

// ============================================
// WiFi Connection
// ============================================
void connectWiFi() {
  Serial.printf("Connecting to %s...\n", WIFI_SSID);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Connecting WiFi...");
  display.println(WIFI_SSID);
  display.display();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  wifiConnected = (WiFi.status() == WL_CONNECTED);
  if (wifiConnected) {
    Serial.println("\nWiFi Connected!");
    Serial.print("IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi FAILED! Check hotspot.");
  }
}

// ============================================
// Send Live Location (every 3 sec, lightweight)
// ============================================
void sendLiveLocation() {
  if (!wifiConnected || !gpsValid) return;
  if (WiFi.status() != WL_CONNECTED) { wifiConnected = false; return; }

  String url = String("http://") + API_HOST + ":" + API_PORT + "/api/iot/live-location";

  StaticJsonDocument<256> doc;
  doc["latitude"] = currentLat;
  doc["longitude"] = currentLng;
  doc["accuracy"] = currentHdop * 2.5;
  doc["satellites"] = currentSats;
  doc["hdop"] = currentHdop;
  doc["altitude"] = currentAlt;

  String json;
  serializeJson(doc, json);

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-Id", DEVICE_ID);
  http.addHeader("X-Device-Key", DEVICE_KEY);
  http.setTimeout(2000);

  int code = http.POST(json);
  http.end();

  if (code == 200) {
    liveSendCount++;
  } else {
    Serial.printf("[LIVE] Failed: %d\n", code);
  }
}

// ============================================
// Send Tree Capture (stable, save to database)
// ============================================
void sendTreeCapture(double lat, double lng, float accuracy) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    if (!wifiConnected) { lastStatus = "No WiFi!"; return; }
  }

  String url = String("http://") + API_HOST + ":" + API_PORT + "/api/iot/location";

  StaticJsonDocument<256> doc;
  doc["latitude"] = lat;
  doc["longitude"] = lng;
  doc["accuracy"] = accuracy;
  doc["satellites"] = currentSats;
  doc["hdop"] = currentHdop;
  doc["altitude"] = currentAlt;

  String json;
  serializeJson(doc, json);

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-Id", DEVICE_ID);
  http.addHeader("X-Device-Key", DEVICE_KEY);

  int code = http.POST(json);
  String response = http.getString();
  http.end();

  if (code == 200 || code == 201) {
    savedTrees++;
    StaticJsonDocument<256> res;
    deserializeJson(res, response);
    const char* label = res["data"]["label"] | "saved";
    lastStatus = String("OK: ") + label;
    Serial.printf("[TREE] Saved! %s\n", label);

    for (int i = 0; i < 5; i++) {
      digitalWrite(LED_PIN, HIGH); delay(100);
      digitalWrite(LED_PIN, LOW); delay(100);
    }
  } else {
    lastStatus = "Send failed!";
    Serial.printf("[TREE] Failed: %d\n", code);
  }
}

// ============================================
// Stability Check
// ============================================
void updateStability() {
  if (!gpsValid || currentHdop > MAX_HDOP) return;

  recentLats[recentIndex] = currentLat;
  recentLngs[recentIndex] = currentLng;
  recentIndex = (recentIndex + 1) % STABILITY_WINDOW;
  if (recentCount < STABILITY_WINDOW) recentCount++;

  if (recentCount >= 3) {
    double cLat = 0, cLng = 0;
    int count = min(recentCount, (int)STABILITY_WINDOW);
    for (int i = 0; i < count; i++) { cLat += recentLats[i]; cLng += recentLngs[i]; }
    cLat /= count; cLng /= count;

    float maxDist = 0;
    for (int i = 0; i < count; i++) {
      float d = distMeters(cLat, cLng, recentLats[i], recentLngs[i]);
      if (d > maxDist) maxDist = d;
    }
    currentSpread = maxDist;
  }

  if (currentSpread <= STABILITY_THRESHOLD && recentCount >= STABILITY_WINDOW) {
    stableCounter++;
    if (stableCounter >= MIN_STABLE_COUNT) isStable = true;
  } else {
    stableCounter = 0;
    isStable = false;
  }
}

// ============================================
// SETUP
// ============================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), onButton, FALLING);

  Wire.begin(OLED_SDA, OLED_SCL);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  Serial.println("\n========================================");
  Serial.println("  COCONUT GPS - Live Tracker + Capture");
  Serial.println("========================================\n");

  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 5);
  display.println("COCONUT");
  display.setCursor(25, 25);
  display.println("GPS");
  display.setTextSize(1);
  display.setCursor(15, 48);
  display.println("Live Tracker v1.0");
  display.display();
  delay(1500);

  connectWiFi();
  delay(1000);
  enableStationaryMode();
}

// ============================================
// MAIN LOOP
// ============================================
void loop() {
  // Read GPS
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  if (gps.location.isValid()) {
    currentLat = gps.location.lat();
    currentLng = gps.location.lng();
    currentSats = gps.satellites.isValid() ? gps.satellites.value() : 0;
    currentHdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.9;
    if (gps.altitude.isValid()) currentAlt = gps.altitude.meters();
    gpsValid = true;
  }

  // Stability check every 1 second
  static unsigned long lastStabCheck = 0;
  if (millis() - lastStabCheck >= 1000) {
    lastStabCheck = millis();
    updateStability();
  }

  // === LIVE MODE: Send location every 3 seconds ===
  if (currentMode == MODE_LIVE && liveEnabled) {
    if (millis() - lastLiveSend >= LIVE_SEND_INTERVAL) {
      lastLiveSend = millis();
      sendLiveLocation();
    }
  }

  // === Handle button press ===
  if (buttonPressed) {
    buttonPressed = false;

    if (currentMode == MODE_LIVE) {
      if (isStable) {
        currentMode = MODE_CAPTURING;
        captureCount = 0;
        lastCaptureTime = 0;
        liveEnabled = false;
        Serial.println("\n--- STABLE CAPTURE START ---");
      } else if (gpsValid) {
        currentMode = MODE_STABILIZING;
        Serial.println("Waiting for stability...");
      } else {
        Serial.println("No GPS fix!");
      }
    } else if (currentMode == MODE_STABILIZING) {
      if (isStable) {
        currentMode = MODE_CAPTURING;
        captureCount = 0;
        lastCaptureTime = 0;
        liveEnabled = false;
        Serial.println("\n--- STABLE CAPTURE START ---");
      }
    } else if (currentMode == MODE_CAPTURING && captureCount > 0) {
      currentMode = MODE_LIVE;
      liveEnabled = true;
      captureCount = 0;
      Serial.println("Cancelled!");
    }
  }

  // Auto-start capture when stable during stabilizing mode
  if (currentMode == MODE_STABILIZING && isStable) {
    currentMode = MODE_CAPTURING;
    captureCount = 0;
    lastCaptureTime = 0;
    liveEnabled = false;
    Serial.println("\n--- AUTO CAPTURE (stable detected) ---");
  }

  // === CAPTURE MODE: Collect readings ===
  if (currentMode == MODE_CAPTURING && gpsValid && currentHdop <= MAX_HDOP
      && millis() - lastCaptureTime >= 1000) {
    lastCaptureTime = millis();

    capLats[captureCount] = currentLat;
    capLngs[captureCount] = currentLng;
    captureCount++;

    Serial.printf("  Capture %2d/%d: %.6f, %.6f (sats:%d hdop:%.1f)\n",
                  captureCount, CAPTURE_READINGS, currentLat, currentLng,
                  currentSats, currentHdop);

    digitalWrite(LED_PIN, HIGH); delay(50); digitalWrite(LED_PIN, LOW);

    if (captureCount >= CAPTURE_READINGS) {
      resultLat = medianOf(capLats, CAPTURE_READINGS);
      resultLng = medianOf(capLngs, CAPTURE_READINGS);
      float accuracy = currentHdop * 2.0;

      Serial.printf("\nMEDIAN: %.6f, %.6f (~%.1fm)\n", resultLat, resultLng, accuracy);

      sendTreeCapture(resultLat, resultLng, accuracy);

      currentMode = MODE_LIVE;
      liveEnabled = true;
      recentCount = 0;
      recentIndex = 0;
      stableCounter = 0;
      isStable = false;
    }
  }

  // Skip bad HDOP during capture
  if (currentMode == MODE_CAPTURING && gpsValid && currentHdop > MAX_HDOP
      && millis() - lastCaptureTime >= 1000) {
    lastCaptureTime = millis();
    Serial.printf("  SKIP: hdop:%.1f\n", currentHdop);
  }

  updateDisplay();

  // LED blink pattern
  static unsigned long lastBlink = 0;
  unsigned long rate;
  if (currentMode == MODE_CAPTURING) rate = 100;
  else if (!gpsValid) rate = 200;
  else if (isStable) rate = 2000;
  else rate = 500;
  if (millis() - lastBlink >= rate) {
    lastBlink = millis();
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  delay(50);
}

// ============================================
// OLED Display
// ============================================
void updateDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  // Top bar
  display.setCursor(0, 0);
  display.printf("S:%d", currentSats);
  display.setCursor(30, 0);
  display.printf("H:%.1f", currentHdop);
  display.setCursor(65, 0);
  display.print(wifiConnected ? "W:OK" : "W:--");
  display.setCursor(95, 0);
  display.printf("T:%d", savedTrees);
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  if (!gpsValid) {
    display.setCursor(10, 20);
    display.println("Waiting GPS...");
    display.setCursor(10, 35);
    display.println("Go outside!");

  } else if (currentMode == MODE_CAPTURING) {
    display.setCursor(0, 13);
    display.printf("CAPTURE: %d/%d", captureCount, CAPTURE_READINGS);
    int barW = map(captureCount, 0, CAPTURE_READINGS, 0, 120);
    display.drawRect(4, 25, 120, 10, SSD1306_WHITE);
    display.fillRect(4, 25, barW, 10, SSD1306_WHITE);
    display.setCursor(0, 40);
    display.printf("%.6f", currentLat);
    display.setCursor(0, 50);
    display.printf("%.6f", currentLng);

  } else if (currentMode == MODE_STABILIZING) {
    display.setCursor(0, 13);
    display.printf("Lat: %.6f", currentLat);
    display.setCursor(0, 23);
    display.printf("Lng: %.6f", currentLng);
    display.setCursor(0, 36);
    display.printf("Stabilizing: %.1fm", currentSpread);
    int stabBar = map(min(stableCounter, (int)MIN_STABLE_COUNT), 0, MIN_STABLE_COUNT, 0, 120);
    display.drawRect(4, 48, 120, 8, SSD1306_WHITE);
    display.fillRect(4, 48, stabBar, 8, SSD1306_WHITE);
    display.setCursor(0, 57);
    display.print("Wait for stable...");

  } else {
    // LIVE MODE
    display.setCursor(0, 13);
    display.printf("Lat: %.6f", currentLat);
    display.setCursor(0, 23);
    display.printf("Lng: %.6f", currentLng);
    display.setCursor(0, 33);
    display.printf("Acc:~%.1fm", currentHdop * 2.5);
    display.setCursor(70, 33);
    display.printf("Tx:%d", liveSendCount);

    if (isStable) {
      display.fillRect(0, 44, 128, 20, SSD1306_WHITE);
      display.setTextColor(SSD1306_BLACK);
      display.setCursor(3, 46);
      display.print("LIVE + STABLE");
      display.setCursor(3, 55);
      display.print("[BTN] Save Tree");
      display.setTextColor(SSD1306_WHITE);
    } else {
      display.setCursor(0, 46);
      display.printf("LIVE  Spread:%.1fm", currentSpread);
      display.setCursor(0, 56);
      if (lastStatus.length() > 0) {
        display.print(lastStatus.substring(0, 21));
      } else {
        display.print("[BTN] Start capture");
      }
    }
  }

  display.display();
}
