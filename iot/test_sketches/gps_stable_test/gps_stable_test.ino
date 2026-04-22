/**
 * GPS Stable Capture Test — High Accuracy Mode
 * ==============================================
 * Strategy:
 * 1. Enable NEO-M8N Stationary Mode (internal Kalman filter optimized)
 * 2. Wait for GPS to STABILIZE (readings stop drifting)
 * 3. Only then capture 15 readings using MEDIAN (not average)
 * 4. Result: much more consistent coordinates!
 *
 * Wiring: Same as before
 *   GPS: VCC→5V(VIN), GND→GND, TX→GPIO16, RX→GPIO17
 *   OLED: VCC→3.3V, GND→GND, SDA→GPIO21, SCL→GPIO22
 *   Button: GPIO4 → GND
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TinyGPSPlus.h>

// OLED
Adafruit_SSD1306 display(128, 64, &Wire, -1);

// GPS
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

#define GPS_RX_PIN  16
#define GPS_TX_PIN  17
#define GPS_BAUD    9600
#define BUTTON_PIN  4
#define LED_PIN     2

// Capture config
#define CAPTURE_READINGS  15      // readings to capture (fewer = less drift)
#define MAX_HDOP          1.5     // strict HDOP filter
#define STABILITY_WINDOW  10      // last N readings to check stability
#define STABILITY_THRESHOLD 3.0   // meters — must be within this to be "stable"
#define MIN_STABLE_COUNT  5       // need N stable readings in a row

// State
volatile bool buttonPressed = false;
unsigned long lastDebounce = 0;
bool isCapturing = false;
bool isStable = false;
int captureCount = 0;
unsigned long lastReadingTime = 0;

// Current GPS
double currentLat = 0, currentLng = 0;
float currentHdop = 99.9;
int currentSats = 0;

// Stability tracking — circular buffer of recent positions
double recentLats[STABILITY_WINDOW];
double recentLngs[STABILITY_WINDOW];
int recentIndex = 0;
int recentCount = 0;
int stableCounter = 0;
float currentSpread = 99.9;

// Capture arrays
double capLats[CAPTURE_READINGS];
double capLngs[CAPTURE_READINGS];

// Results
double resultLat = 0, resultLng = 0;
int savedCount = 0;
#define MAX_SAVES 10
double savedLats[MAX_SAVES];
double savedLngs[MAX_SAVES];
int saveIndex = 0;

void IRAM_ATTR onButton() {
  if (millis() - lastDebounce > 500) {
    buttonPressed = true;
    lastDebounce = millis();
  }
}

// ============================================
// Send UBX command to GPS — enable Stationary Mode
// ============================================
void enableStationaryMode() {
  // UBX-CFG-NAV5: Set navigation mode to Stationary (2)
  // This tells GPS "I'm not moving" → much better filtering
  uint8_t cmd[] = {
    0xB5, 0x62,             // UBX header
    0x06, 0x24,             // CFG-NAV5
    0x24, 0x00,             // Length: 36 bytes
    0xFF, 0xFF,             // Mask: apply all
    0x02,                   // dynModel: 2 = Stationary ← KEY!
    0x03,                   // fixMode: auto 2D/3D
    0x00, 0x00, 0x00, 0x00, // fixedAlt
    0x10, 0x27, 0x00, 0x00, // fixedAltVar
    0x05,                   // minElev: 5 degrees
    0x00,                   // drLimit
    0xFA, 0x00,             // pDOP: 25.0
    0xFA, 0x00,             // tDOP: 25.0
    0x64, 0x00,             // pAcc: 100m
    0x2C, 0x01,             // tAcc: 300m
    0x00,                   // staticHoldThresh
    0x00,                   // dgnssTimeout
    0x00, 0x00, 0x00, 0x00, // cnoThreshNumSVs, cnoThresh, reserved
    0x00, 0x00,             // staticHoldMaxDist
    0x00,                   // utcStandard
    0x00, 0x00, 0x00, 0x00, 0x00, // reserved
    0x00, 0x00              // checksum placeholder
  };

  // Calculate checksum
  uint8_t ck_a = 0, ck_b = 0;
  for (int i = 2; i < sizeof(cmd) - 2; i++) {
    ck_a += cmd[i];
    ck_b += ck_a;
  }
  cmd[sizeof(cmd) - 2] = ck_a;
  cmd[sizeof(cmd) - 1] = ck_b;

  gpsSerial.write(cmd, sizeof(cmd));
  delay(250);
  Serial.println("[GPS] Stationary mode enabled (UBX-CFG-NAV5 dynModel=2)");
}

// ============================================
// Calculate distance between two GPS points (meters)
// ============================================
float distanceMeters(double lat1, double lng1, double lat2, double lng2) {
  double dLat = (lat2 - lat1) * 111000.0;
  double dLng = (lng2 - lng1) * 111000.0 * cos(lat1 * PI / 180.0);
  return sqrt(dLat * dLat + dLng * dLng);
}

// ============================================
// Calculate spread of recent readings
// ============================================
float calculateSpread() {
  if (recentCount < 3) return 99.9;

  // Find center
  double cLat = 0, cLng = 0;
  int count = min(recentCount, STABILITY_WINDOW);
  for (int i = 0; i < count; i++) {
    cLat += recentLats[i];
    cLng += recentLngs[i];
  }
  cLat /= count;
  cLng /= count;

  // Find max distance from center
  float maxDist = 0;
  for (int i = 0; i < count; i++) {
    float d = distanceMeters(cLat, cLng, recentLats[i], recentLngs[i]);
    if (d > maxDist) maxDist = d;
  }
  return maxDist;
}

// ============================================
// Median calculation (sort-based)
// ============================================
double medianDouble(double arr[], int n) {
  // Simple bubble sort (small array)
  double sorted[CAPTURE_READINGS];
  for (int i = 0; i < n; i++) sorted[i] = arr[i];

  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (sorted[j] > sorted[j + 1]) {
        double temp = sorted[j];
        sorted[j] = sorted[j + 1];
        sorted[j + 1] = temp;
      }
    }
  }

  if (n % 2 == 0) {
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
  }
  return sorted[n / 2];
}

// ============================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), onButton, FALLING);

  Wire.begin(21, 22);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  Serial.println();
  Serial.println("======================================");
  Serial.println("  GPS STABLE CAPTURE — High Accuracy");
  Serial.println("======================================");
  Serial.println("1. Wait for STABLE indicator");
  Serial.println("2. Press button to capture");
  Serial.println("3. Uses MEDIAN of 15 readings");
  Serial.println();

  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(5, 5);
  display.println("STABLE");
  display.setCursor(5, 25);
  display.println("GPS");
  display.setTextSize(1);
  display.setCursor(5, 50);
  display.println("Initializing...");
  display.display();

  // Wait a moment then enable stationary mode
  delay(2000);
  enableStationaryMode();
  delay(1000);
}

void loop() {
  // Read GPS
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Update current values
  if (gps.location.isValid()) {
    currentLat = gps.location.lat();
    currentLng = gps.location.lng();
    currentSats = gps.satellites.isValid() ? gps.satellites.value() : 0;
    currentHdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.9;
  }

  // Track stability — every 1 second
  static unsigned long lastStabilityCheck = 0;
  if (gps.location.isValid() && currentHdop <= MAX_HDOP
      && millis() - lastStabilityCheck >= 1000) {
    lastStabilityCheck = millis();

    recentLats[recentIndex] = currentLat;
    recentLngs[recentIndex] = currentLng;
    recentIndex = (recentIndex + 1) % STABILITY_WINDOW;
    if (recentCount < STABILITY_WINDOW) recentCount++;

    currentSpread = calculateSpread();

    if (currentSpread <= STABILITY_THRESHOLD && recentCount >= STABILITY_WINDOW) {
      stableCounter++;
      if (stableCounter >= MIN_STABLE_COUNT) {
        isStable = true;
      }
    } else {
      stableCounter = 0;
      isStable = false;
    }
  }

  // Handle button
  if (buttonPressed) {
    buttonPressed = false;
    if (!isCapturing && isStable) {
      isCapturing = true;
      captureCount = 0;
      lastReadingTime = 0;
      Serial.println("\n--- STABLE CAPTURE START ---");
      Serial.printf("  Stability: %.1fm spread, %d stable readings\n",
                    currentSpread, stableCounter);
    } else if (!isCapturing && gps.location.isValid() && !isStable) {
      Serial.printf("  NOT STABLE YET! Spread: %.1fm (need <%.1fm for %d readings)\n",
                    currentSpread, STABILITY_THRESHOLD, MIN_STABLE_COUNT);
      Serial.println("  Wait for STABLE indicator on OLED...");
    } else if (isCapturing && captureCount > 0) {
      isCapturing = false;
      captureCount = 0;
      Serial.println("  Cancelled!");
    }
  }

  // Capture readings
  if (isCapturing && gps.location.isValid() && currentHdop <= MAX_HDOP
      && millis() - lastReadingTime >= 1000) {
    lastReadingTime = millis();

    capLats[captureCount] = currentLat;
    capLngs[captureCount] = currentLng;
    captureCount++;

    Serial.printf("  Capture %2d/%d: %.6f, %.6f (sats:%d hdop:%.1f)\n",
                  captureCount, CAPTURE_READINGS, currentLat, currentLng,
                  currentSats, currentHdop);

    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);

    if (captureCount >= CAPTURE_READINGS) {
      finishCapture();
    }
  }

  // Skip message during capture
  if (isCapturing && gps.location.isValid() && currentHdop > MAX_HDOP
      && millis() - lastReadingTime >= 1000) {
    lastReadingTime = millis();
    Serial.printf("  SKIP: hdop:%.1f > %.1f\n", currentHdop, MAX_HDOP);
  }

  updateDisplay();

  // LED pattern
  static unsigned long lastBlink = 0;
  unsigned long blinkRate = isStable ? 2000 : (gps.location.isValid() ? 500 : 200);
  if (isCapturing) blinkRate = 100;
  if (millis() - lastBlink >= blinkRate) {
    lastBlink = millis();
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  delay(50);
}

void finishCapture() {
  isCapturing = false;

  // Use MEDIAN — much more robust than average!
  resultLat = medianDouble(capLats, CAPTURE_READINGS);
  resultLng = medianDouble(capLngs, CAPTURE_READINGS);

  // Calculate spread
  float maxDist = 0;
  for (int i = 0; i < CAPTURE_READINGS; i++) {
    float d = distanceMeters(resultLat, resultLng, capLats[i], capLngs[i]);
    if (d > maxDist) maxDist = d;
  }

  // Save
  savedLats[saveIndex % MAX_SAVES] = resultLat;
  savedLngs[saveIndex % MAX_SAVES] = resultLng;
  saveIndex++;
  savedCount++;

  Serial.println();
  Serial.println("===== STABLE RESULT =====");
  Serial.printf("MEDIAN Location: %.6f, %.6f\n", resultLat, resultLng);
  Serial.printf("Spread: %.1f m\n", maxDist);
  Serial.printf("Save #%d\n", savedCount);

  if (savedCount >= 2) {
    int prevIdx = (saveIndex - 2) % MAX_SAVES;
    float dist = distanceMeters(resultLat, resultLng,
                                savedLats[prevIdx], savedLngs[prevIdx]);
    Serial.printf("Distance from previous: %.1f m\n", dist);
  }
  Serial.println("=========================");
  Serial.println();

  // Flash LED
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED_PIN, HIGH); delay(200);
    digitalWrite(LED_PIN, LOW); delay(200);
  }

  // Reset stability tracking for next capture
  recentCount = 0;
  recentIndex = 0;
  stableCounter = 0;
  isStable = false;
}

void updateDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  // Top bar
  display.setCursor(0, 0);
  display.printf("S:%d H:%.1f #%d", currentSats, currentHdop, savedCount);

  // Stability indicator
  display.setCursor(90, 0);
  if (isStable) {
    display.print("STABLE");
  } else if (currentSpread < 99) {
    display.printf("%.0fm", currentSpread);
  } else {
    display.print("---");
  }

  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  if (!gps.location.isValid()) {
    display.setCursor(10, 20);
    display.println("Waiting for GPS...");
    display.setCursor(10, 35);
    display.println("Go outside!");

  } else if (isCapturing) {
    display.setCursor(0, 13);
    display.printf("CAPTURING: %d/%d", captureCount, CAPTURE_READINGS);
    int barW = map(captureCount, 0, CAPTURE_READINGS, 0, 120);
    display.drawRect(4, 25, 120, 12, SSD1306_WHITE);
    display.fillRect(4, 25, barW, 12, SSD1306_WHITE);
    display.setCursor(0, 42);
    display.printf("%.6f, %.6f", currentLat, currentLng);

  } else if (!isStable) {
    // Waiting for stability
    display.setCursor(0, 13);
    display.printf("Lat: %.6f", currentLat);
    display.setCursor(0, 23);
    display.printf("Lng: %.6f", currentLng);

    display.setCursor(0, 36);
    display.print("Stabilizing...");
    display.setCursor(0, 46);
    display.printf("Spread: %.1fm (need <%.0fm)", currentSpread, STABILITY_THRESHOLD);

    // Progress bar for stability
    int stabBar = map(min(stableCounter, MIN_STABLE_COUNT), 0, MIN_STABLE_COUNT, 0, 120);
    display.drawRect(4, 56, 120, 7, SSD1306_WHITE);
    display.fillRect(4, 56, stabBar, 7, SSD1306_WHITE);

  } else if (savedCount > 0 && !isCapturing) {
    // Show result + ready
    display.setCursor(0, 13);
    display.printf("RESULT #%d:", savedCount);
    display.setCursor(0, 24);
    display.printf("Lat: %.6f", resultLat);
    display.setCursor(0, 34);
    display.printf("Lng: %.6f", resultLng);

    display.fillRect(0, 48, 128, 16, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(5, 50);
    display.print("STABLE! [BTN] Capture");
    display.setTextColor(SSD1306_WHITE);

  } else {
    // Stable, ready to capture
    display.setCursor(0, 13);
    display.printf("Lat: %.6f", currentLat);
    display.setCursor(0, 23);
    display.printf("Lng: %.6f", currentLng);
    display.setCursor(0, 35);
    display.printf("Acc: ~%.1fm", currentHdop * 2.5);

    display.fillRect(0, 48, 128, 16, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(5, 50);
    display.print("STABLE! [BTN] Capture");
    display.setTextColor(SSD1306_WHITE);
  }

  display.display();
}
