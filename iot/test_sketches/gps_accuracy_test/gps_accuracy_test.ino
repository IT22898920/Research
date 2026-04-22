/**
 * GPS Accuracy Test with Averaging
 * =================================
 * Press button → takes 30 GPS readings → shows average on OLED
 * Compare averaged vs single readings to see improvement!
 *
 * Wiring: Same as gps_oled_test
 *   GPS: VCC→5V(VIN), GND→GND, TX→GPIO16, RX→GPIO17
 *   OLED: VCC→3.3V, GND→GND, SDA→GPIO21, SCL→GPIO22
 *   Button: one leg→GPIO4, other leg→GND
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

// Button
#define BUTTON_PIN 4
#define LED_PIN 2

// Averaging config
#define NUM_READINGS 30         // back to 30 — less drift!
#define MIN_SATS 4              // minimum satellites
#define MAX_HDOP 2.0            // reject readings with HDOP above this

// State
volatile bool buttonPressed = false;
unsigned long lastDebounce = 0;
bool isAveraging = false;
int readingCount = 0;
unsigned long lastReadingTime = 0;  // timer for readings

// GPS values
double currentLat = 0, currentLng = 0;
float currentHdop = 99.9;
int currentSats = 0;

// Averaging arrays
double latReadings[NUM_READINGS];
double lngReadings[NUM_READINGS];
float hdopReadings[NUM_READINGS];

// Results
double avgLat = 0, avgLng = 0;
float avgHdop = 0;
int savedCount = 0;

// Saved results for comparison (max 10)
#define MAX_SAVES 10
double savedLats[MAX_SAVES];
double savedLngs[MAX_SAVES];
int saveIndex = 0;

void IRAM_ATTR onButton() {
  if (millis() - lastDebounce > 300) {
    buttonPressed = true;
    lastDebounce = millis();
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), onButton, FALLING);

  Wire.begin(21, 22);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);

  Serial.println();
  Serial.println("=================================");
  Serial.println("  GPS ACCURACY TEST");
  Serial.println("  Press button to capture " + String(NUM_READINGS) + " readings");
  Serial.println("=================================");
  Serial.println();

  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(15, 5);
  display.println("GPS ACC");
  display.setCursor(25, 25);
  display.println("TEST");
  display.setTextSize(1);
  display.setCursor(5, 50);
  display.println("Waiting for GPS...");
  display.display();
  delay(1500);
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
    currentSats = gps.satellites.value();
    currentHdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.9;
  }

  // Handle button — longer debounce to prevent accidental cancel
  if (buttonPressed) {
    buttonPressed = false;
    if (!isAveraging) {
      // Start averaging — no GPS check needed, we'll wait for fix
      isAveraging = true;
      readingCount = 0;
      lastReadingTime = 0;
      Serial.println("\n--- Starting " + String(NUM_READINGS) + " reading capture ---");
      Serial.printf("  GPS Valid: %s, Sats: %d, HDOP: %.1f, Chars: %d\n",
                    gps.location.isValid() ? "YES" : "NO",
                    currentSats, currentHdop, gps.charsProcessed());
    } else if (readingCount > 0) {
      // Only allow cancel after at least 1 reading (prevent accidental cancel)
      isAveraging = false;
      readingCount = 0;
      Serial.println("Cancelled!");
    } else {
      Serial.println("Waiting for first reading... hold on!");
    }
  }

  // Debug — print status every 3 seconds while averaging with 0 readings
  static unsigned long lastDebugPrint = 0;
  if (isAveraging && readingCount == 0 && millis() - lastDebugPrint >= 3000) {
    lastDebugPrint = millis();
    Serial.printf("  [DEBUG] GPS valid: %s | Sats: %d | Chars: %d | Lat: %.6f | Lng: %.6f\n",
                  gps.location.isValid() ? "YES" : "NO",
                  gps.satellites.isValid() ? gps.satellites.value() : 0,
                  gps.charsProcessed(),
                  gps.location.lat(), gps.location.lng());
  }

  // Collect readings — every 1 second while averaging
  if (isAveraging && gps.location.isValid()
      && millis() - lastReadingTime >= 1000) {
    lastReadingTime = millis();
    // Re-read fresh values
    currentLat = gps.location.lat();
    currentLng = gps.location.lng();
    currentSats = gps.satellites.isValid() ? gps.satellites.value() : 0;
    currentHdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.9;

    // HDOP Filter — reject bad readings!
    if (currentHdop > MAX_HDOP) {
      Serial.printf("  SKIP: hdop:%.1f > %.1f (bad signal, waiting...)\n",
                    currentHdop, MAX_HDOP);
      return;  // skip this reading, try next second
    }

    latReadings[readingCount] = currentLat;
    lngReadings[readingCount] = currentLng;
    hdopReadings[readingCount] = currentHdop;
    readingCount++;

    Serial.printf("  Reading %2d/%d: %.6f, %.6f (sats:%d hdop:%.1f) OK\n",
                  readingCount, NUM_READINGS, currentLat, currentLng,
                  currentSats, currentHdop);

    // Flash LED
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);

    if (readingCount >= NUM_READINGS) {
      finishAveraging();
    }
  }

  // Update display
  updateDisplay();

  delay(100);
}

void finishAveraging() {
  isAveraging = false;

  // Calculate average
  avgLat = 0;
  avgLng = 0;
  avgHdop = 0;
  for (int i = 0; i < NUM_READINGS; i++) {
    avgLat += latReadings[i];
    avgLng += lngReadings[i];
    avgHdop += hdopReadings[i];
  }
  avgLat /= NUM_READINGS;
  avgLng /= NUM_READINGS;
  avgHdop /= NUM_READINGS;

  // Calculate spread (max deviation from average)
  double maxLatDev = 0, maxLngDev = 0;
  for (int i = 0; i < NUM_READINGS; i++) {
    double latDev = abs(latReadings[i] - avgLat);
    double lngDev = abs(lngReadings[i] - avgLng);
    if (latDev > maxLatDev) maxLatDev = latDev;
    if (lngDev > maxLngDev) maxLngDev = lngDev;
  }
  // Convert to meters (approximate at equator-ish latitudes)
  float latSpreadM = maxLatDev * 111000.0;
  float lngSpreadM = maxLngDev * 111000.0 * cos(avgLat * PI / 180.0);
  float totalSpread = sqrt(latSpreadM * latSpreadM + lngSpreadM * lngSpreadM);

  // Save result
  savedLats[saveIndex % MAX_SAVES] = avgLat;
  savedLngs[saveIndex % MAX_SAVES] = avgLng;
  saveIndex++;
  savedCount++;

  // Print results
  Serial.println();
  Serial.println("===== RESULT =====");
  Serial.printf("Averaged Location: %.6f, %.6f\n", avgLat, avgLng);
  Serial.printf("Avg HDOP: %.2f (~%.1fm accuracy)\n", avgHdop, avgHdop * 2.5);
  Serial.printf("Spread: %.1f m (max deviation from average)\n", totalSpread);
  Serial.printf("Readings used: %d\n", NUM_READINGS);
  Serial.printf("Save #%d\n", savedCount);

  // Compare with previous saves
  if (savedCount >= 2) {
    int prevIdx = (saveIndex - 2) % MAX_SAVES;
    double dLat = abs(avgLat - savedLats[prevIdx]) * 111000.0;
    double dLng = abs(avgLng - savedLngs[prevIdx]) * 111000.0 * cos(avgLat * PI / 180.0);
    float dist = sqrt(dLat * dLat + dLng * dLng);
    Serial.printf("Distance from previous save: %.1f m\n", dist);
  }
  Serial.println("==================");
  Serial.println();

  // Flash LED 5 times
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
}

void updateDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  // Top bar
  display.setCursor(0, 0);
  display.printf("SAT:%d  HDOP:%.1f  #%d",
                 currentSats, currentHdop, savedCount);
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  if (!gps.location.isValid()) {
    // No fix
    display.setCursor(10, 20);
    display.println("Searching GPS...");
    display.setCursor(10, 35);
    display.println("Go OUTSIDE!");
    display.setCursor(10, 50);
    display.printf("Chars: %d", gps.charsProcessed());

  } else if (isAveraging) {
    // Averaging in progress
    display.setCursor(0, 13);
    display.printf("Capturing: %d/%d", readingCount, NUM_READINGS);

    // Progress bar
    int barW = map(readingCount, 0, NUM_READINGS, 0, 120);
    display.drawRect(4, 25, 120, 12, SSD1306_WHITE);
    display.fillRect(4, 25, barW, 12, SSD1306_WHITE);

    display.setCursor(0, 42);
    display.printf("Lat: %.6f", currentLat);
    display.setCursor(0, 52);
    display.printf("Lng: %.6f", currentLng);

  } else if (savedCount > 0) {
    // Show last result
    display.setCursor(0, 13);
    display.printf("RESULT #%d:", savedCount);

    display.setCursor(0, 24);
    display.printf("Lat: %.6f", avgLat);
    display.setCursor(0, 34);
    display.printf("Lng: %.6f", avgLng);

    display.setCursor(0, 46);
    display.printf("Acc:~%.1fm", avgHdop * 2.5);

    // Bottom
    display.setCursor(0, 56);
    display.print("[BTN] Capture again");

  } else {
    // Ready - show live GPS
    display.setCursor(0, 13);
    display.printf("Lat: %.6f", currentLat);
    display.setCursor(0, 23);
    display.printf("Lng: %.6f", currentLng);
    display.setCursor(0, 35);
    display.printf("Acc: ~%.1fm", currentHdop * 2.5);
    display.setCursor(0, 47);
    display.printf("Sats: %d", currentSats);

    // Ready
    display.fillRect(0, 55, 128, 9, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(5, 56);
    display.print("[BTN] Start capture");
    display.setTextColor(SSD1306_WHITE);
  }

  display.display();
}
