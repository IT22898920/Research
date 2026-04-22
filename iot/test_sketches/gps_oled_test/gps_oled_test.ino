/**
 * GY-GPS6MV2 + OLED Display + ESP32 Combined Test
 * =================================================
 * Wiring:
 *   GPS VCC  → 3.3V
 *   GPS GND  → GND
 *   GPS TX   → GPIO 16 (RX2)
 *   GPS RX   → GPIO 17 (TX2)
 *
 *   OLED VCC → 3.3V
 *   OLED GND → GND
 *   OLED SDA → GPIO 21
 *   OLED SCL → GPIO 22
 *
 * Libraries:
 *   - TinyGPSPlus (by Mikal Hart)
 *   - Adafruit SSD1306
 *   - Adafruit GFX Library
 *
 * Open Serial Monitor at 115200 baud for debug output.
 * GO OUTSIDE for GPS fix!
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TinyGPSPlus.h>

// OLED
#define OLED_WIDTH  128
#define OLED_HEIGHT 64
#define OLED_SDA    21
#define OLED_SCL    22
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);

// GPS
#define GPS_RX_PIN  16
#define GPS_TX_PIN  17
#define GPS_BAUD    9600
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

// LED
#define LED_PIN 2

bool oledOK = false;
unsigned long lastUpdate = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  Serial.println();
  Serial.println("====================================");
  Serial.println("  GPS + OLED Combined Test");
  Serial.println("====================================");

  // Init OLED
  Wire.begin(OLED_SDA, OLED_SCL);
  if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    oledOK = true;
    Serial.println("[OK] OLED initialized");

    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(10, 5);
    display.println("COCONUT");
    display.setCursor(20, 28);
    display.println("GPS");
    display.setTextSize(1);
    display.setCursor(10, 52);
    display.println("Waiting for GPS...");
    display.display();
  } else {
    Serial.println("[FAIL] OLED not found! Check SDA/SCL wiring");
  }

  // Init GPS
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("[OK] GPS serial started on UART2");
  Serial.println();
  Serial.println("Go OUTSIDE for satellite fix!");
  Serial.println();

  delay(1500);
}

void loop() {
  // Read GPS data
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Update display and serial every 1.5 seconds
  if (millis() - lastUpdate >= 1500) {
    lastUpdate = millis();
    updateDisplay();
    printSerial();
  }

  // Blink LED to show device is alive
  static unsigned long lastBlink = 0;
  if (millis() - lastBlink >= (gps.location.isValid() ? 2000 : 500)) {
    lastBlink = millis();
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
}

void updateDisplay() {
  if (!oledOK) return;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // ---- Top bar ----
  display.setTextSize(1);
  display.setCursor(0, 0);

  // Satellites
  display.print("SAT:");
  if (gps.satellites.isValid()) {
    display.print(gps.satellites.value());
  } else {
    display.print("--");
  }

  // HDOP
  display.setCursor(50, 0);
  display.print("HDOP:");
  if (gps.hdop.isValid()) {
    display.print(gps.hdop.hdop(), 1);
  } else {
    display.print("--");
  }

  // Chars (data flow indicator)
  display.setCursor(100, 0);
  if (gps.charsProcessed() > 10) {
    display.print("DATA");
  } else {
    display.print("NO!!");
  }

  // Separator
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  // ---- Main content ----
  if (gps.charsProcessed() < 10) {
    // No data at all — wiring problem
    display.setTextSize(1);
    display.setCursor(5, 15);
    display.println("NO GPS DATA!");
    display.setCursor(5, 28);
    display.println("Check wiring:");
    display.setCursor(5, 40);
    display.println("GPS TX -> GPIO 16");
    display.setCursor(5, 50);
    display.println("GPS RX -> GPIO 17");

  } else if (!gps.location.isValid()) {
    // Data flowing but no fix yet
    display.setTextSize(1);
    display.setCursor(10, 15);
    display.println("Searching GPS...");

    // Animated dots
    int dots = (millis() / 500) % 4;
    display.setCursor(10, 28);
    for (int i = 0; i < dots; i++) display.print(". ");

    display.setCursor(5, 42);
    display.println("Go OUTSIDE!");
    display.setCursor(5, 53);
    display.print("Sats: ");
    display.print(gps.satellites.isValid() ? gps.satellites.value() : 0);
    display.print("  Wait...");

  } else {
    // GPS fix! Show location
    display.setTextSize(1);

    // Latitude
    display.setCursor(0, 13);
    display.print("LAT: ");
    display.setTextSize(1);
    display.print(gps.location.lat(), 6);

    // Longitude
    display.setCursor(0, 24);
    display.print("LNG: ");
    display.print(gps.location.lng(), 6);

    // Accuracy
    display.setCursor(0, 36);
    display.print("Acc: ~");
    if (gps.hdop.isValid()) {
      float acc = gps.hdop.hdop() * 2.5;
      display.print(acc, 1);
      display.print("m");
    } else {
      display.print("--");
    }

    // Altitude
    display.setCursor(70, 36);
    display.print("Alt:");
    if (gps.altitude.isValid()) {
      display.print(gps.altitude.meters(), 0);
      display.print("m");
    } else {
      display.print("--");
    }

    // Status bar at bottom
    display.fillRect(0, 49, 128, 15, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(5, 52);
    display.print("GPS FIXED! SAT:");
    display.print(gps.satellites.value());
    display.print(" OK!");
    display.setTextColor(SSD1306_WHITE);
  }

  display.display();
}

void printSerial() {
  Serial.println("--- GPS + OLED Status ---");
  Serial.print("OLED: ");
  Serial.println(oledOK ? "OK" : "FAILED");
  Serial.print("GPS chars: ");
  Serial.println(gps.charsProcessed());
  Serial.print("Satellites: ");
  Serial.println(gps.satellites.isValid() ? String(gps.satellites.value()) : "searching");

  if (gps.location.isValid()) {
    Serial.print("Lat: ");
    Serial.println(gps.location.lat(), 6);
    Serial.print("Lng: ");
    Serial.println(gps.location.lng(), 6);
    Serial.print("HDOP: ");
    Serial.println(gps.hdop.hdop(), 2);
    Serial.print("Accuracy: ~");
    Serial.print(gps.hdop.hdop() * 2.5, 1);
    Serial.println("m");
  } else {
    Serial.println("Location: NO FIX (go outside!)");
  }
  Serial.println();
}
