/**
 * GY-GPS6MV2 + ESP32 Test
 * =======================
 * Wiring:
 *   GPS VCC → 3.3V
 *   GPS GND → GND
 *   GPS TX  → GPIO 16 (RX2)
 *   GPS RX  → GPIO 17 (TX2)
 *
 * Library: TinyGPSPlus by Mikal Hart
 *
 * Open Serial Monitor at 115200 baud to see output.
 * MUST go OUTSIDE for GPS fix — won't work indoors!
 */

#include <TinyGPSPlus.h>

TinyGPSPlus gps;
HardwareSerial gpsSerial(2);  // UART2

// Pins
#define GPS_RX_PIN 16   // ESP32 RX2 ← GPS TX
#define GPS_TX_PIN 17   // ESP32 TX2 → GPS RX
#define GPS_BAUD   9600 // GY-GPS6MV2 default baud

void setup() {
  Serial.begin(115200);
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  Serial.println();
  Serial.println("================================");
  Serial.println("  GY-GPS6MV2 + ESP32 Test");
  Serial.println("================================");
  Serial.println("Waiting for GPS data...");
  Serial.println("(Go OUTSIDE for satellite fix!)");
  Serial.println();
}

void loop() {
  // Read all available GPS data
  while (gpsSerial.available() > 0) {
    char c = gpsSerial.read();
    gps.encode(c);
  }

  // Print every 2 seconds
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint >= 2000) {
    lastPrint = millis();
    printGPSData();
  }

  // Warn if no data received at all
  if (millis() > 5000 && gps.charsProcessed() < 10) {
    Serial.println("** NO GPS DATA! Check wiring: **");
    Serial.println("   GPS TX -> GPIO 16");
    Serial.println("   GPS RX -> GPIO 17");
    Serial.println("   GPS VCC -> 3.3V");
    Serial.println("   GPS GND -> GND");
    Serial.println();
    delay(5000);
  }
}

void printGPSData() {
  Serial.println("--- GPS Status ---");

  // Characters processed (proves data is flowing)
  Serial.print("Chars processed: ");
  Serial.println(gps.charsProcessed());

  // Satellites
  Serial.print("Satellites: ");
  if (gps.satellites.isValid()) {
    Serial.println(gps.satellites.value());
  } else {
    Serial.println("searching...");
  }

  // Location
  Serial.print("Location: ");
  if (gps.location.isValid()) {
    Serial.print("Lat: ");
    Serial.print(gps.location.lat(), 6);
    Serial.print("  Lng: ");
    Serial.println(gps.location.lng(), 6);
  } else {
    Serial.println("NO FIX (go outside, wait 1-2 min)");
  }

  // Altitude
  Serial.print("Altitude: ");
  if (gps.altitude.isValid()) {
    Serial.print(gps.altitude.meters());
    Serial.println(" m");
  } else {
    Serial.println("--");
  }

  // HDOP (accuracy indicator - lower is better)
  Serial.print("HDOP: ");
  if (gps.hdop.isValid()) {
    Serial.print(gps.hdop.hdop());
    float accuracy = gps.hdop.hdop() * 2.5;
    Serial.print("  (~");
    Serial.print(accuracy, 1);
    Serial.println("m accuracy)");
  } else {
    Serial.println("--");
  }

  // Date & Time (UTC)
  Serial.print("Date/Time: ");
  if (gps.date.isValid() && gps.time.isValid()) {
    Serial.printf("%04d-%02d-%02d %02d:%02d:%02d UTC",
                  gps.date.year(), gps.date.month(), gps.date.day(),
                  gps.time.hour(), gps.time.minute(), gps.time.second());
    Serial.println();
  } else {
    Serial.println("--");
  }

  Serial.println();
}
