/**
 * GY-GPS6MV2 Raw Data Test
 * ========================
 * No libraries needed! Just reads raw NMEA data from GPS.
 * This tells us if the GPS module is alive and sending data.
 *
 * Wiring:
 *   GPS VCC → 5V (VIN pin) or 3.3V
 *   GPS GND → GND
 *   GPS TX  → GPIO 16
 *   GPS RX  → GPIO 17
 *
 * Open Serial Monitor at 115200 baud.
 */

HardwareSerial gpsSerial(2);

void setup() {
  Serial.begin(115200);

  Serial.println();
  Serial.println("============================");
  Serial.println("  GPS RAW DATA TEST");
  Serial.println("============================");
  Serial.println();

  // Try default baud rate 9600
  Serial.println("Trying 9600 baud...");
  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);
  delay(2000);

  int chars = 0;
  unsigned long start = millis();
  while (millis() - start < 3000) {
    if (gpsSerial.available()) {
      chars++;
      Serial.write(gpsSerial.read());
    }
  }

  if (chars > 0) {
    Serial.println();
    Serial.println();
    Serial.print("[OK] GPS responding at 9600 baud! (");
    Serial.print(chars);
    Serial.println(" chars received)");
    Serial.println("You should see $GPRMC, $GPGGA lines above.");
    Serial.println();
    Serial.println("Streaming GPS data continuously...");
    Serial.println("================================");

    // Continue streaming
    while (true) {
      if (gpsSerial.available()) {
        Serial.write(gpsSerial.read());
      }
    }
  }

  // If 9600 didn't work, try other baud rates
  Serial.println();
  Serial.println("[FAIL] No data at 9600. Trying other baud rates...");
  Serial.println();

  int baudRates[] = {4800, 19200, 38400, 57600, 115200};

  for (int i = 0; i < 5; i++) {
    Serial.print("Trying ");
    Serial.print(baudRates[i]);
    Serial.print(" baud... ");

    gpsSerial.end();
    delay(100);
    gpsSerial.begin(baudRates[i], SERIAL_8N1, 16, 17);
    delay(1500);

    chars = 0;
    start = millis();
    while (millis() - start < 2000) {
      if (gpsSerial.available()) {
        char c = gpsSerial.read();
        if (c == '$' || (c >= 'A' && c <= 'Z') || c == ',' || (c >= '0' && c <= '9')) {
          chars++;
        }
      }
    }

    if (chars > 50) {
      Serial.print("FOUND! (");
      Serial.print(chars);
      Serial.println(" valid chars)");
      Serial.println();
      Serial.print("Your GPS uses ");
      Serial.print(baudRates[i]);
      Serial.println(" baud rate!");
      Serial.println("Update GPS_BAUD in your main code.");
      Serial.println();
      Serial.println("Streaming data:");
      Serial.println("================");

      while (true) {
        if (gpsSerial.available()) {
          Serial.write(gpsSerial.read());
        }
      }
    } else {
      Serial.println("no data");
    }
  }

  // Nothing worked
  Serial.println();
  Serial.println("================================");
  Serial.println("  NO GPS DATA ON ANY BAUD RATE!");
  Serial.println("================================");
  Serial.println();
  Serial.println("Possible problems:");
  Serial.println("1. VCC not connected - try 5V (VIN) instead of 3.3V");
  Serial.println("2. TX/RX swapped - swap yellow and green wires");
  Serial.println("3. Wrong pins - verify GPIO 16 and GPIO 17");
  Serial.println("4. Dead module - check for physical damage");
  Serial.println("5. Bad jumper wire - try different wires");
  Serial.println();
  Serial.println("Quick fix: Swap TX/RX wires and restart ESP32");
}

void loop() {
  // If we get here, no baud rate worked
  // Keep checking in case user swaps wires
  Serial.println("Rechecking... (swap TX/RX wires if not done)");

  gpsSerial.end();
  delay(100);
  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);

  unsigned long start = millis();
  while (millis() - start < 5000) {
    if (gpsSerial.available()) {
      Serial.println();
      Serial.println("[OK!] GPS DATA DETECTED! Streaming:");
      while (true) {
        if (gpsSerial.available()) {
          Serial.write(gpsSerial.read());
        }
      }
    }
  }
}
