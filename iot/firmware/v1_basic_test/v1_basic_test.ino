/**
 * V1 — Basic Test
 * ================
 * WiFi connect + GPS read + OLED display
 * NO backend, NO button, NO averaging
 * Just verify everything works!
 *
 * Wiring:
 *   GPS: VCC→5V(VIN), GND→GND, TX→GPIO16, RX→GPIO17
 *   OLED: VCC→3.3V, GND→GND, SDA→GPIO21, SCL→GPIO22
 */

#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TinyGPSPlus.h>

// CONFIG
const char* WIFI_SSID     = "LL";
const char* WIFI_PASSWORD = "87654321";

// Hardware
Adafruit_SSD1306 display(128, 64, &Wire, -1);
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

bool wifiOK = false;
bool oledOK = false;

void setup() {
  Serial.begin(115200);
  pinMode(2, OUTPUT);

  Serial.println("\n=== V1 BASIC TEST ===\n");

  // OLED
  Wire.begin(21, 22);
  oledOK = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  Serial.printf("[OLED] %s\n", oledOK ? "OK" : "FAILED");

  if (oledOK) {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(20, 10);
    display.println("V1");
    display.setCursor(10, 35);
    display.println("TEST");
    display.display();
    delay(1000);
  }

  // GPS
  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);
  Serial.println("[GPS] Serial started");

  // WiFi
  Serial.printf("[WiFi] Connecting to %s...\n", WIFI_SSID);
  showOLED("WiFi...", WIFI_SSID, "");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 20) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiOK = true;
    Serial.printf("\n[WiFi] OK! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] FAILED!");
  }

  Serial.println("\n--- STATUS ---");
  Serial.printf("OLED: %s\n", oledOK ? "OK" : "FAIL");
  Serial.printf("WiFi: %s\n", wifiOK ? "OK" : "FAIL");
  Serial.println("GPS: waiting for fix (go outside!)");
  Serial.println("--------------\n");
}

void loop() {
  // Read GPS
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Print & display every 2 seconds
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint >= 2000) {
    lastPrint = millis();

    int sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
    float hdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.9;
    long chars = gps.charsProcessed();

    Serial.printf("Sats:%d  HDOP:%.1f  Chars:%ld  WiFi:%s  ",
                  sats, hdop, chars, wifiOK ? "OK" : "NO");

    if (gps.location.isValid()) {
      double lat = gps.location.lat();
      double lng = gps.location.lng();
      Serial.printf("LAT:%.6f  LNG:%.6f\n", lat, lng);

      // OLED — show GPS data
      if (oledOK) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);

        display.setCursor(0, 0);
        display.printf("S:%d H:%.1f W:%s", sats, hdop, wifiOK ? "OK" : "--");
        display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

        display.setCursor(0, 14);
        display.printf("LAT: %.6f", lat);
        display.setCursor(0, 26);
        display.printf("LNG: %.6f", lng);
        display.setCursor(0, 38);
        display.printf("Acc: ~%.1fm", hdop * 2.5);

        display.fillRect(0, 50, 128, 14, SSD1306_WHITE);
        display.setTextColor(SSD1306_BLACK);
        display.setCursor(10, 52);
        display.print("GPS OK! V1 TEST");
        display.setTextColor(SSD1306_WHITE);
        display.display();
      }

      // Blink slow — GPS working
      digitalWrite(2, !digitalRead(2));

    } else {
      Serial.println("NO FIX");

      if (oledOK) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);

        display.setCursor(0, 0);
        display.printf("S:%d  W:%s  C:%ld", sats, wifiOK ? "OK" : "--", chars);
        display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

        display.setCursor(10, 20);
        display.println("Waiting GPS...");
        display.setCursor(10, 35);
        display.println("Go outside!");
        display.setCursor(10, 50);
        display.printf("Chars: %ld", chars);
        display.display();
      }
    }
  }

  delay(50);
}

void showOLED(const char* line1, const char* line2, const char* line3) {
  if (!oledOK) return;
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.println(line1);
  display.setCursor(0, 25);
  display.println(line2);
  display.setCursor(0, 40);
  display.println(line3);
  display.display();
}
