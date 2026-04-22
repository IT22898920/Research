/**
 * Coconut GPS Live Tracker — Configuration
 * Edit these values before uploading!
 */

#ifndef CONFIG_H
#define CONFIG_H

// WiFi (Phone Hotspot)
#define WIFI_SSID     "YourPhoneHotspot"
#define WIFI_PASSWORD "YourPassword123"

// Backend API
#define API_HOST      "192.168.43.1"     // Phone hotspot IP
#define API_PORT      5000

// Device Auth (from mobile app registration)
#define DEVICE_ID     "ESP32_001"
#define DEVICE_KEY    "your_device_key_here"

// GPS
#define GPS_RX_PIN    16
#define GPS_TX_PIN    17
#define GPS_BAUD      9600

// OLED
#define OLED_SDA      21
#define OLED_SCL      22

// Button & LED
#define BUTTON_PIN    4
#define LED_PIN       2

// Live tracking interval (milliseconds)
#define LIVE_SEND_INTERVAL  3000   // send every 3 seconds

// Stable capture config
#define CAPTURE_READINGS    15
#define MAX_HDOP            1.5
#define STABILITY_WINDOW    10
#define STABILITY_THRESHOLD 3.0
#define MIN_STABLE_COUNT    5

#endif
