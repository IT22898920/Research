/**
 * Coconut GPS IoT Device - Configuration
 * ========================================
 * Edit these values before uploading to ESP32!
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================
// WiFi Configuration (Phone Hotspot)
// ============================================
#define WIFI_SSID     "LL"
#define WIFI_PASSWORD "87654321"

// ============================================
// Backend API Configuration
// ============================================
// Backend API — Cloud URL (Railway) + local fallback
#define CLOUD_HOST    "research-production-ed2e.up.railway.app"
#define CLOUD_URL     "https://research-production-ed2e.up.railway.app"
#define API_PORT      5000
#define API_ENDPOINT  "/api/iot/location"
#define DISCOVER_PORT 5001
#define DISCOVER_MSG  "COCONUT_DISCOVER"
#define USE_CLOUD     true    // true = cloud, false = local only

// ============================================
// Device Authentication
// (Get these from the mobile app after registering device)
// ============================================
#define DEVICE_ID     "ESP32_001"
#define DEVICE_KEY    "2de60b58e801fb571fbcd5818738d41e1901302d17f6fdf9365bab8b18d8d325"

// ============================================
// GPS Configuration
// ============================================
#define GPS_BAUD_RATE 9600          // NEO-M8N default baud rate
#define GPS_RX_PIN    16            // ESP32 RX2 <- GPS TX
#define GPS_TX_PIN    17            // ESP32 TX2 -> GPS RX
#define MIN_SATELLITES 4            // Minimum satellites for valid fix
#define MAX_HDOP      2.0           // Maximum HDOP for valid fix
#define GPS_READINGS  30            // Number of readings to average
#define GPS_INTERVAL_MS 1000        // Time between readings (ms)

// ============================================
// OLED Display Configuration (I2C)
// ============================================
#define OLED_WIDTH    128
#define OLED_HEIGHT   64
#define OLED_SDA      21            // ESP32 default I2C SDA
#define OLED_SCL      22            // ESP32 default I2C SCL
#define OLED_ADDRESS  0x3C          // Common I2C address for SSD1306

// ============================================
// Button Configuration
// ============================================
#define BUTTON_PIN    4             // GPIO for save button
#define BUTTON_DEBOUNCE_MS 300      // Debounce delay

// ============================================
// LED Configuration (built-in)
// ============================================
#define LED_PIN       2             // ESP32 built-in LED

// ============================================
// Battery Configuration (optional)
// ============================================
#define BATTERY_PIN   34            // ADC pin for battery voltage
#define BATTERY_LOW_VOLTAGE 3.3     // Low battery warning threshold

#endif // CONFIG_H
