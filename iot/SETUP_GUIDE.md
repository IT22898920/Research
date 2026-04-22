# ESP32 GPS Device - Setup Guide

## Step 1: Install Arduino IDE
1. Download Arduino IDE from https://www.arduino.cc/en/software
2. Install and open it

## Step 2: Add ESP32 Board Support
1. Open Arduino IDE → **File** → **Preferences**
2. In "Additional Board Manager URLs", paste:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Click OK
4. Go to **Tools** → **Board** → **Boards Manager**
5. Search "ESP32"
6. Install **"esp32 by Espressif Systems"** (latest version)
7. Wait for download (~200MB)

## Step 3: Install Libraries
Go to **Sketch** → **Include Library** → **Manage Libraries** and install:

| Library | Author | Search Term |
|---------|--------|-------------|
| TinyGPSPlus | Mikal Hart | "TinyGPSPlus" |
| Adafruit SSD1306 | Adafruit | "Adafruit SSD1306" |
| Adafruit GFX Library | Adafruit | "Adafruit GFX" |
| ArduinoJson | Benoit Blanchon | "ArduinoJson" |

> When installing Adafruit SSD1306, it will ask to install dependencies - click "Install All"

## Step 4: Connect ESP32
1. Connect ESP32 to computer with USB cable
2. Go to **Tools** → **Board** → **esp32** → **ESP32 Dev Module**
3. Go to **Tools** → **Port** → Select the COM port (e.g., COM3, COM4)
   - If no port appears, install the USB driver:
     - CP2102: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
     - CH340: https://sparks.gogo.co.nz/ch340.html
4. Set these settings in **Tools** menu:
   - Board: "ESP32 Dev Module"
   - Upload Speed: "921600"
   - CPU Frequency: "240MHz"
   - Flash Frequency: "80MHz"
   - Flash Mode: "QIO"
   - Flash Size: "4MB"
   - Partition Scheme: "Default 4MB with spiffs"

## Step 5: Test ESP32 (Hello World)
1. Go to **File** → **Examples** → **01.Basics** → **Blink**
2. Click Upload (→ arrow button)
3. Wait for "Done uploading"
4. The blue LED on ESP32 should blink!

If upload fails:
- Hold the **BOOT** button on ESP32
- Click Upload
- Release BOOT button when you see "Connecting..."

## Step 6: Configure the Device
1. Open the firmware folder:
   `iot/firmware/coconut_gps_device/coconut_gps_device.ino`
2. Open `config.h` and edit:
   ```cpp
   #define WIFI_SSID     "YourPhoneHotspot"    // Your hotspot name
   #define WIFI_PASSWORD "YourPassword123"      // Your hotspot password
   #define DEVICE_ID     "ESP32_001"            // From mobile app
   #define DEVICE_KEY    "your_key_here"        // From mobile app
   #define API_HOST      "192.168.43.1"         // Phone's IP on hotspot
   ```

## Step 7: Upload Firmware
1. Open `coconut_gps_device.ino` in Arduino IDE
2. Click **Verify** (checkmark) to compile — should say "Done compiling"
3. Click **Upload** (arrow) to flash to ESP32
4. Open **Serial Monitor** (magnifying glass icon, top-right)
5. Set baud rate to **115200**
6. You should see: "Coconut GPS Device Starting..."

## Step 8: Register Device in Mobile App
1. Open the Coconut Health Monitor app
2. Go to Dashboard → IoT Devices
3. Tap "+ Add" to register a new device
4. Enter the Device ID (same as config.h, e.g., "ESP32_001")
5. Select a plantation
6. Save the Device Key — copy it to config.h
7. Re-upload firmware with the new device key

## Step 9: Field Test
1. Turn on phone hotspot
2. Power on ESP32 (USB or battery)
3. Wait for WiFi connection (OLED shows "W:OK")
4. Go outside for GPS fix (OLED shows satellites count)
5. Wait until status shows "READY" (6+ satellites, HDOP < 2.5)
6. Press button → device averages 30 GPS readings
7. Location is sent to backend and appears in the app!

## Finding Your Phone's Hotspot IP
When ESP32 connects to phone hotspot, it needs the phone's IP:
- Most Android phones: `192.168.43.1` (default)
- Some phones: Check in hotspot settings or WiFi settings
- Or run backend on the phone itself (advanced)

If backend is on your PC and you're using PC hotspot instead:
1. On PC, run: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Find the hotspot adapter IP
3. Update `API_HOST` in config.h
