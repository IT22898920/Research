# ESP32 GPS IoT Device - Wiring Guide

## Components
- ESP32 DevKit V1 (38-pin)
- NEO-M8N GPS Module (with antenna)
- 0.96" OLED Display (I2C, SSD1306)
- Tactile Push Button
- Breadboard + Jumper Wires

## Wiring Diagram

```
                    ESP32 DevKit V1 (38-pin)
                    ┌─────────────────────┐
                    │         USB         │
                    │    ┌───────────┐    │
                    │    │           │    │
              3.3V ─┤    │           │    ├─ GND
               GND ─┤    │           │    ├─ GPIO 23
         GPIO 15  ─┤    │           │    ├─ GPIO 22 ──→ OLED SCL
          GPIO 2  ─┤    │  ESP32    │    ├─ GPIO 21 ──→ OLED SDA
          GPIO 4  ─┤←── Button     │    ├─ GPIO 19
         GPIO 16  ─┤←── GPS TX     │    ├─ GPIO 18
         GPIO 17  ─┤──→ GPS RX     │    ├─ GPIO 5
          GPIO 5  ─┤    │           │    ├─ GPIO 17
         GPIO 18  ─┤    │           │    ├─ GPIO 16
         GPIO 19  ─┤    │           │    ├─ GPIO 4
              GND ─┤    │           │    ├─ GPIO 2 (LED)
         GPIO 21  ─┤    │           │    ├─ GPIO 15
              3V3 ─┤    │           │    ├─ GND
                    │    └───────────┘    │
                    └─────────────────────┘
```

## Connection Table

### NEO-M8N GPS Module → ESP32
| GPS Pin | ESP32 Pin | Wire Color (suggested) |
|---------|-----------|----------------------|
| VCC     | 3.3V      | Red                  |
| GND     | GND       | Black                |
| TX      | GPIO 16   | Yellow               |
| RX      | GPIO 17   | Green                |

### OLED Display (I2C) → ESP32
| OLED Pin | ESP32 Pin | Wire Color (suggested) |
|----------|-----------|----------------------|
| VCC      | 3.3V      | Red                  |
| GND      | GND       | Black                |
| SDA      | GPIO 21   | Blue                 |
| SCL      | GPIO 22   | White                |

### Push Button → ESP32
| Button Pin | ESP32 Pin | Note                    |
|------------|-----------|-------------------------|
| Leg 1      | GPIO 4    | Internal pull-up used   |
| Leg 2      | GND       | Any GND pin             |

## Step-by-Step Wiring Instructions

### Step 1: Place ESP32 on Breadboard
1. Place ESP32 on the breadboard, straddling the center divider
2. Make sure pins are firmly in the breadboard holes

### Step 2: Connect GPS Module (4 wires)
1. **Red wire**: GPS VCC → ESP32 3.3V pin
2. **Black wire**: GPS GND → ESP32 GND pin
3. **Yellow wire**: GPS TX → ESP32 GPIO 16 (this is RX2)
4. **Green wire**: GPS RX → ESP32 GPIO 17 (this is TX2)

> **IMPORTANT**: GPS TX goes to ESP32 RX (GPIO 16), GPS RX goes to ESP32 TX (GPIO 17)
> This is a cross-connection!

### Step 3: Connect OLED Display (4 wires)
1. **Red wire**: OLED VCC → ESP32 3.3V pin (share with GPS)
2. **Black wire**: OLED GND → ESP32 GND pin (share with GPS)
3. **Blue wire**: OLED SDA → ESP32 GPIO 21
4. **White wire**: OLED SCL → ESP32 GPIO 22

### Step 4: Connect Button (2 wires)
1. Place button on breadboard
2. **Wire 1**: One leg → ESP32 GPIO 4
3. **Wire 2**: Opposite leg → ESP32 GND

> No resistor needed! The code uses ESP32's internal pull-up resistor.

### Step 5: Power Up
1. Connect ESP32 to computer via Micro USB cable
2. OLED should light up with boot screen
3. GPS module's LED should start blinking (searching for satellites)

## Battery Connection (Phase 2)

```
18650 Battery (+) → Slide Switch → TP4056 B+
18650 Battery (-) → TP4056 B-
TP4056 OUT+       → ESP32 VIN pin
TP4056 OUT-       → ESP32 GND pin
```

> Charge via TP4056 USB-C port. Device works while charging.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OLED blank | Check SDA/SCL wires, verify I2C address (0x3C or 0x3D) |
| GPS no data | Check TX→RX cross-connection, wait 1-2min for first fix |
| GPS no fix | Go OUTSIDE, clear sky needed (won't work indoors!) |
| WiFi won't connect | Check SSID/password in config.h, ensure hotspot is on |
| Button not working | Check GPIO 4 connection, try different button |
| ESP32 not detected | Try different USB cable (must be DATA cable, not charge-only) |
| Upload fails | Hold BOOT button while uploading, release after "Connecting..." |
