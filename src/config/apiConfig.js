/**
 * API Configuration - CHANGE IP HERE ONLY!
 *
 * When your IP changes, update ONLY this file.
 * All screens and services will automatically use the new IP.
 */

// ============================================
// 👇 CHANGE YOUR IP HERE 👇
// ============================================
// const CURRENT_IP = '10.104.221.192'; // WiFi IP (for physical device on same network)
// const CURRENT_IP = '10.0.2.2'; // Emulator (use this for Android Emulator)
const CURRENT_IP = 'localhost'; // USB with adb reverse (works for both phone & emulator)
// ============================================

// API Configuration
export const API_CONFIG = {
  // ML API (Flask - Port 5001)
  ML_API: `http://${CURRENT_IP}:5001`,

  // Backend API (Node.js - Port 5000)
  BACKEND_API: `http://${CURRENT_IP}:5000/api`,

  // For Android Emulator (don't change)
  EMULATOR_ML: 'http://10.0.2.2:5001',
  EMULATOR_BACKEND: 'http://10.0.2.2:5000/api',

  // For USB connection with adb reverse (don't change)
  USB_ML: 'http://localhost:5001',
  USB_BACKEND: 'http://localhost:5000/api',
};

// Export the active URLs
export const ML_API_URL = API_CONFIG.ML_API;
export const BACKEND_API_URL = API_CONFIG.BACKEND_API;

export default API_CONFIG;
