/**
 * API Configuration - CHANGE IP HERE ONLY!
 *
 * When your IP changes, update ONLY this file.
 * All screens and services will automatically use the new IP.
 */

// ============================================
// 👇 CHANGE YOUR IP HERE 👇
// ============================================
// const CURRENT_IP = 'localhost'; // USB with adb reverse (development)
const CURRENT_IP = 'research-production-ed2e.up.railway.app'; // Railway cloud (production)
// ============================================

// Cloud URLs
const CLOUD_BACKEND = 'https://research-production-ed2e.up.railway.app';
const CLOUD_ML = 'https://ravindu111-coconut-health-ml-api.hf.space';

// Detect if using cloud or local
const IS_CLOUD = CURRENT_IP.includes('railway.app');
const PROTOCOL = IS_CLOUD ? 'https' : 'http';
const PORT_SUFFIX = IS_CLOUD ? '' : ':5000';

// API Configuration
export const API_CONFIG = {
  // ML API (Hugging Face Spaces / local Flask)
  ML_API: IS_CLOUD ? CLOUD_ML : `http://${CURRENT_IP}:5001`,

  // Backend API (Railway / local Node.js)
  BACKEND_API: IS_CLOUD ? `${CLOUD_BACKEND}/api` : `http://${CURRENT_IP}:5000/api`,

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

/**
 * Fetch with timeout and retry - handles HF Spaces cold start
 */
export const fetchWithTimeout = async (url, options = {}, timeoutMs = 60000, retries = 2) => {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      if (attempt === retries) throw error;
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
};

export default API_CONFIG;
