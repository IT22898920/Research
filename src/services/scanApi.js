import AsyncStorage from '@react-native-async-storage/async-storage';
import {Platform} from 'react-native';

const API_BASE_URL = 'http://10.0.2.2:5000/api'; // Android Emulator
const TOKEN_KEY = '@auth_token';

const apiRequest = async (endpoint, options = {}) => {
  const token = await AsyncStorage.getItem(TOKEN_KEY);

  const config = {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  };

  const response = await fetch(`${API_BASE_URL}${endpoint}`, config);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.message || 'Request failed');
  }

  return data;
};

export const scanAPI = {
  // Save scan result with optional image
  saveScan: async (scanData, imageBase64 = null) => {
    return apiRequest('/scans', {
      method: 'POST',
      body: JSON.stringify({
        ...scanData,
        imageBase64: imageBase64, // Will be uploaded to Cloudinary
        deviceInfo: {
          platform: Platform.OS,
          model: Platform.constants?.Model || 'Unknown',
        },
      }),
    });
  },

  // Get user's scan history
  getMyScans: async (params = {}) => {
    const queryString = new URLSearchParams(params).toString();
    return apiRequest(`/scans/my-scans?${queryString}`);
  },

  // Get user's statistics
  getMyStats: async () => {
    return apiRequest('/scans/my-stats');
  },

  // Get single scan
  getScan: async scanId => {
    return apiRequest(`/scans/${scanId}`);
  },

  // Delete scan
  deleteScan: async scanId => {
    return apiRequest(`/scans/${scanId}`, {method: 'DELETE'});
  },

  // Admin: Get overall analytics
  getAnalytics: async (period = 30) => {
    return apiRequest(`/scans/admin/analytics?period=${period}`);
  },

  // Admin: Get infection trends
  getTrends: async (period = 30, groupBy = 'day') => {
    return apiRequest(`/scans/admin/trends?period=${period}&groupBy=${groupBy}`);
  },

  // Admin: Get pest distribution
  getPestDistribution: async () => {
    return apiRequest('/scans/admin/pest-distribution');
  },
};

// Notification API
export const notificationAPI = {
  // Update FCM token on server
  updateFCMToken: async (fcmToken) => {
    return apiRequest('/notifications/token', {
      method: 'POST',
      body: JSON.stringify({ fcmToken }),
    });
  },

  // Get notification settings
  getSettings: async () => {
    return apiRequest('/notifications/settings');
  },

  // Update notification settings
  updateSettings: async (enabled) => {
    return apiRequest('/notifications/settings', {
      method: 'PUT',
      body: JSON.stringify({ enabled }),
    });
  },
};

export default scanAPI;
