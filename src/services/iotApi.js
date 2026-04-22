/**
 * IoT Device API Service
 * Handles IoT device management and location data
 * Version: 1.0
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import {BACKEND_API_URL} from '../config/apiConfig';

const TOKEN_KEY = '@auth_token';

const apiRequest = async (endpoint, options = {}) => {
  const token = await AsyncStorage.getItem(TOKEN_KEY);

  if (!token) {
    throw new Error('Not authenticated');
  }

  const config = {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  };

  const response = await fetch(`${BACKEND_API_URL}${endpoint}`, config);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.message || 'Request failed');
  }

  return data;
};

export const iotAPI = {
  /**
   * Register a new IoT device
   * @param {Object} deviceData - { deviceId, name, defaultPlantation }
   * @returns {Promise<Object>} - Device with deviceKey (save this!)
   */
  registerDevice: async (deviceData) => {
    const response = await apiRequest('/iot/devices', {
      method: 'POST',
      body: JSON.stringify(deviceData),
    });
    return response.data;
  },

  /**
   * Get all user's IoT devices
   * @returns {Promise<Array>} - List of devices
   */
  getMyDevices: async () => {
    try {
      const response = await apiRequest('/iot/devices');
      return response.data || [];
    } catch (error) {
      console.error('Error getting IoT devices:', error);
      return [];
    }
  },

  /**
   * Update an IoT device
   * @param {string} deviceId - Device document ID
   * @param {Object} updateData - { name, defaultPlantation, isActive }
   * @returns {Promise<Object>} - Updated device
   */
  updateDevice: async (deviceId, updateData) => {
    const response = await apiRequest(`/iot/devices/${deviceId}`, {
      method: 'PUT',
      body: JSON.stringify(updateData),
    });
    return response.data;
  },

  /**
   * Delete an IoT device
   * @param {string} deviceId - Device document ID
   * @returns {Promise<boolean>}
   */
  deleteDevice: async (deviceId) => {
    await apiRequest(`/iot/devices/${deviceId}`, {
      method: 'DELETE',
    });
    return true;
  },

  /**
   * Regenerate device key
   * @param {string} deviceId - Device document ID
   * @returns {Promise<Object>} - { deviceId, deviceKey }
   */
  regenerateKey: async (deviceId) => {
    const response = await apiRequest(`/iot/devices/${deviceId}/regenerate-key`, {
      method: 'POST',
    });
    return response.data;
  },

  /**
   * Get locations captured by a device
   * @param {string} deviceId - Device document ID
   * @returns {Promise<Object>} - { trees, device info }
   */
  /**
   * Get live location of a device (for real-time map tracking)
   * @param {string} deviceId - Device document ID
   * @returns {Promise<Object>} - { isOnline, location, liveData }
   */
  getDeviceLiveLocation: async (deviceId) => {
    try {
      const response = await apiRequest(`/iot/devices/${deviceId}/live`);
      return response.data || null;
    } catch (error) {
      console.error('Error getting live location:', error);
      return null;
    }
  },

  getDeviceLocations: async (deviceId) => {
    try {
      const response = await apiRequest(`/iot/devices/${deviceId}/locations`);
      return {
        trees: response.data || [],
        device: response.device || null,
      };
    } catch (error) {
      console.error('Error getting device locations:', error);
      return {trees: [], device: null};
    }
  },
};

export default iotAPI;
