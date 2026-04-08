/**
 * Plantation API Service
 * Handles all plantation-related API calls to the backend
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import {BACKEND_API_URL} from '../config/apiConfig';

const TOKEN_KEY = '@auth_token';

/**
 * Make authenticated API request
 */
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

export const plantationAPI = {
  /**
   * Get all plantations for current user
   * @param {Object} params - Query params (page, limit, sort)
   * @returns {Promise<Object>} - Plantations list with pagination
   */
  getAll: async (params = {}) => {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = queryString ? `/plantations?${queryString}` : '/plantations';
    return apiRequest(endpoint);
  },

  /**
   * Get single plantation by ID
   * @param {string} plantationId - Plantation ID
   * @returns {Promise<Object>} - Plantation data
   */
  getById: async (plantationId) => {
    return apiRequest(`/plantations/${plantationId}`);
  },

  /**
   * Create new plantation
   * @param {Object} plantationData - { name, description, location, area }
   * @returns {Promise<Object>} - Created plantation
   */
  create: async (plantationData) => {
    return apiRequest('/plantations', {
      method: 'POST',
      body: JSON.stringify(plantationData),
    });
  },

  /**
   * Update plantation
   * @param {string} plantationId - Plantation ID
   * @param {Object} updateData - Fields to update
   * @returns {Promise<Object>} - Updated plantation
   */
  update: async (plantationId, updateData) => {
    return apiRequest(`/plantations/${plantationId}`, {
      method: 'PUT',
      body: JSON.stringify(updateData),
    });
  },

  /**
   * Delete plantation and all its trees
   * @param {string} plantationId - Plantation ID
   * @returns {Promise<Object>} - Deletion result
   */
  delete: async (plantationId) => {
    return apiRequest(`/plantations/${plantationId}`, {
      method: 'DELETE',
    });
  },

  /**
   * Get plantation statistics
   * @param {string} plantationId - Plantation ID
   * @returns {Promise<Object>} - Plantation stats
   */
  getStats: async (plantationId) => {
    return apiRequest(`/plantations/${plantationId}/stats`);
  },

  /**
   * Get overall analytics across all plantations
   * @returns {Promise<Object>} - Analytics data
   */
  getAnalytics: async () => {
    return apiRequest('/plantations/analytics');
  },
};

export default plantationAPI;
