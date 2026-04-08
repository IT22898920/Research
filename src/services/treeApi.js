/**
 * Tree API Service
 * Handles all tree-related API calls to the backend
 * Version: 2.0 - Backend Integration
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

/**
 * Tree API functions
 */
export const treeAPI = {
  /**
   * Get all trees for current user
   * @param {Object} params - Query params (page, limit, sort)
   * @returns {Promise<Array>} - Trees list
   */
  getAllTrees: async (params = {}) => {
    try {
      const queryString = new URLSearchParams(params).toString();
      const endpoint = queryString ? `/trees?${queryString}` : '/trees';
      const response = await apiRequest(endpoint);
      return response.data || [];
    } catch (error) {
      console.error('Error getting trees:', error);
      return [];
    }
  },

  /**
   * Get trees by plantation ID
   * @param {string} plantationId - Plantation ID
   * @returns {Promise<Object>} - { trees, plantation, pagination }
   */
  getTreesByPlantation: async (plantationId) => {
    try {
      const response = await apiRequest(`/trees/plantation/${plantationId}`);
      return {
        trees: response.data || [],
        plantation: response.plantation || null,
        pagination: response.pagination || null,
      };
    } catch (error) {
      console.error('Error getting trees by plantation:', error);
      return {trees: [], plantation: null, pagination: null};
    }
  },

  /**
   * Get single tree by ID
   * @param {string} treeId - Tree ID
   * @returns {Promise<Object|null>} - Tree data
   */
  getTreeById: async (treeId) => {
    try {
      const response = await apiRequest(`/trees/${treeId}`);
      return response.data || null;
    } catch (error) {
      console.error('Error getting tree:', error);
      return null;
    }
  },

  /**
   * Add new tree to a plantation
   * @param {Object} treeData - { plantation, label, latitude, longitude, accuracy, locationName, notes }
   * @returns {Promise<Object>} - Created tree
   */
  addTree: async (treeData) => {
    try {
      const response = await apiRequest('/trees', {
        method: 'POST',
        body: JSON.stringify(treeData),
      });
      return response.data;
    } catch (error) {
      console.error('Error adding tree:', error);
      throw error;
    }
  },

  /**
   * Update tree data
   * @param {string} treeId - Tree ID
   * @param {Object} updateData - Fields to update
   * @returns {Promise<Object>} - Updated tree
   */
  updateTree: async (treeId, updateData) => {
    try {
      const response = await apiRequest(`/trees/${treeId}`, {
        method: 'PUT',
        body: JSON.stringify(updateData),
      });
      return response.data;
    } catch (error) {
      console.error('Error updating tree:', error);
      throw error;
    }
  },

  /**
   * Add health scan to tree
   * @param {string} treeId - Tree ID
   * @param {Object} scanData - { status, scanType, confidence, details }
   * @returns {Promise<Object>} - Updated tree with health record
   */
  addHealthScan: async (treeId, scanData) => {
    try {
      const response = await apiRequest(`/trees/${treeId}/health-scan`, {
        method: 'POST',
        body: JSON.stringify(scanData),
      });
      return response.data;
    } catch (error) {
      console.error('Error adding health scan:', error);
      throw error;
    }
  },

  /**
   * Update scan results (nuts, bunches, pests, issues)
   * @param {string} treeId - Tree ID
   * @param {Object} scanResults - { nutCount, bunchCount, pestCount, detectedIssues, healthStatus }
   * @returns {Promise<Object>} - Updated tree
   */
  updateScanResults: async (treeId, scanResults) => {
    try {
      const response = await apiRequest(`/trees/${treeId}/scan-results`, {
        method: 'PUT',
        body: JSON.stringify(scanResults),
      });
      return response.data;
    } catch (error) {
      console.error('Error updating scan results:', error);
      throw error;
    }
  },

  /**
   * Clear detected issues for a tree
   * @param {string} treeId - Tree ID
   * @returns {Promise<Object>} - Updated tree
   */
  clearDetectedIssues: async (treeId) => {
    try {
      const response = await apiRequest(`/trees/${treeId}/issues`, {
        method: 'DELETE',
      });
      return response.data;
    } catch (error) {
      console.error('Error clearing issues:', error);
      throw error;
    }
  },

  /**
   * Delete tree
   * @param {string} treeId - Tree ID
   * @returns {Promise<boolean>} - Success status
   */
  deleteTree: async (treeId) => {
    try {
      await apiRequest(`/trees/${treeId}`, {
        method: 'DELETE',
      });
      return true;
    } catch (error) {
      console.error('Error deleting tree:', error);
      throw error;
    }
  },

  /**
   * Get tree statistics
   * @param {string|null} plantationId - Optional plantation ID to filter
   * @returns {Promise<Object>} - Stats { totalTrees, healthyTrees, unhealthyTrees, infectedTrees, notScannedTrees }
   */
  getTreeStats: async (plantationId = null) => {
    try {
      const endpoint = plantationId
        ? `/trees/stats?plantationId=${plantationId}`
        : '/trees/stats';
      const response = await apiRequest(endpoint);

      // Map to consistent format
      const data = response.data || {};
      return {
        total: data.totalTrees || 0,
        healthy: data.healthyTrees || 0,
        unhealthy: data.unhealthyTrees || 0,
        infected: data.infectedTrees || 0,
        notScanned: data.notScannedTrees || 0,
      };
    } catch (error) {
      console.error('Error getting tree stats:', error);
      return {total: 0, healthy: 0, unhealthy: 0, infected: 0, notScanned: 0};
    }
  },

  /**
   * Bulk import trees
   * @param {Array} trees - Array of tree objects
   * @param {string} plantationId - Plantation ID
   * @returns {Promise<Array>} - Imported trees
   */
  bulkImport: async (trees, plantationId) => {
    try {
      const response = await apiRequest('/trees/bulk', {
        method: 'POST',
        body: JSON.stringify({trees, plantationId}),
      });
      return response.data || [];
    } catch (error) {
      console.error('Error bulk importing trees:', error);
      throw error;
    }
  },

  /**
   * Get nearby trees
   * @param {number} lat - Latitude
   * @param {number} lng - Longitude
   * @param {number} radius - Radius in meters (default 1000)
   * @returns {Promise<Array>} - Nearby trees
   */
  getNearby: async (lat, lng, radius = 1000) => {
    try {
      const response = await apiRequest(
        `/trees/nearby?lat=${lat}&lng=${lng}&radius=${radius}`,
      );
      return response.data || [];
    } catch (error) {
      console.error('Error getting nearby trees:', error);
      return [];
    }
  },
};

// Re-export plantationAPI from the new service for backwards compatibility
export {plantationAPI} from './plantationApi';

export default {treeAPI};
