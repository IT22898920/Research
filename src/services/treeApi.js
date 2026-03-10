/**
 * Tree Location API Service
 * Manages coconut tree locations and data
 * Version: 1.0
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const TREES_STORAGE_KEY = '@coconut_trees';
const PLANTATIONS_STORAGE_KEY = '@plantations';

/**
 * Generate unique ID for trees
 */
const generateTreeId = () => {
  return 'TREE_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

/**
 * Generate unique ID for plantations
 */
const generatePlantationId = () => {
  return 'PLANT_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

/**
 * Tree API functions
 */
export const treeAPI = {
  /**
   * Get all trees
   */
  getAllTrees: async () => {
    try {
      const treesJson = await AsyncStorage.getItem(TREES_STORAGE_KEY);
      return treesJson ? JSON.parse(treesJson) : [];
    } catch (error) {
      console.error('Error getting trees:', error);
      return [];
    }
  },

  /**
   * Get trees by plantation ID
   */
  getTreesByPlantation: async (plantationId) => {
    try {
      const trees = await treeAPI.getAllTrees();
      return trees.filter(tree => tree.plantationId === plantationId);
    } catch (error) {
      console.error('Error getting trees by plantation:', error);
      return [];
    }
  },

  /**
   * Get single tree by ID
   */
  getTreeById: async (treeId) => {
    try {
      const trees = await treeAPI.getAllTrees();
      return trees.find(tree => tree.id === treeId) || null;
    } catch (error) {
      console.error('Error getting tree:', error);
      return null;
    }
  },

  /**
   * Add new tree
   */
  addTree: async (treeData) => {
    try {
      const trees = await treeAPI.getAllTrees();
      const newTree = {
        id: generateTreeId(),
        ...treeData,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        healthHistory: [],
        yieldHistory: [],
      };
      trees.push(newTree);
      await AsyncStorage.setItem(TREES_STORAGE_KEY, JSON.stringify(trees));
      return newTree;
    } catch (error) {
      console.error('Error adding tree:', error);
      throw error;
    }
  },

  /**
   * Update tree data
   */
  updateTree: async (treeId, updateData) => {
    try {
      const trees = await treeAPI.getAllTrees();
      const index = trees.findIndex(tree => tree.id === treeId);
      if (index !== -1) {
        trees[index] = {
          ...trees[index],
          ...updateData,
          updatedAt: new Date().toISOString(),
        };
        await AsyncStorage.setItem(TREES_STORAGE_KEY, JSON.stringify(trees));
        return trees[index];
      }
      return null;
    } catch (error) {
      console.error('Error updating tree:', error);
      throw error;
    }
  },

  /**
   * Add health scan to tree
   */
  addHealthScan: async (treeId, scanData) => {
    try {
      const trees = await treeAPI.getAllTrees();
      const index = trees.findIndex(tree => tree.id === treeId);
      if (index !== -1) {
        const healthRecord = {
          id: 'SCAN_' + Date.now(),
          ...scanData,
          scannedAt: new Date().toISOString(),
        };
        trees[index].healthHistory.push(healthRecord);
        trees[index].lastHealthStatus = scanData.status;
        trees[index].updatedAt = new Date().toISOString();
        await AsyncStorage.setItem(TREES_STORAGE_KEY, JSON.stringify(trees));
        return healthRecord;
      }
      return null;
    } catch (error) {
      console.error('Error adding health scan:', error);
      throw error;
    }
  },

  /**
   * Update scan results (nuts, bunches, pests, issues)
   */
  updateScanResults: async (treeId, scanResults) => {
    try {
      const trees = await treeAPI.getAllTrees();
      const index = trees.findIndex(tree => tree.id === treeId);
      if (index !== -1) {
        // Update counts
        if (scanResults.nutCount !== undefined) {
          trees[index].nutCount = scanResults.nutCount;
        }
        if (scanResults.bunchCount !== undefined) {
          trees[index].bunchCount = scanResults.bunchCount;
        }
        if (scanResults.pestCount !== undefined) {
          trees[index].pestCount = scanResults.pestCount;
        }

        // Update detected issues (append new ones, avoid duplicates)
        if (scanResults.detectedIssues && scanResults.detectedIssues.length > 0) {
          const existingIssues = trees[index].detectedIssues || [];
          const newIssues = scanResults.detectedIssues.filter(
            issue => !existingIssues.includes(issue)
          );
          trees[index].detectedIssues = [...existingIssues, ...newIssues];
        }

        // Update health status if provided
        if (scanResults.healthStatus) {
          trees[index].lastHealthStatus = scanResults.healthStatus;
        }

        trees[index].lastScanAt = new Date().toISOString();
        trees[index].updatedAt = new Date().toISOString();

        await AsyncStorage.setItem(TREES_STORAGE_KEY, JSON.stringify(trees));
        return trees[index];
      }
      return null;
    } catch (error) {
      console.error('Error updating scan results:', error);
      throw error;
    }
  },

  /**
   * Clear detected issues for a tree
   */
  clearDetectedIssues: async (treeId) => {
    try {
      const trees = await treeAPI.getAllTrees();
      const index = trees.findIndex(tree => tree.id === treeId);
      if (index !== -1) {
        trees[index].detectedIssues = [];
        trees[index].updatedAt = new Date().toISOString();
        await AsyncStorage.setItem(TREES_STORAGE_KEY, JSON.stringify(trees));
        return trees[index];
      }
      return null;
    } catch (error) {
      console.error('Error clearing issues:', error);
      throw error;
    }
  },

  /**
   * Delete tree
   */
  deleteTree: async (treeId) => {
    try {
      const trees = await treeAPI.getAllTrees();
      const filteredTrees = trees.filter(tree => tree.id !== treeId);
      await AsyncStorage.setItem(TREES_STORAGE_KEY, JSON.stringify(filteredTrees));
      return true;
    } catch (error) {
      console.error('Error deleting tree:', error);
      throw error;
    }
  },

  /**
   * Get tree statistics
   */
  getTreeStats: async (plantationId = null) => {
    try {
      let trees = await treeAPI.getAllTrees();
      if (plantationId) {
        trees = trees.filter(tree => tree.plantationId === plantationId);
      }

      const stats = {
        total: trees.length,
        healthy: trees.filter(t => t.lastHealthStatus === 'healthy').length,
        unhealthy: trees.filter(t => t.lastHealthStatus === 'unhealthy').length,
        infected: trees.filter(t => t.lastHealthStatus === 'infected').length,
        notScanned: trees.filter(t => !t.lastHealthStatus).length,
      };

      return stats;
    } catch (error) {
      console.error('Error getting tree stats:', error);
      return { total: 0, healthy: 0, unhealthy: 0, infected: 0, notScanned: 0 };
    }
  },
};

/**
 * Plantation API functions
 */
export const plantationAPI = {
  /**
   * Get all plantations
   */
  getAllPlantations: async () => {
    try {
      const plantationsJson = await AsyncStorage.getItem(PLANTATIONS_STORAGE_KEY);
      return plantationsJson ? JSON.parse(plantationsJson) : [];
    } catch (error) {
      console.error('Error getting plantations:', error);
      return [];
    }
  },

  /**
   * Add new plantation
   */
  addPlantation: async (plantationData) => {
    try {
      const plantations = await plantationAPI.getAllPlantations();
      const newPlantation = {
        id: generatePlantationId(),
        ...plantationData,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      plantations.push(newPlantation);
      await AsyncStorage.setItem(PLANTATIONS_STORAGE_KEY, JSON.stringify(plantations));
      return newPlantation;
    } catch (error) {
      console.error('Error adding plantation:', error);
      throw error;
    }
  },

  /**
   * Update plantation
   */
  updatePlantation: async (plantationId, updateData) => {
    try {
      const plantations = await plantationAPI.getAllPlantations();
      const index = plantations.findIndex(p => p.id === plantationId);
      if (index !== -1) {
        plantations[index] = {
          ...plantations[index],
          ...updateData,
          updatedAt: new Date().toISOString(),
        };
        await AsyncStorage.setItem(PLANTATIONS_STORAGE_KEY, JSON.stringify(plantations));
        return plantations[index];
      }
      return null;
    } catch (error) {
      console.error('Error updating plantation:', error);
      throw error;
    }
  },

  /**
   * Delete plantation and all its trees
   */
  deletePlantation: async (plantationId) => {
    try {
      // Delete plantation
      const plantations = await plantationAPI.getAllPlantations();
      const filteredPlantations = plantations.filter(p => p.id !== plantationId);
      await AsyncStorage.setItem(PLANTATIONS_STORAGE_KEY, JSON.stringify(filteredPlantations));

      // Delete all trees in this plantation
      const trees = await treeAPI.getAllTrees();
      const filteredTrees = trees.filter(t => t.plantationId !== plantationId);
      await AsyncStorage.setItem(TREES_STORAGE_KEY, JSON.stringify(filteredTrees));

      return true;
    } catch (error) {
      console.error('Error deleting plantation:', error);
      throw error;
    }
  },
};

export default { treeAPI, plantationAPI };
