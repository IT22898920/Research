import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@tree_health_history';
const MAX_RECORDS = 50;

/**
 * Save a tree health analysis result to local history
 */
export const saveTreeHealthRecord = async (result, imageUri = null) => {
  try {
    const existing = await getTreeHealthHistory();

    const record = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      prediction: result.prediction,
      isHealthy: result.isHealthy,
      confidence: result.confidence,
      unhealthyPercentage: result.unhealthyPercentage || 0,
      severity: result.severityInfo?.severity || null,
      urgency: result.severityInfo?.urgency || null,
      urgencyColor: result.severityInfo?.urgencyColor || null,
      message: result.message,
      imageUri: imageUri || null,
    };

    const updated = [record, ...existing].slice(0, MAX_RECORDS);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return record;
  } catch (error) {
    console.error('Failed to save tree health record:', error);
    return null;
  }
};

/**
 * Get all tree health history records
 */
export const getTreeHealthHistory = async () => {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('Failed to load tree health history:', error);
    return [];
  }
};

/**
 * Delete a specific record by id
 */
export const deleteTreeHealthRecord = async (id) => {
  try {
    const existing = await getTreeHealthHistory();
    const updated = existing.filter(r => r.id !== id);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return true;
  } catch (error) {
    console.error('Failed to delete record:', error);
    return false;
  }
};

/**
 * Clear all history
 */
export const clearTreeHealthHistory = async () => {
  try {
    await AsyncStorage.removeItem(STORAGE_KEY);
    return true;
  } catch (error) {
    console.error('Failed to clear history:', error);
    return false;
  }
};

/**
 * Get trend data (last N records, oldest first) for chart
 */
export const getTreeHealthTrend = async (limit = 10) => {
  try {
    const history = await getTreeHealthHistory();
    return history.slice(0, limit).reverse();
  } catch (error) {
    return [];
  }
};
