/**
 * Pest Detection API Service v4.0
 * Connects to Flask ML API for pest detection
 *
 * Models:
 * - Mite: v10 (3-class: coconut_mite, healthy, not_coconut) - 91.44% accuracy
 * - Caterpillar: v2 (3-class: caterpillar, healthy, not_coconut) - 97.47% accuracy
 *
 * Features:
 * - Both models can detect non-coconut images
 * - Smart combined logic for "All Pests" detection
 * - One coherent answer with recommendations
 */

// API Configuration
// Use your computer's IP for real device, localhost for emulator
// ML API runs on port 5001 (Auth backend runs on port 5000)
const API_CONFIG = {
  // For Android Emulator
  emulator: 'http://10.0.2.2:5001',
  // For iOS Simulator
  ios: 'http://localhost:5001',
  // For Real Device - replace with your computer's IP
  device: 'http://192.168.8.196:5001',
};

// Change this based on your testing environment
const API_BASE_URL = API_CONFIG.emulator;

/**
 * Check if the ML API is available
 */
export const checkApiHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });
    const data = await response.json();
    return {
      success: response.ok,
      healthy: data.status === 'healthy',
      models: data.models, // { mite: true, caterpillar: true }
    };
  } catch (error) {
    return {
      success: false,
      error: error.message || 'Cannot connect to ML API',
    };
  }
};

/**
 * Get all models information
 */
export const getModelsInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/models`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });
    const data = await response.json();
    return {
      success: response.ok,
      models: data,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
    };
  }
};

/**
 * Helper function to create form data
 */
const createFormData = (imageUri) => {
  const formData = new FormData();
  const filename = imageUri.split('/').pop();

  formData.append('image', {
    uri: imageUri,
    type: 'image/jpeg',
    name: filename || 'image.jpg',
  });

  return formData;
};

/**
 * Detect Coconut Mite infection
 */
export const detectMite = async (imageUri) => {
  try {
    const formData = createFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}/predict/mite`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        pestType: 'mite',
        prediction: data.prediction,
        probabilities: data.probabilities,
        timestamp: data.timestamp,
      };
    } else {
      return {
        success: false,
        error: data.error || 'Prediction failed',
      };
    }
  } catch (error) {
    return {
      success: false,
      error: error.message || 'Failed to analyze image',
    };
  }
};

/**
 * Detect Caterpillar damage
 */
export const detectCaterpillar = async (imageUri) => {
  try {
    const formData = createFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}/predict/caterpillar`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        pestType: 'caterpillar',
        prediction: data.prediction,
        probabilities: data.probabilities,
        timestamp: data.timestamp,
      };
    } else {
      return {
        success: false,
        error: data.error || 'Prediction failed',
      };
    }
  } catch (error) {
    return {
      success: false,
      error: error.message || 'Failed to analyze image',
    };
  }
};

/**
 * Detect ALL pests (Mite + Caterpillar) with Smart Combined Logic
 *
 * Response includes:
 * - results: individual model results
 * - summary: combined decision with status, label, message, recommendation
 * - models_used: version info for both models
 */
export const detectAllPests = async (imageUri) => {
  try {
    const formData = createFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}/predict/all`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        results: data.results,
        summary: data.summary,
        modelsUsed: data.models_used,
        timestamp: data.timestamp,
      };
    } else {
      return {
        success: false,
        error: data.error || 'Prediction failed',
      };
    }
  } catch (error) {
    return {
      success: false,
      error: error.message || 'Failed to analyze image',
    };
  }
};

/**
 * Legacy function - Predict pest (defaults to mite for backward compatibility)
 * @deprecated Use detectMite, detectCaterpillar, or detectAllPests instead
 */
export const predictPest = async (imageUri) => {
  return detectMite(imageUri);
};

// Pest types for convenience
export const PEST_TYPES = {
  MITE: 'mite',
  CATERPILLAR: 'caterpillar',
  ALL: 'all',
};

export default {
  checkApiHealth,
  getModelsInfo,
  detectMite,
  detectCaterpillar,
  detectAllPests,
  predictPest,
  PEST_TYPES,
  API_BASE_URL,
};
