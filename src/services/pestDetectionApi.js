/**
 * Pest & Disease Detection API Service v7.0
 * Connects to Flask ML API for pest and disease detection
 *
 * Models:
 * - Mite: v10 (3-class: coconut_mite, healthy, not_coconut) - 91.44% accuracy
 * - Unified: v1 (4-class: caterpillar, healthy, not_coconut, white_fly) - 96.08% accuracy
 * - Disease: v2 (4-class: Leaf Rot, Leaf_Spot, healthy, not_cocount) - 98.69% accuracy
 * - Leaf Health: v1 (2-class: healthy, unhealthy) - 93.70% accuracy
 * - Branch Health: v1 (2-class: healthy, unhealthy) - 99.63% accuracy
 *
 * Features:
 * - All models can detect non-coconut images
 * - Smart combined logic for "All Pests" detection
 * - Disease detection for Leaf Rot and Leaf Spot
 * - Leaf health detection with detailed conditions and solutions
 * - Branch health detection with unhealthy percentage
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
 * Detect White Fly damage
 */
export const detectWhiteFly = async (imageUri) => {
  try {
    const formData = createFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}/predict/white_fly`, {
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
        pestType: 'white_fly',
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
 * Detect ALL pests (Mite + Caterpillar + White Fly) with Smart Combined Logic
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
 * Detect Leaf Diseases (Leaf Rot, Leaf Spot)
 *
 * Response includes:
 * - prediction: class, confidence, is_diseased, is_leaf_rot, is_leaf_spot, label, message, status
 * - probabilities: leaf_rot, leaf_spot, healthy, not_coconut
 */
export const detectDisease = async (imageUri) => {
  try {
    const formData = createFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}/predict/disease`, {
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
        detectionType: 'disease',
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
 * Detect Leaf Health (healthy vs unhealthy/yellowing)
 *
 * Response includes:
 * - prediction: class name (healthy/unhealthy)
 * - confidence: prediction confidence
 * - probabilities: healthy, unhealthy
 * - message: detailed message about condition
 * - recommendation: action to take
 * - possible_conditions: detailed list of 9 conditions (if unhealthy)
 */
export const detectLeafHealth = async (imageUri) => {
  try {
    const formData = createFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}/predict/leaf-health`, {
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
        detectionType: 'leaf_health',
        prediction: data.prediction,
        confidence: data.confidence,
        probabilities: data.probabilities,
        isHealthy: data.is_healthy,
        message: data.message,
        recommendation: data.recommendation,
        possibleConditions: data.possible_conditions,
        conditionsCount: data.conditions_count,
        modelInfo: data.model_info,
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
 * Detect Branch Health (healthy vs unhealthy)
 *
 * Response includes:
 * - prediction: class name (healthy/unhealthy)
 * - confidence: prediction confidence
 * - probabilities: healthy, unhealthy
 * - unhealthy_percentage: percentage of branch that's unhealthy
 * - message: detailed message about condition
 * - recommendation: action to take
 */
export const detectBranchHealth = async (imageUri) => {
  try {
    const formData = createFormData(imageUri);

    const response = await fetch(`${API_BASE_URL}/predict/branch-health`, {
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
        detectionType: 'branch_health',
        prediction: data.prediction,
        confidence: data.confidence,
        probabilities: data.probabilities,
        unhealthyPercentage: data.unhealthy_percentage,
        isHealthy: data.is_healthy,
        message: data.message,
        recommendation: data.recommendation,
        modelInfo: data.model_info,
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
  WHITE_FLY: 'white_fly',
  ALL: 'all',
};

// Disease types for convenience
export const DISEASE_TYPES = {
  LEAF_ROT: 'Leaf Rot',
  LEAF_SPOT: 'Leaf_Spot',
  HEALTHY: 'healthy',
  NOT_COCONUT: 'not_cocount',
};

export default {
  checkApiHealth,
  getModelsInfo,
  detectMite,
  detectCaterpillar,
  detectWhiteFly,
  detectAllPests,
  detectDisease,
  detectLeafHealth,
  detectBranchHealth,
  predictPest,
  PEST_TYPES,
  DISEASE_TYPES,
  API_BASE_URL,
};
