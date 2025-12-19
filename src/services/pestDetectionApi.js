/**
 * Pest Detection API Service
 * Connects to Flask ML API for coconut mite detection
 */

// API Configuration
// Use your computer's IP for real device, localhost for emulator
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
        'Accept': 'application/json',
      },
    });
    const data = await response.json();
    return {
      success: response.ok,
      data: data,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message || 'Cannot connect to ML API',
    };
  }
};

/**
 * Get model information
 */
export const getModelInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/model-info`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    const data = await response.json();
    return {
      success: response.ok,
      data: data,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
    };
  }
};

/**
 * Predict pest infection from image
 * @param {string} imageUri - Local URI of the image
 * @returns {Promise} Prediction result
 */
export const predictPest = async (imageUri) => {
  try {
    // Create form data
    const formData = new FormData();

    // Get filename from URI
    const filename = imageUri.split('/').pop();

    // Append image to form data
    formData.append('image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: filename || 'image.jpg',
    });

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
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
 * Predict pest infection for multiple images
 * @param {Array} imageUris - Array of local image URIs
 * @returns {Promise} Batch prediction results
 */
export const predictBatch = async (imageUris) => {
  try {
    const formData = new FormData();

    imageUris.forEach((uri, index) => {
      const filename = uri.split('/').pop();
      formData.append('images', {
        uri: uri,
        type: 'image/jpeg',
        name: filename || `image_${index}.jpg`,
      });
    });

    const response = await fetch(`${API_BASE_URL}/predict/batch`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    const data = await response.json();
    return {
      success: data.success,
      results: data.results,
      count: data.count,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
    };
  }
};

export default {
  checkApiHealth,
  getModelInfo,
  predictPest,
  predictBatch,
  API_BASE_URL,
};
