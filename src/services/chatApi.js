/**
 * Chat API Service v1.0
 * Connects to Flask API for AI Chatbot (Groq)
 *
 * Features:
 * - Coconut Health Expert Assistant
 * - Multi-language support (English, Sinhala, Tamil)
 * - Chat history support
 * - Fast responses via Groq API (Llama 3.3 70B)
 */

// API Configuration - Same as pest detection API
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
 * Check if the chat service is available
 * @returns {Promise<{success: boolean, status: string, message: string}>}
 */
export const checkChatHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      timeout: 10000,
    });
    const data = await response.json();
    return {
      success: data.success,
      status: data.status,
      message: data.message,
      model: data.model,
    };
  } catch (error) {
    return {
      success: false,
      status: 'offline',
      message: error.message || 'Failed to connect to chat service',
    };
  }
};

/**
 * Send a chat message and get AI response
 * @param {string} message - User's message
 * @param {Array} history - Previous chat messages [{role, content}]
 * @returns {Promise<{success: boolean, response?: string, error?: string}>}
 */
export const sendChatMessage = async (message, history = []) => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        message: message,
        history: history,
      }),
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        response: data.response,
        model: data.model,
        usage: data.usage,
      };
    } else {
      return {
        success: false,
        error: data.error || 'Failed to get response',
      };
    }
  } catch (error) {
    console.error('Chat API Error:', error);
    return {
      success: false,
      error: error.message || 'Network error. Please check your connection.',
    };
  }
};

/**
 * Get quick treatment advice for a specific pest
 * This is a convenience function that asks about treatment
 * @param {string} pestType - Type of pest (mite, caterpillar, white_fly)
 * @returns {Promise<{success: boolean, response?: string, error?: string}>}
 */
export const getQuickTreatmentAdvice = async pestType => {
  const pestNames = {
    mite: 'coconut mite',
    caterpillar: 'black-headed caterpillar',
    white_fly: 'white fly',
    coconut_mite: 'coconut mite',
  };

  const pestName = pestNames[pestType] || pestType;
  const message = `What is the best treatment for ${pestName} infection on coconut trees? Please provide step-by-step treatment instructions.`;

  return sendChatMessage(message, []);
};

/**
 * Get farming tips for coconut care
 * @param {string} topic - Topic (irrigation, fertilization, harvesting, general)
 * @returns {Promise<{success: boolean, response?: string, error?: string}>}
 */
export const getFarmingTips = async (topic = 'general') => {
  const topicQuestions = {
    irrigation: 'What are the best irrigation practices for coconut trees?',
    fertilization: 'How should I fertilize my coconut trees for maximum yield?',
    harvesting: 'What is the best way to harvest coconuts?',
    general: 'What are the top 5 tips for maintaining healthy coconut trees?',
  };

  const message = topicQuestions[topic] || topicQuestions.general;
  return sendChatMessage(message, []);
};

export default {
  checkChatHealth,
  sendChatMessage,
  getQuickTreatmentAdvice,
  getFarmingTips,
};
