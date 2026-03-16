import AsyncStorage from '@react-native-async-storage/async-storage';

// API Base URL - change this for production
const API_BASE_URL = 'http://10.0.2.2:5000/api'; // Android Emulator
// const API_BASE_URL = 'http://192.168.1.10:5000/api'; // Physical Device
// const API_BASE_URL = 'http://localhost:5000/api'; // iOS Simulator
// const API_BASE_URL = 'https://your-production-url.com/api'; // Production

// Storage keys
const TOKEN_KEY = '@auth_token';
const USER_KEY = '@user_data';

// API Helper
const apiRequest = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  console.log('API Request:', url, options.method || 'GET');

  // Get token from storage
  const token = await AsyncStorage.getItem(TOKEN_KEY);

  const defaultHeaders = {
    'Content-Type': 'application/json',
  };

  if (token) {
    defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  const config = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };

  try {
    console.log('Sending request to:', url);
    const response = await fetch(url, config);
    const data = await response.json();
    console.log('API Response:', data);

    if (!response.ok) {
      throw new Error(data.message || 'Something went wrong');
    }

    return data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

// Auth API
export const authAPI = {
  // Register with email/password
  register: async (email, password, displayName) => {
    const response = await apiRequest('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, displayName }),
    });

    if (response.success && response.data) {
      await AsyncStorage.setItem(TOKEN_KEY, response.data.token);
      await AsyncStorage.setItem(USER_KEY, JSON.stringify(response.data.user));
    }

    return response;
  },

  // Login with email/password
  login: async (email, password) => {
    const response = await apiRequest('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });

    if (response.success && response.data) {
      await AsyncStorage.setItem(TOKEN_KEY, response.data.token);
      await AsyncStorage.setItem(USER_KEY, JSON.stringify(response.data.user));
    }

    return response;
  },

  // Google Sign-In
  googleAuth: async (userData) => {
    const response = await apiRequest('/auth/google', {
      method: 'POST',
      body: JSON.stringify(userData),
    });

    if (response.success && response.data) {
      await AsyncStorage.setItem(TOKEN_KEY, response.data.token);
      await AsyncStorage.setItem(USER_KEY, JSON.stringify(response.data.user));
    }

    return response;
  },

  // Get current user
  getMe: async () => {
    return await apiRequest('/auth/me');
  },

  // Logout
  logout: async () => {
    try {
      await apiRequest('/auth/logout', { method: 'POST' });
    } catch (error) {
      // Ignore logout API errors
    }
    await AsyncStorage.removeItem(TOKEN_KEY);
    await AsyncStorage.removeItem(USER_KEY);
  },

  // Check if user is logged in
  isLoggedIn: async () => {
    const token = await AsyncStorage.getItem(TOKEN_KEY);
    return !!token;
  },

  // Get stored user
  getStoredUser: async () => {
    const user = await AsyncStorage.getItem(USER_KEY);
    return user ? JSON.parse(user) : null;
  },

  // Get stored token
  getToken: async () => {
    return await AsyncStorage.getItem(TOKEN_KEY);
  },
};

export default apiRequest;
