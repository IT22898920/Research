import apiRequest from './api';

// User Management API (Admin only)
export const userAPI = {
  // Get all users with pagination, search, and filters
  getUsers: async (params = {}) => {
    const {
      page = 1,
      limit = 10,
      search = '',
      role = '',
      isActive = '',
      sort = '-createdAt'
    } = params;

    const queryParams = new URLSearchParams();
    queryParams.append('page', page);
    queryParams.append('limit', limit);
    if (search) queryParams.append('search', search);
    if (role) queryParams.append('role', role);
    if (isActive !== '') queryParams.append('isActive', isActive);
    queryParams.append('sort', sort);

    return await apiRequest(`/users?${queryParams.toString()}`);
  },

  // Get user statistics
  getUserStats: async () => {
    return await apiRequest('/users/stats');
  },

  // Get single user by ID
  getUserById: async (id) => {
    return await apiRequest(`/users/${id}`);
  },

  // Update user (role, isActive, displayName)
  updateUser: async (id, data) => {
    return await apiRequest(`/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  // Delete user
  deleteUser: async (id) => {
    return await apiRequest(`/users/${id}`, {
      method: 'DELETE',
    });
  },
};

export default userAPI;
