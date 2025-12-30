import React, {useState, useEffect, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  TextInput,
  ActivityIndicator,
  Modal,
  Alert,
  RefreshControl,
} from 'react-native';
import {userAPI} from '../services/userApi';
import {useLanguage} from '../context/LanguageContext';

export default function UserManagementScreen({navigation}) {
  const {t} = useLanguage();
  const [users, setUsers] = useState([]);
  const [stats, setStats] = useState({
    totalUsers: 0,
    activeUsers: 0,
    inactiveUsers: 0,
    adminUsers: 0,
    regularUsers: 0,
  });
  const [pagination, setPagination] = useState({
    currentPage: 1,
    totalPages: 1,
    totalUsers: 0,
    hasMore: false,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');

  // Modal states
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [editRole, setEditRole] = useState('');
  const [editIsActive, setEditIsActive] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const fetchStats = async () => {
    try {
      const response = await userAPI.getUserStats();
      if (response.success) {
        setStats(response.data);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchUsers = async (page = 1, refresh = false) => {
    try {
      if (refresh) {
        setIsRefreshing(true);
      } else if (page === 1) {
        setIsLoading(true);
      }

      const response = await userAPI.getUsers({
        page,
        limit: 10,
        search,
        role: roleFilter,
        isActive: statusFilter,
      });

      if (response.success) {
        setUsers(response.data.users);
        setPagination(response.data.pagination);
      }
    } catch (error) {
      console.error('Error fetching users:', error);
      Alert.alert('Error', 'Failed to fetch users');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchUsers(1);
  }, []);

  useEffect(() => {
    const delaySearch = setTimeout(() => {
      fetchUsers(1);
    }, 500);
    return () => clearTimeout(delaySearch);
  }, [search, roleFilter, statusFilter]);

  const handleRefresh = useCallback(() => {
    fetchStats();
    fetchUsers(1, true);
  }, [search, roleFilter, statusFilter]);

  const handlePageChange = (page) => {
    if (page >= 1 && page <= pagination.totalPages) {
      fetchUsers(page);
    }
  };

  const handleEditPress = (user) => {
    setSelectedUser(user);
    setEditRole(user.role);
    setEditIsActive(user.isActive);
    setEditModalVisible(true);
  };

  const handleDeletePress = (user) => {
    setSelectedUser(user);
    setDeleteModalVisible(true);
  };

  const handleUpdateUser = async () => {
    if (!selectedUser) return;

    setIsSubmitting(true);
    try {
      const response = await userAPI.updateUser(selectedUser._id, {
        role: editRole,
        isActive: editIsActive,
      });

      if (response.success) {
        Alert.alert('Success', 'User updated successfully');
        setEditModalVisible(false);
        fetchStats();
        fetchUsers(pagination.currentPage);
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to update user');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteUser = async () => {
    if (!selectedUser) return;

    setIsSubmitting(true);
    try {
      const response = await userAPI.deleteUser(selectedUser._id);

      if (response.success) {
        Alert.alert('Success', 'User deleted successfully');
        setDeleteModalVisible(false);
        fetchStats();
        fetchUsers(1);
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to delete user');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderUserCard = ({item}) => (
    <View style={styles.userCard}>
      <View style={styles.userAvatar}>
        <Text style={styles.avatarText}>
          {item.displayName?.charAt(0)?.toUpperCase() || 'U'}
        </Text>
      </View>
      <View style={styles.userInfo}>
        <Text style={styles.userName}>{item.displayName || 'Unknown'}</Text>
        <Text style={styles.userEmail}>{item.email}</Text>
        <View style={styles.badgeRow}>
          <View style={[styles.badge, item.role === 'admin' ? styles.adminBadge : styles.userBadge]}>
            <Text style={styles.badgeText}>{item.role === 'admin' ? t('userManagement.admin').toUpperCase() : t('userManagement.user').toUpperCase()}</Text>
          </View>
          <View style={[styles.badge, item.isActive ? styles.activeBadge : styles.inactiveBadge]}>
            <Text style={styles.badgeText}>{item.isActive ? t('userManagement.active').toUpperCase() : t('userManagement.inactive').toUpperCase()}</Text>
          </View>
        </View>
      </View>
      <View style={styles.actionButtons}>
        <TouchableOpacity
          style={styles.editButton}
          onPress={() => handleEditPress(item)}
        >
          <Text style={styles.editButtonText}>{t('common.edit')}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.deleteButton}
          onPress={() => handleDeletePress(item)}
        >
          <Text style={styles.deleteButtonText}>{t('common.delete')}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderPagination = () => (
    <View style={styles.paginationContainer}>
      <TouchableOpacity
        style={[styles.pageButton, pagination.currentPage === 1 && styles.pageButtonDisabled]}
        onPress={() => handlePageChange(pagination.currentPage - 1)}
        disabled={pagination.currentPage === 1}
      >
        <Text style={styles.pageButtonText}>{t('common.prev')}</Text>
      </TouchableOpacity>
      <Text style={styles.pageInfo}>
        {pagination.currentPage} / {pagination.totalPages}
      </Text>
      <TouchableOpacity
        style={[styles.pageButton, !pagination.hasMore && styles.pageButtonDisabled]}
        onPress={() => handlePageChange(pagination.currentPage + 1)}
        disabled={!pagination.hasMore}
      >
        <Text style={styles.pageButtonText}>{t('common.next')}</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Text style={styles.backButtonText}>{t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('userManagement.title')}</Text>
        <View style={styles.placeholder} />
      </View>

      {/* Stats Cards */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{stats.totalUsers}</Text>
          <Text style={styles.statLabel}>{t('userManagement.totalUsers')}</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statNumber, {color: '#4ade80'}]}>{stats.activeUsers}</Text>
          <Text style={styles.statLabel}>{t('userManagement.active')}</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statNumber, {color: '#e94560'}]}>{stats.adminUsers}</Text>
          <Text style={styles.statLabel}>{t('userManagement.adminUsers')}</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statNumber, {color: '#60a5fa'}]}>{stats.regularUsers}</Text>
          <Text style={styles.statLabel}>{t('userManagement.regularUsers')}</Text>
        </View>
      </View>

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          placeholder={t('userManagement.searchPlaceholder')}
          placeholderTextColor="#666"
          value={search}
          onChangeText={setSearch}
        />
      </View>

      {/* Filter Chips */}
      <View style={styles.filterContainer}>
        <TouchableOpacity
          style={[styles.filterChip, roleFilter === '' && styles.filterChipActive]}
          onPress={() => setRoleFilter('')}
        >
          <Text style={[styles.filterChipText, roleFilter === '' && styles.filterChipTextActive]}>{t('userManagement.allRoles')}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterChip, roleFilter === 'admin' && styles.filterChipActive]}
          onPress={() => setRoleFilter('admin')}
        >
          <Text style={[styles.filterChipText, roleFilter === 'admin' && styles.filterChipTextActive]}>{t('userManagement.admin')}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterChip, roleFilter === 'user' && styles.filterChipActive]}
          onPress={() => setRoleFilter('user')}
        >
          <Text style={[styles.filterChipText, roleFilter === 'user' && styles.filterChipTextActive]}>{t('userManagement.user')}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterChip, statusFilter === 'true' && styles.filterChipActive]}
          onPress={() => setStatusFilter(statusFilter === 'true' ? '' : 'true')}
        >
          <Text style={[styles.filterChipText, statusFilter === 'true' && styles.filterChipTextActive]}>{t('userManagement.active')}</Text>
        </TouchableOpacity>
      </View>

      {/* User List */}
      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#e94560" />
        </View>
      ) : (
        <FlatList
          data={users}
          renderItem={renderUserCard}
          keyExtractor={(item) => item._id}
          style={styles.userList}
          contentContainerStyle={styles.userListContent}
          refreshControl={
            <RefreshControl
              refreshing={isRefreshing}
              onRefresh={handleRefresh}
              tintColor="#e94560"
            />
          }
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>{t('userManagement.noUsers')}</Text>
            </View>
          }
          ListFooterComponent={users.length > 0 ? renderPagination : null}
        />
      )}

      {/* Edit Modal */}
      <Modal
        visible={editModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setEditModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>{t('userManagement.editUser')}</Text>
            <Text style={styles.modalUserName}>{selectedUser?.displayName}</Text>
            <Text style={styles.modalUserEmail}>{selectedUser?.email}</Text>

            <Text style={styles.modalLabel}>{t('userManagement.role')}</Text>
            <View style={styles.roleSelector}>
              <TouchableOpacity
                style={[styles.roleOption, editRole === 'user' && styles.roleOptionActive]}
                onPress={() => setEditRole('user')}
              >
                <Text style={[styles.roleOptionText, editRole === 'user' && styles.roleOptionTextActive]}>{t('userManagement.user')}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.roleOption, editRole === 'admin' && styles.roleOptionActive]}
                onPress={() => setEditRole('admin')}
              >
                <Text style={[styles.roleOptionText, editRole === 'admin' && styles.roleOptionTextActive]}>{t('userManagement.admin')}</Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.modalLabel}>{t('userManagement.status')}</Text>
            <View style={styles.roleSelector}>
              <TouchableOpacity
                style={[styles.roleOption, editIsActive && styles.roleOptionActive]}
                onPress={() => setEditIsActive(true)}
              >
                <Text style={[styles.roleOptionText, editIsActive && styles.roleOptionTextActive]}>{t('userManagement.active')}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.roleOption, !editIsActive && styles.roleOptionActive]}
                onPress={() => setEditIsActive(false)}
              >
                <Text style={[styles.roleOptionText, !editIsActive && styles.roleOptionTextActive]}>{t('userManagement.inactive')}</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => setEditModalVisible(false)}
                disabled={isSubmitting}
              >
                <Text style={styles.cancelButtonText}>{t('common.cancel')}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.saveButton}
                onPress={handleUpdateUser}
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <Text style={styles.saveButtonText}>{t('common.save')}</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        visible={deleteModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setDeleteModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>{t('userManagement.deleteUser')}</Text>
            <Text style={styles.deleteWarning}>
              {t('userManagement.deleteConfirm')}
            </Text>
            <Text style={styles.modalUserName}>{selectedUser?.displayName}</Text>
            <Text style={styles.modalUserEmail}>{selectedUser?.email}</Text>
            <Text style={styles.deleteNote}>{t('userManagement.deleteWarning')}</Text>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => setDeleteModalVisible(false)}
                disabled={isSubmitting}
              >
                <Text style={styles.cancelButtonText}>{t('common.cancel')}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.confirmDeleteButton}
                onPress={handleDeleteUser}
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <Text style={styles.confirmDeleteButtonText}>{t('common.delete')}</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 50,
    paddingHorizontal: 20,
    paddingBottom: 15,
    backgroundColor: '#16213e',
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    color: '#e94560',
    fontSize: 16,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  placeholder: {
    width: 50,
  },
  statsRow: {
    flexDirection: 'row',
    paddingHorizontal: 15,
    paddingVertical: 15,
    justifyContent: 'space-between',
  },
  statCard: {
    backgroundColor: '#16213e',
    borderRadius: 10,
    padding: 12,
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 4,
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  statLabel: {
    fontSize: 11,
    color: '#aaa',
    marginTop: 4,
  },
  searchContainer: {
    paddingHorizontal: 15,
    marginBottom: 10,
  },
  searchInput: {
    backgroundColor: '#16213e',
    borderRadius: 10,
    padding: 12,
    color: '#fff',
    fontSize: 14,
  },
  filterContainer: {
    flexDirection: 'row',
    paddingHorizontal: 15,
    marginBottom: 10,
    flexWrap: 'wrap',
  },
  filterChip: {
    backgroundColor: '#16213e',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 5,
  },
  filterChipActive: {
    backgroundColor: '#e94560',
  },
  filterChipText: {
    color: '#aaa',
    fontSize: 12,
  },
  filterChipTextActive: {
    color: '#fff',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  userList: {
    flex: 1,
  },
  userListContent: {
    paddingHorizontal: 15,
    paddingBottom: 20,
  },
  userCard: {
    backgroundColor: '#16213e',
    borderRadius: 12,
    padding: 15,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
  },
  userAvatar: {
    width: 45,
    height: 45,
    borderRadius: 22.5,
    backgroundColor: '#e94560',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  avatarText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  userInfo: {
    flex: 1,
  },
  userName: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  userEmail: {
    color: '#888',
    fontSize: 12,
    marginTop: 2,
  },
  badgeRow: {
    flexDirection: 'row',
    marginTop: 6,
  },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
    marginRight: 6,
  },
  adminBadge: {
    backgroundColor: '#e9456033',
  },
  userBadge: {
    backgroundColor: '#60a5fa33',
  },
  activeBadge: {
    backgroundColor: '#4ade8033',
  },
  inactiveBadge: {
    backgroundColor: '#f87171',
  },
  badgeText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#fff',
  },
  actionButtons: {
    alignItems: 'flex-end',
  },
  editButton: {
    backgroundColor: '#60a5fa',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    marginBottom: 6,
  },
  editButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  deleteButton: {
    backgroundColor: '#f87171',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  deleteButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  paginationContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 15,
  },
  pageButton: {
    backgroundColor: '#e94560',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 6,
  },
  pageButtonDisabled: {
    backgroundColor: '#444',
  },
  pageButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  pageInfo: {
    color: '#fff',
    fontSize: 14,
    marginHorizontal: 20,
  },
  emptyContainer: {
    padding: 40,
    alignItems: 'center',
  },
  emptyText: {
    color: '#666',
    fontSize: 16,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#16213e',
    borderRadius: 15,
    padding: 20,
    width: '85%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 15,
  },
  modalUserName: {
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
  },
  modalUserEmail: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    marginBottom: 20,
  },
  modalLabel: {
    fontSize: 14,
    color: '#aaa',
    marginBottom: 8,
  },
  roleSelector: {
    flexDirection: 'row',
    marginBottom: 15,
  },
  roleOption: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    padding: 12,
    borderRadius: 8,
    marginRight: 10,
    alignItems: 'center',
  },
  roleOptionActive: {
    backgroundColor: '#e94560',
  },
  roleOptionText: {
    color: '#888',
    fontSize: 14,
  },
  roleOptionTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  modalButtons: {
    flexDirection: 'row',
    marginTop: 20,
  },
  cancelButton: {
    flex: 1,
    backgroundColor: '#333',
    padding: 12,
    borderRadius: 8,
    marginRight: 10,
    alignItems: 'center',
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 14,
  },
  saveButton: {
    flex: 1,
    backgroundColor: '#e94560',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  deleteWarning: {
    color: '#f87171',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 15,
  },
  deleteNote: {
    color: '#666',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 10,
  },
  confirmDeleteButton: {
    flex: 1,
    backgroundColor: '#f87171',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  confirmDeleteButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});
