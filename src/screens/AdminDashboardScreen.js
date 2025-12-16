import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import {signOutFromGoogle} from '../config/googleAuth';
import {authAPI} from '../services/api';

export default function AdminDashboardScreen({navigation, route}) {
  const user = route.params?.user;
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalScans: 0,
    activeDevices: 0,
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleLogout = async () => {
    try {
      await authAPI.logout();
      try {
        await signOutFromGoogle();
      } catch (e) {
        // Ignore Google sign-out errors
      }
      navigation.reset({
        index: 0,
        routes: [{name: 'Login'}],
      });
    } catch (error) {
      navigation.reset({
        index: 0,
        routes: [{name: 'Login'}],
      });
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.welcomeText}>Admin Panel</Text>
        <Text style={styles.appTitle}>Coconut Health Monitor</Text>
      </View>

      <View style={styles.userCard}>
        <View style={styles.adminBadge}>
          <Text style={styles.adminBadgeText}>ADMIN</Text>
        </View>
        <Text style={styles.userName}>{user?.displayName || 'Admin'}</Text>
        <Text style={styles.userEmail}>{user?.email || ''}</Text>
      </View>

      {/* Stats Section */}
      <View style={styles.statsContainer}>
        <Text style={styles.sectionTitle}>Dashboard Statistics</Text>
        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{stats.totalUsers}</Text>
            <Text style={styles.statLabel}>Total Users</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{stats.totalScans}</Text>
            <Text style={styles.statLabel}>Total Scans</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{stats.activeDevices}</Text>
            <Text style={styles.statLabel}>Active Drones</Text>
          </View>
        </View>
      </View>

      {/* Admin Features */}
      <View style={styles.featuresContainer}>
        <Text style={styles.sectionTitle}>Admin Features</Text>

        <TouchableOpacity style={styles.featureCard}>
          <Text style={styles.featureIcon}>👥</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>User Management</Text>
            <Text style={styles.featureSubtext}>Manage users and permissions</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard}>
          <Text style={styles.featureIcon}>🚁</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>Drone Fleet</Text>
            <Text style={styles.featureSubtext}>Monitor and manage drones</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard}>
          <Text style={styles.featureIcon}>📊</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>Analytics</Text>
            <Text style={styles.featureSubtext}>View system analytics</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard}>
          <Text style={styles.featureIcon}>🌴</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>Farm Management</Text>
            <Text style={styles.featureSubtext}>Manage all farms and trees</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard}>
          <Text style={styles.featureIcon}>⚙️</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>System Settings</Text>
            <Text style={styles.featureSubtext}>Configure system parameters</Text>
          </View>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
        <Text style={styles.logoutButtonText}>Logout</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginTop: 40,
    marginBottom: 20,
  },
  welcomeText: {
    fontSize: 16,
    color: '#aaa',
  },
  appTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#e94560',
    marginTop: 5,
  },
  userCard: {
    backgroundColor: '#16213e',
    borderRadius: 15,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e94560',
  },
  adminBadge: {
    backgroundColor: '#e94560',
    paddingHorizontal: 15,
    paddingVertical: 5,
    borderRadius: 20,
    marginBottom: 10,
  },
  adminBadgeText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 12,
  },
  userName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  userEmail: {
    fontSize: 14,
    color: '#aaa',
    marginTop: 5,
  },
  statsContainer: {
    marginTop: 25,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 15,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statCard: {
    backgroundColor: '#16213e',
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 5,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#e94560',
  },
  statLabel: {
    fontSize: 12,
    color: '#aaa',
    marginTop: 5,
    textAlign: 'center',
  },
  featuresContainer: {
    marginTop: 25,
  },
  featureCard: {
    backgroundColor: '#16213e',
    borderRadius: 10,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  featureIcon: {
    fontSize: 28,
    marginRight: 15,
  },
  featureContent: {
    flex: 1,
  },
  featureText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
  },
  featureSubtext: {
    fontSize: 12,
    color: '#aaa',
    marginTop: 2,
  },
  logoutButton: {
    backgroundColor: '#e94560',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 40,
  },
  logoutButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
