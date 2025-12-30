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
import {scanAPI} from '../services/scanApi';
import {useLanguage} from '../context/LanguageContext';

export default function AdminDashboardScreen({navigation, route}) {
  const {t} = useLanguage();
  const user = route.params?.user;
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalScans: 0,
    infectedScans: 0,
    infectionRate: 0,
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      setIsLoading(true);
      const response = await scanAPI.getAnalytics(30);
      if (response.data?.overview) {
        setStats({
          totalUsers: response.data.overview.totalUsers || 0,
          totalScans: response.data.overview.totalScans || 0,
          infectedScans: response.data.overview.infectedScans || 0,
          infectionRate: response.data.overview.infectionRate || 0,
        });
      }
    } catch (error) {
      console.log('Failed to load admin stats:', error.message);
    } finally {
      setIsLoading(false);
    }
  };

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
        <Text style={styles.welcomeText}>{t('dashboard.adminPanel')}</Text>
        <Text style={styles.appTitle}>{t('common.appName')}</Text>
      </View>

      <View style={styles.userCard}>
        <View style={styles.adminBadge}>
          <Text style={styles.adminBadgeText}>{t('userManagement.admin').toUpperCase()}</Text>
        </View>
        <Text style={styles.userName}>{user?.displayName || t('userManagement.admin')}</Text>
        <Text style={styles.userEmail}>{user?.email || ''}</Text>
      </View>

      {/* Stats Section */}
      <View style={styles.statsContainer}>
        <Text style={styles.sectionTitle}>{t('dashboard.totalScans')}</Text>
        {isLoading ? (
          <ActivityIndicator size="small" color="#e94560" />
        ) : (
          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{stats.totalScans}</Text>
              <Text style={styles.statLabel}>{t('dashboard.totalScans')}</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={[styles.statNumber, {color: '#f44336'}]}>
                {stats.infectedScans}
              </Text>
              <Text style={styles.statLabel}>{t('dashboard.infectedTrees')}</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={[styles.statNumber, {color: '#4caf50'}]}>
                {parseFloat(stats.infectionRate || 0).toFixed(1)}%
              </Text>
              <Text style={styles.statLabel}>Infection Rate</Text>
            </View>
          </View>
        )}
      </View>

      {/* Admin Features */}
      <View style={styles.featuresContainer}>
        <Text style={styles.sectionTitle}>{t('adminFeatures.title')}</Text>

        <TouchableOpacity
          style={styles.featureCard}
          onPress={() => navigation.navigate('UserManagement')}
        >
          <Text style={styles.featureIcon}>👥</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('adminFeatures.userManagement')}</Text>
            <Text style={styles.featureSubtext}>{t('adminFeatures.userManagementDesc')}</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard}>
          <Text style={styles.featureIcon}>🚁</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('adminFeatures.droneFleet')}</Text>
            <Text style={styles.featureSubtext}>{t('adminFeatures.droneFleetDesc')}</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.featureCard}
          onPress={() => navigation.navigate('Analytics', {isAdmin: true})}>
          <Text style={styles.featureIcon}>📊</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('adminFeatures.analytics')}</Text>
            <Text style={styles.featureSubtext}>{t('adminFeatures.analyticsDesc')}</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={styles.featureCard}>
          <Text style={styles.featureIcon}>🌴</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('adminFeatures.farmManagement')}</Text>
            <Text style={styles.featureSubtext}>{t('adminFeatures.farmManagementDesc')}</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.featureCard}
          onPress={() => navigation.navigate('Settings')}
        >
          <Text style={styles.featureIcon}>⚙️</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('adminFeatures.settings')}</Text>
            <Text style={styles.featureSubtext}>{t('adminFeatures.settingsDesc')}</Text>
          </View>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
        <Text style={styles.logoutButtonText}>{t('auth.logout')}</Text>
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
