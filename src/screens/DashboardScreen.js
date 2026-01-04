import React, {useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
} from 'react-native';
import {signOutFromGoogle} from '../config/googleAuth';
import {authAPI} from '../services/api';
import {useLanguage} from '../context/LanguageContext';
import {syncFCMToken} from '../services/notificationService';

export default function DashboardScreen({navigation, route}) {
  const {t} = useLanguage();
  const user = route.params?.user;

  // Sync FCM token with server after login
  useEffect(() => {
    syncFCMToken().catch(err =>
      console.log('FCM token sync error:', err.message),
    );
  }, []);

  const handleLogout = async () => {
    try {
      // Clear backend token and user data from AsyncStorage
      await authAPI.logout();

      // Try to sign out from Google/Firebase (ignore errors)
      try {
        await signOutFromGoogle();
      } catch (e) {
        // Ignore Google sign-out errors
      }

      // Navigate to Login screen
      navigation.reset({
        index: 0,
        routes: [{name: 'Login'}],
      });
    } catch (error) {
      // Even if backend logout fails, clear local data and navigate
      navigation.reset({
        index: 0,
        routes: [{name: 'Login'}],
      });
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.welcomeText}>{t('dashboard.welcome')}!</Text>
        <Text style={styles.appTitle}>{t('common.appName')}</Text>
      </View>

      <View style={styles.userCard}>
        {user?.photoURL && (
          <Image source={{uri: user.photoURL}} style={styles.userPhoto} />
        )}
        <Text style={styles.userName}>{user?.displayName || t('userManagement.user')}</Text>
        <Text style={styles.userEmail}>{user?.email || ''}</Text>
      </View>

      <View style={styles.featuresContainer}>
        <Text style={styles.sectionTitle}>{t('dashboard.quickActions')}</Text>

        <TouchableOpacity
          style={styles.featureCardActive}
          onPress={() => navigation.navigate('PestDetection')}>
          <Text style={styles.featureIcon}>🐛</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('pestDetection.title')}</Text>
            <Text style={styles.featureSubtext}>{t('pestDetection.detectAll')}</Text>
          </View>
          <Text style={styles.featureArrow}>→</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.featureCardActive}
          onPress={() => navigation.navigate('LeafHealth')}>
          <Text style={styles.featureIcon}>🌿</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>Health Monitoring</Text>
            <Text style={styles.featureSubtext}>Check leaf health status</Text>
          </View>
          <Text style={styles.featureArrow}>→</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.featureCardActive}
          onPress={() => navigation.navigate('DiseaseDetection')}>
          <Text style={styles.featureIcon}>🍃</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('diseaseDetection.title')}</Text>
            <Text style={styles.featureSubtext}>{t('diseaseDetection.subtitle')}</Text>
          </View>
          <Text style={styles.featureArrow}>→</Text>
        </TouchableOpacity>

        <View style={styles.featureCard}>
          <Text style={styles.featureIcon}>🚁</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('adminFeatures.droneFleet')}</Text>
            <Text style={styles.featureSubtext}>Coming soon</Text>
          </View>
        </View>

        <TouchableOpacity
          style={styles.featureCardActive}
          onPress={() => navigation.navigate('ScanHistory')}>
          <Text style={styles.featureIcon}>📊</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('adminFeatures.analytics')}</Text>
            <Text style={styles.featureSubtext}>View scan history & stats</Text>
          </View>
          <Text style={styles.featureArrow}>→</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.featureCardActive}
          onPress={() => navigation.navigate('Chat')}>
          <Text style={styles.featureIcon}>💬</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('chat.title')}</Text>
            <Text style={styles.featureSubtext}>{t('chat.subtitle')}</Text>
          </View>
          <Text style={styles.featureArrow}>→</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.featureCardActive}
          onPress={() => navigation.navigate('Settings')}>
          <Text style={styles.featureIcon}>⚙️</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureText}>{t('settings.title')}</Text>
            <Text style={styles.featureSubtext}>{t('settings.language')}</Text>
          </View>
          <Text style={styles.featureArrow}>→</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
        <Text style={styles.logoutButtonText}>{t('auth.logout')}</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginTop: 40,
    marginBottom: 30,
  },
  welcomeText: {
    fontSize: 16,
    color: '#666',
  },
  appTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginTop: 5,
  },
  userCard: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  userPhoto: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginBottom: 10,
  },
  userName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  userEmail: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  featuresContainer: {
    marginTop: 30,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  featureCard: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
    opacity: 0.6,
  },
  featureCardActive: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderWidth: 2,
    borderColor: '#2e7d32',
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
    fontWeight: 'bold',
    color: '#333',
  },
  featureSubtext: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
  featureArrow: {
    fontSize: 20,
    color: '#2e7d32',
    fontWeight: 'bold',
  },
  logoutButton: {
    backgroundColor: '#d32f2f',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 'auto',
    marginBottom: 20,
  },
  logoutButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
