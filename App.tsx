/**
 * Coconut Health Monitor App
 * AI-Powered Drone-Based System for Coconut Tree Health Monitoring
 */

import React, {useEffect} from 'react';
import {StatusBar} from 'react-native';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';

import {LanguageProvider} from './src/context/LanguageContext';
import {initializeNotifications} from './src/services/notificationService';
import LoginScreen from './src/screens/LoginScreen';
import SignupScreen from './src/screens/SignupScreen';
import DashboardScreen from './src/screens/DashboardScreen';
import AdminDashboardScreen from './src/screens/AdminDashboardScreen';
import PestDetectionScreen from './src/screens/PestDetectionScreen';
import LeafHealthScreen from './src/screens/LeafHealthScreen';
import UserManagementScreen from './src/screens/UserManagementScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import ScanHistoryScreen from './src/screens/ScanHistoryScreen';
import AnalyticsScreen from './src/screens/AnalyticsScreen';
import ScanDetailScreen from './src/screens/ScanDetailScreen';
import ChatScreen from './src/screens/ChatScreen';
import DiseaseDetectionScreen from './src/screens/DiseaseDetectionScreen';

const Stack = createNativeStackNavigator();

function App(): React.JSX.Element {

  console.log('=== App component rendering ===');

  console.log('=== About to render SafeAreaProvider ===');
  useEffect(() => {
    // Initialize push notifications
    initializeNotifications()
      .then(token => {
        if (token) {
          console.log('Notifications initialized with token:', token.substring(0, 20) + '...');
        }
      })
      .catch(err => console.log('Notification init error:', err));
  }, []);


  return (
    <LanguageProvider>
      <SafeAreaProvider>
        <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
        {console.log('=== About to render NavigationContainer ===')}
        <NavigationContainer
          onReady={() => console.log('=== NavigationContainer ready ===')}
          onStateChange={() => console.log('=== Navigation state changed ===')}>
          {console.log('=== About to render Stack.Navigator ===')}
          <Stack.Navigator
            initialRouteName="Login"
            screenOptions={{
              headerShown: false,
            }}>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Signup" component={SignupScreen} />
            <Stack.Screen name="Dashboard" component={DashboardScreen} />
            <Stack.Screen name="AdminDashboard" component={AdminDashboardScreen} />
            <Stack.Screen name="PestDetection" component={PestDetectionScreen} />
            <Stack.Screen name="LeafHealth" component={LeafHealthScreen} />
            <Stack.Screen name="UserManagement" component={UserManagementScreen} />
            <Stack.Screen name="Settings" component={SettingsScreen} />
            <Stack.Screen name="ScanHistory" component={ScanHistoryScreen} />
            <Stack.Screen name="ScanDetail" component={ScanDetailScreen} />
            <Stack.Screen name="Analytics" component={AnalyticsScreen} />
            <Stack.Screen name="Chat" component={ChatScreen} />
            <Stack.Screen name="DiseaseDetection" component={DiseaseDetectionScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </SafeAreaProvider>
    </LanguageProvider>
  );
}

export default App;
