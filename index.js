/**
 * @format
 */

import { AppRegistry } from 'react-native';
import App from './App';
import { name as appName } from './app.json';
import messaging from '@react-native-firebase/messaging';
import { displayNotification } from './src/services/notificationService';

// Background message handler must be registered at root level
try {
  messaging().setBackgroundMessageHandler(async remoteMessage => {
    try {
      await displayNotification(remoteMessage);
    } catch (e) {}
  });
} catch (e) {}

AppRegistry.registerComponent(appName, () => App);
