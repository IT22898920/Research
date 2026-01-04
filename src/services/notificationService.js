import messaging from '@react-native-firebase/messaging';
import notifee, {AndroidImportance, AndroidStyle} from '@notifee/react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const FCM_TOKEN_KEY = '@fcm_token';

// Notification channel for Android
const CHANNEL_ID = 'coconut_health_alerts';

/**
 * Initialize notification service
 */
export const initializeNotifications = async () => {
  try {
    // Create Android notification channel
    const channelId = await notifee.createChannel({
      id: CHANNEL_ID,
      name: 'Coconut Health Alerts',
      description: 'Pest detection and health monitoring alerts',
      importance: AndroidImportance.HIGH,
      sound: 'default',
      vibration: true,
      lights: true,
      badge: true,
    });
    console.log('📣 Notification channel created:', channelId);

    // Request permission
    const hasPermission = await requestNotificationPermission();
    if (!hasPermission) {
      console.log('Notification permission denied');
      return null;
    }

    // Get FCM token
    const token = await getFCMToken();
    console.log('FCM Token:', token);

    // Set up message handlers
    setupMessageHandlers();

    return token;
  } catch (error) {
    console.error('Error initializing notifications:', error);
    return null;
  }
};

/**
 * Request notification permission
 */
export const requestNotificationPermission = async () => {
  try {
    const authStatus = await messaging().requestPermission();
    const enabled =
      authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
      authStatus === messaging.AuthorizationStatus.PROVISIONAL;

    if (enabled) {
      console.log('Notification permission granted');
    }

    return enabled;
  } catch (error) {
    console.error('Permission request error:', error);
    return false;
  }
};

/**
 * Get FCM token
 */
export const getFCMToken = async () => {
  try {
    // Check if we have a stored token
    let token = await AsyncStorage.getItem(FCM_TOKEN_KEY);

    if (!token) {
      // Get new token from Firebase
      token = await messaging().getToken();
      if (token) {
        await AsyncStorage.setItem(FCM_TOKEN_KEY, token);
      }
    }

    return token;
  } catch (error) {
    console.error('Error getting FCM token:', error);
    return null;
  }
};

/**
 * Set up message handlers for foreground and background
 */
const setupMessageHandlers = () => {
  // Handle foreground messages
  messaging().onMessage(async remoteMessage => {
    console.log('Foreground message received:', remoteMessage);
    await displayNotification(remoteMessage);
  });

  // Handle background messages (when app is in background)
  messaging().setBackgroundMessageHandler(async remoteMessage => {
    console.log('Background message received:', remoteMessage);
    await displayNotification(remoteMessage);
  });

  // Handle notification opened (when user taps notification)
  messaging().onNotificationOpenedApp(remoteMessage => {
    console.log('Notification opened app:', remoteMessage);
    handleNotificationAction(remoteMessage);
  });

  // Check if app was opened from a notification
  messaging()
    .getInitialNotification()
    .then(remoteMessage => {
      if (remoteMessage) {
        console.log('App opened from notification:', remoteMessage);
        handleNotificationAction(remoteMessage);
      }
    });
};

/**
 * Display a local notification
 */
export const displayNotification = async remoteMessage => {
  try {
    const {notification, data} = remoteMessage;
    console.log('📢 Displaying notification:', notification?.title);

    // Determine notification style based on type
    const notificationType = data?.type || 'general';
    const style = getNotificationStyle(notificationType, data);

    const notificationId = await notifee.displayNotification({
      title: notification?.title || 'Coconut Health Monitor',
      body: notification?.body || '',
      android: {
        channelId: CHANNEL_ID,
        importance: AndroidImportance.HIGH,
        pressAction: {
          id: 'default',
        },
        smallIcon: 'ic_notification',
        color: '#e94560',
        ...style,
      },
      data: data,
    });
    console.log('✅ Notification displayed successfully, ID:', notificationId);
  } catch (error) {
    console.error('❌ Error displaying notification:', error);
  }
};

/**
 * Get notification style based on type
 */
const getNotificationStyle = (type, data) => {
  switch (type) {
    case 'pest_detected':
      return {
        style: {
          type: AndroidStyle.BIGTEXT,
          text: data?.details || 'Pest detected in your scan. Tap to view details.',
        },
      };
    case 'severe_infection':
      return {
        color: '#f44336',
        style: {
          type: AndroidStyle.BIGTEXT,
          text: data?.details || 'Severe infection detected! Immediate action recommended.',
        },
      };
    case 'scan_reminder':
      return {
        style: {
          type: AndroidStyle.BIGTEXT,
          text: 'Regular scanning helps maintain tree health.',
        },
      };
    case 'treatment_reminder':
      return {
        style: {
          type: AndroidStyle.BIGTEXT,
          text: data?.details || 'Time for treatment follow-up.',
        },
      };
    default:
      return {};
  }
};

/**
 * Handle notification action (when user taps)
 */
const handleNotificationAction = remoteMessage => {
  const {data} = remoteMessage;

  if (data?.scanId) {
    // Navigate to scan details
    // This will be handled by navigation in the app
    console.log('Navigate to scan:', data.scanId);
  }
};

/**
 * Show local pest detection notification
 */
export const showPestDetectionNotification = async (pestType, severity, scanId) => {
  const title = severity === 'severe'
    ? 'Severe Infection Detected!'
    : 'Pest Detected';

  const body = `${pestType} detected in your scan. ${
    severity === 'severe'
      ? 'Immediate treatment recommended!'
      : 'Tap to view treatment options.'
  }`;

  await notifee.displayNotification({
    title,
    body,
    android: {
      channelId: CHANNEL_ID,
      importance: AndroidImportance.HIGH,
      pressAction: {
        id: 'default',
      },
      smallIcon: 'ic_notification',
      color: severity === 'severe' ? '#f44336' : '#e94560',
      style: {
        type: AndroidStyle.BIGTEXT,
        text: body,
      },
    },
    data: {
      type: severity === 'severe' ? 'severe_infection' : 'pest_detected',
      scanId,
      pestType,
    },
  });
};

/**
 * Show scan reminder notification
 */
export const showScanReminderNotification = async () => {
  await notifee.displayNotification({
    title: 'Time to Scan!',
    body: 'Regular scanning helps detect pests early. Scan your coconut trees today.',
    android: {
      channelId: CHANNEL_ID,
      importance: AndroidImportance.DEFAULT,
      pressAction: {
        id: 'default',
      },
      smallIcon: 'ic_notification',
      color: '#4caf50',
    },
    data: {
      type: 'scan_reminder',
    },
  });
};

/**
 * Show treatment reminder notification
 */
export const showTreatmentReminderNotification = async (treatmentType, daysAgo) => {
  await notifee.displayNotification({
    title: 'Treatment Follow-up',
    body: `It's been ${daysAgo} days since your ${treatmentType} treatment. Time to check your trees!`,
    android: {
      channelId: CHANNEL_ID,
      importance: AndroidImportance.DEFAULT,
      pressAction: {
        id: 'default',
      },
      smallIcon: 'ic_notification',
      color: '#ff9800',
    },
    data: {
      type: 'treatment_reminder',
    },
  });
};

/**
 * Schedule a daily scan reminder
 */
export const scheduleDailyScanReminder = async (hour = 9, minute = 0) => {
  // Cancel existing reminder
  await notifee.cancelAllNotifications();

  // Create trigger for daily notification
  const trigger = {
    type: notifee.TriggerType.TIMESTAMP,
    timestamp: getNextTriggerTime(hour, minute),
    repeatFrequency: notifee.RepeatFrequency.DAILY,
  };

  await notifee.createTriggerNotification(
    {
      title: 'Daily Scan Reminder',
      body: 'Good morning! Start your day by checking your coconut trees.',
      android: {
        channelId: CHANNEL_ID,
        smallIcon: 'ic_notification',
        color: '#4caf50',
      },
    },
    trigger,
  );

  console.log('Daily reminder scheduled for', hour, ':', minute);
};

/**
 * Get next trigger time for scheduled notification
 */
const getNextTriggerTime = (hour, minute) => {
  const now = new Date();
  const trigger = new Date();
  trigger.setHours(hour, minute, 0, 0);

  // If time has passed today, schedule for tomorrow
  if (trigger <= now) {
    trigger.setDate(trigger.getDate() + 1);
  }

  return trigger.getTime();
};

/**
 * Cancel all scheduled notifications
 */
export const cancelAllNotifications = async () => {
  await notifee.cancelAllNotifications();
};

/**
 * Update FCM token on server
 */
export const updateFCMTokenOnServer = async token => {
  try {
    // Import dynamically to avoid circular dependencies
    const {notificationAPI} = require('./scanApi');
    const result = await notificationAPI.updateFCMToken(token);
    console.log('FCM token updated on server:', result.success);
    return result.success;
  } catch (error) {
    console.error('Error updating FCM token on server:', error.message);
    return false;
  }
};

/**
 * Sync FCM token with server (call after user login)
 */
export const syncFCMToken = async () => {
  try {
    const token = await AsyncStorage.getItem(FCM_TOKEN_KEY);
    if (token) {
      return await updateFCMTokenOnServer(token);
    }
    return false;
  } catch (error) {
    console.error('Error syncing FCM token:', error.message);
    return false;
  }
};

export default {
  initializeNotifications,
  requestNotificationPermission,
  syncFCMToken,
  getFCMToken,
  displayNotification,
  showPestDetectionNotification,
  showScanReminderNotification,
  showTreatmentReminderNotification,
  scheduleDailyScanReminder,
  cancelAllNotifications,
};
