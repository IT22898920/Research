const admin = require('firebase-admin');
const User = require('../models/User');

let firebaseInitialized = false;

/**
 * Initialize Firebase Admin SDK
 * Requires FIREBASE_SERVICE_ACCOUNT_KEY in .env (base64 encoded JSON)
 * Or GOOGLE_APPLICATION_CREDENTIALS path to service-account.json
 */
const initializeFirebase = () => {
  if (firebaseInitialized) {
    return true;
  }

  try {
    // Try to get service account from environment variable (base64 encoded)
    if (process.env.FIREBASE_SERVICE_ACCOUNT_KEY) {
      const serviceAccount = JSON.parse(
        Buffer.from(process.env.FIREBASE_SERVICE_ACCOUNT_KEY, 'base64').toString('utf8')
      );

      admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
      });

      firebaseInitialized = true;
      console.log('Firebase Admin initialized with service account');
      return true;
    }

    // Try to use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      admin.initializeApp({
        credential: admin.credential.applicationDefault(),
      });

      firebaseInitialized = true;
      console.log('Firebase Admin initialized with application default credentials');
      return true;
    }

    console.log('Firebase Admin not initialized - no credentials provided');
    console.log('Set FIREBASE_SERVICE_ACCOUNT_KEY (base64) or GOOGLE_APPLICATION_CREDENTIALS env var');
    return false;
  } catch (error) {
    console.error('Firebase Admin initialization error:', error.message);
    return false;
  }
};

/**
 * Send push notification to a single user
 */
const sendNotification = async (fcmToken, title, body, data = {}) => {
  if (!firebaseInitialized) {
    if (!initializeFirebase()) {
      console.log('Skipping push notification - Firebase not initialized');
      return { success: false, reason: 'firebase_not_initialized' };
    }
  }

  try {
    const message = {
      notification: {
        title,
        body,
      },
      data: {
        ...data,
        click_action: 'FLUTTER_NOTIFICATION_CLICK',
      },
      android: {
        priority: 'high',
        notification: {
          channelId: 'coconut_health_alerts',
          icon: 'ic_notification',
          color: '#e94560',
        },
      },
      token: fcmToken,
    };

    const response = await admin.messaging().send(message);
    console.log('Push notification sent:', response);
    return { success: true, messageId: response };
  } catch (error) {
    console.error('Send notification error:', error.message);

    // Handle invalid token
    if (error.code === 'messaging/invalid-registration-token' ||
        error.code === 'messaging/registration-token-not-registered') {
      return { success: false, reason: 'invalid_token' };
    }

    return { success: false, reason: error.message };
  }
};

/**
 * Send notification to a user by userId
 */
const sendNotificationToUser = async (userId, title, body, data = {}) => {
  try {
    const user = await User.findById(userId);

    if (!user) {
      console.log('User not found for notification:', userId);
      return { success: false, reason: 'user_not_found' };
    }

    if (!user.notificationsEnabled) {
      console.log('Notifications disabled for user:', userId);
      return { success: false, reason: 'notifications_disabled' };
    }

    if (!user.fcmToken) {
      console.log('No FCM token for user:', userId);
      return { success: false, reason: 'no_token' };
    }

    const result = await sendNotification(user.fcmToken, title, body, data);

    // Clear invalid token from user
    if (result.reason === 'invalid_token') {
      await User.findByIdAndUpdate(userId, { fcmToken: null });
      console.log('Cleared invalid FCM token for user:', userId);
    }

    return result;
  } catch (error) {
    console.error('Send notification to user error:', error);
    return { success: false, reason: error.message };
  }
};

/**
 * Send pest detection notification
 */
const sendPestDetectionNotification = async (userId, scanData) => {
  const { pestsDetected, severity, scanId } = scanData;

  const isSevere = severity?.level === 'severe' || severity?.percent >= 70;
  const pestName = pestsDetected?.includes('coconut_mite')
    ? 'Coconut Mite'
    : 'Caterpillar';

  const title = isSevere
    ? 'Severe Infection Detected!'
    : 'Pest Detected';

  const body = `${pestName} detected in your scan. ${
    isSevere
      ? 'Immediate treatment recommended!'
      : 'Tap to view treatment options.'
  }`;

  const data = {
    type: isSevere ? 'severe_infection' : 'pest_detected',
    scanId: scanId?.toString() || '',
    pestType: pestName,
    severity: severity?.level || 'moderate',
  };

  return sendNotificationToUser(userId, title, body, data);
};

/**
 * Send notification to multiple users (admin broadcast)
 */
const sendBroadcastNotification = async (userIds, title, body, data = {}) => {
  const results = [];

  for (const userId of userIds) {
    const result = await sendNotificationToUser(userId, title, body, data);
    results.push({ userId, ...result });
  }

  return results;
};

/**
 * Update user's FCM token
 */
const updateUserFCMToken = async (userId, fcmToken) => {
  try {
    await User.findByIdAndUpdate(userId, { fcmToken });
    console.log('FCM token updated for user:', userId);
    return { success: true };
  } catch (error) {
    console.error('Update FCM token error:', error);
    return { success: false, reason: error.message };
  }
};

// Initialize Firebase on module load
initializeFirebase();

module.exports = {
  initializeFirebase,
  sendNotification,
  sendNotificationToUser,
  sendPestDetectionNotification,
  sendBroadcastNotification,
  updateUserFCMToken,
};
