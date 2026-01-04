const User = require('../models/User');
const { updateUserFCMToken, sendBroadcastNotification } = require('../services/pushNotificationService');

/**
 * @desc    Update user's FCM token
 * @route   POST /api/notifications/token
 * @access  Private
 */
exports.updateFCMToken = async (req, res) => {
  try {
    const { fcmToken } = req.body;
    const userId = req.user._id;

    if (!fcmToken) {
      return res.status(400).json({
        success: false,
        message: 'FCM token is required',
      });
    }

    const result = await updateUserFCMToken(userId, fcmToken);

    if (result.success) {
      res.status(200).json({
        success: true,
        message: 'FCM token updated successfully',
      });
    } else {
      res.status(500).json({
        success: false,
        message: result.reason || 'Failed to update FCM token',
      });
    }
  } catch (error) {
    console.error('Update FCM token error:', error);
    res.status(500).json({
      success: false,
      message: 'Error updating FCM token',
    });
  }
};

/**
 * @desc    Toggle user notifications
 * @route   PUT /api/notifications/settings
 * @access  Private
 */
exports.updateNotificationSettings = async (req, res) => {
  try {
    const { enabled } = req.body;
    const userId = req.user._id;

    await User.findByIdAndUpdate(userId, {
      notificationsEnabled: enabled !== false,
    });

    res.status(200).json({
      success: true,
      message: `Notifications ${enabled ? 'enabled' : 'disabled'}`,
    });
  } catch (error) {
    console.error('Update notification settings error:', error);
    res.status(500).json({
      success: false,
      message: 'Error updating notification settings',
    });
  }
};

/**
 * @desc    Get user notification settings
 * @route   GET /api/notifications/settings
 * @access  Private
 */
exports.getNotificationSettings = async (req, res) => {
  try {
    const user = await User.findById(req.user._id).select('notificationsEnabled fcmToken');

    res.status(200).json({
      success: true,
      data: {
        enabled: user.notificationsEnabled,
        hasToken: !!user.fcmToken,
      },
    });
  } catch (error) {
    console.error('Get notification settings error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching notification settings',
    });
  }
};

/**
 * @desc    Send broadcast notification to all users (Admin only)
 * @route   POST /api/notifications/broadcast
 * @access  Private/Admin
 */
exports.sendBroadcast = async (req, res) => {
  try {
    const { title, body, data } = req.body;

    if (!title || !body) {
      return res.status(400).json({
        success: false,
        message: 'Title and body are required',
      });
    }

    // Get all users with FCM tokens and notifications enabled
    const users = await User.find({
      notificationsEnabled: true,
      fcmToken: { $ne: null },
    }).select('_id');

    if (users.length === 0) {
      return res.status(200).json({
        success: true,
        message: 'No users with notifications enabled',
        sent: 0,
      });
    }

    const userIds = users.map(u => u._id);
    const results = await sendBroadcastNotification(userIds, title, body, data || {});

    const successCount = results.filter(r => r.success).length;

    res.status(200).json({
      success: true,
      message: `Broadcast sent to ${successCount}/${userIds.length} users`,
      sent: successCount,
      total: userIds.length,
    });
  } catch (error) {
    console.error('Send broadcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Error sending broadcast notification',
    });
  }
};
