const express = require('express');
const router = express.Router();
const {
  updateFCMToken,
  updateNotificationSettings,
  getNotificationSettings,
  sendBroadcast,
} = require('../controllers/notificationController');
const { protect, authorize } = require('../middleware/auth');

// All routes require authentication
router.use(protect);

// User routes
router.post('/token', updateFCMToken);
router.get('/settings', getNotificationSettings);
router.put('/settings', updateNotificationSettings);

// Admin only routes
router.post('/broadcast', authorize('admin'), sendBroadcast);

module.exports = router;
