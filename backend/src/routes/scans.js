const express = require('express');
const router = express.Router();
const {
  saveScanResult,
  getUserScans,
  getScanStats,
  getScanById,
  deleteScan,
  getAnalytics,
  getInfectionTrends,
  getPestDistribution,
} = require('../controllers/scanController');
const { protect, authorize } = require('../middleware/auth');

// All routes require authentication
router.use(protect);

// User routes - personal scan history
router.post('/', saveScanResult); // Save a new scan
router.get('/my-scans', getUserScans); // Get user's scan history
router.get('/my-stats', getScanStats); // Get user's personal stats

// Admin routes - aggregate analytics (must be before /:id to avoid conflict)
router.get('/admin/analytics', authorize('admin'), getAnalytics);
router.get('/admin/trends', authorize('admin'), getInfectionTrends);
router.get('/admin/pest-distribution', authorize('admin'), getPestDistribution);

// Individual scan routes
router.get('/:id', getScanById); // Get single scan details
router.delete('/:id', (req, res, next) => {
  console.log('=== DELETE SCAN ROUTE HIT ===');
  console.log('Scan ID:', req.params.id);
  console.log('User:', req.user?._id);
  next();
}, deleteScan); // Delete a scan

module.exports = router;
