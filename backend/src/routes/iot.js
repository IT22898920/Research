const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const {
  registerDevice,
  getMyDevices,
  updateDevice,
  deleteDevice,
  regenerateDeviceKey,
  receiveLocation,
  receiveLiveLocation,
  getDeviceLiveLocation,
  getDeviceLocations,
} = require('../controllers/iotController');

// ============================================
// ESP32 Device Endpoints (device auth via headers)
// ============================================
router.post('/location', receiveLocation);           // POST /api/iot/location (save tree)
router.post('/live-location', receiveLiveLocation);   // POST /api/iot/live-location (tracking)

// ============================================
// Device Management (user JWT auth required)
// ============================================
router.use(protect); // All routes below require JWT

router.route('/devices')
  .get(getMyDevices)        // GET /api/iot/devices
  .post(registerDevice);    // POST /api/iot/devices

router.route('/devices/:id')
  .put(updateDevice)        // PUT /api/iot/devices/:id
  .delete(deleteDevice);    // DELETE /api/iot/devices/:id

router.post('/devices/:id/regenerate-key', regenerateDeviceKey);
router.get('/devices/:id/live', getDeviceLiveLocation);    // GET live location
router.get('/devices/:id/locations', getDeviceLocations);

module.exports = router;
