const crypto = require('crypto');
const IoTDevice = require('../models/IoTDevice');
const Tree = require('../models/Tree');
const Plantation = require('../models/Plantation');

/**
 * @desc    Register a new IoT device
 * @route   POST /api/iot/devices
 * @access  Private (authenticated user)
 */
exports.registerDevice = async (req, res) => {
  try {
    const { deviceId, name, defaultPlantation } = req.body;

    if (!deviceId) {
      return res.status(400).json({
        success: false,
        message: 'Device ID is required',
      });
    }

    // Check if device already exists
    const existing = await IoTDevice.findOne({ deviceId });
    if (existing) {
      return res.status(400).json({
        success: false,
        message: 'Device with this ID already exists',
      });
    }

    // Verify plantation if provided
    if (defaultPlantation) {
      const plantation = await Plantation.findOne({
        _id: defaultPlantation,
        owner: req.user._id,
      });
      if (!plantation) {
        return res.status(404).json({
          success: false,
          message: 'Plantation not found',
        });
      }
    }

    // Generate device key
    const deviceKey = crypto.randomBytes(32).toString('hex');

    const device = await IoTDevice.create({
      deviceId,
      name: name || `GPS Device ${deviceId}`,
      owner: req.user._id,
      deviceKey,
      defaultPlantation,
    });

    res.status(201).json({
      success: true,
      message: 'Device registered successfully',
      data: {
        _id: device._id,
        deviceId: device.deviceId,
        name: device.name,
        deviceKey, // Only returned once at registration!
        defaultPlantation: device.defaultPlantation,
      },
    });
  } catch (error) {
    console.error('Register device error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to register device',
    });
  }
};

/**
 * @desc    Get user's IoT devices
 * @route   GET /api/iot/devices
 * @access  Private
 */
exports.getMyDevices = async (req, res) => {
  try {
    // First try user's own devices, then show all active devices
    let devices = await IoTDevice.find({ owner: req.user._id })
      .populate('defaultPlantation', 'name')
      .select('-deviceKey')
      .sort('-createdAt');

    // If user has no devices, show all available devices (shared access)
    if (devices.length === 0) {
      devices = await IoTDevice.find({ isActive: true })
        .populate('defaultPlantation', 'name')
        .select('-deviceKey')
        .sort('-createdAt');
    }

    res.json({
      success: true,
      data: devices,
    });
  } catch (error) {
    console.error('Get devices error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get devices',
    });
  }
};

/**
 * @desc    Update IoT device
 * @route   PUT /api/iot/devices/:id
 * @access  Private
 */
exports.updateDevice = async (req, res) => {
  try {
    const { name, defaultPlantation, isActive } = req.body;

    const device = await IoTDevice.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!device) {
      return res.status(404).json({
        success: false,
        message: 'Device not found',
      });
    }

    if (name !== undefined) device.name = name;
    if (isActive !== undefined) device.isActive = isActive;

    if (defaultPlantation !== undefined) {
      if (defaultPlantation) {
        const plantation = await Plantation.findOne({
          _id: defaultPlantation,
          owner: req.user._id,
        });
        if (!plantation) {
          return res.status(404).json({
            success: false,
            message: 'Plantation not found',
          });
        }
      }
      device.defaultPlantation = defaultPlantation;
    }

    await device.save();

    res.json({
      success: true,
      message: 'Device updated successfully',
      data: device,
    });
  } catch (error) {
    console.error('Update device error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to update device',
    });
  }
};

/**
 * @desc    Delete IoT device
 * @route   DELETE /api/iot/devices/:id
 * @access  Private
 */
exports.deleteDevice = async (req, res) => {
  try {
    const device = await IoTDevice.findOneAndDelete({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!device) {
      return res.status(404).json({
        success: false,
        message: 'Device not found',
      });
    }

    res.json({
      success: true,
      message: 'Device deleted successfully',
    });
  } catch (error) {
    console.error('Delete device error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to delete device',
    });
  }
};

/**
 * @desc    Regenerate device key
 * @route   POST /api/iot/devices/:id/regenerate-key
 * @access  Private
 */
exports.regenerateDeviceKey = async (req, res) => {
  try {
    const device = await IoTDevice.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!device) {
      return res.status(404).json({
        success: false,
        message: 'Device not found',
      });
    }

    const newKey = crypto.randomBytes(32).toString('hex');
    device.deviceKey = newKey;
    await device.save();

    res.json({
      success: true,
      message: 'Device key regenerated',
      data: {
        deviceId: device.deviceId,
        deviceKey: newKey,
      },
    });
  } catch (error) {
    console.error('Regenerate key error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to regenerate key',
    });
  }
};

/**
 * @desc    Receive GPS location from IoT device (ESP32)
 * @route   POST /api/iot/location
 * @access  Device Auth (X-Device-Id + X-Device-Key headers)
 */
exports.receiveLocation = async (req, res) => {
  try {
    const deviceId = req.headers['x-device-id'];
    const deviceKey = req.headers['x-device-key'];

    if (!deviceId || !deviceKey) {
      return res.status(401).json({
        success: false,
        message: 'Device credentials required (X-Device-Id, X-Device-Key headers)',
      });
    }

    // Authenticate device
    const device = await IoTDevice.findOne({
      deviceId,
      deviceKey,
      isActive: true,
    });

    if (!device) {
      return res.status(401).json({
        success: false,
        message: 'Invalid device credentials',
      });
    }

    const {
      latitude,
      longitude,
      accuracy,
      satellites,
      hdop,
      altitude,
      treeLabel,
      plantationId,
    } = req.body;

    // Validate GPS data
    if (latitude === undefined || longitude === undefined) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required',
      });
    }

    if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
      return res.status(400).json({
        success: false,
        message: 'Invalid coordinates',
      });
    }

    // Use provided plantation or device's default
    const targetPlantation = plantationId || device.defaultPlantation;

    if (!targetPlantation) {
      // No plantation specified — just save the location reading
      device.lastSeenAt = new Date();
      device.lastLocation = { latitude, longitude, accuracy };
      device.totalReadings += 1;
      await device.save();

      return res.json({
        success: true,
        message: 'Location received (no plantation set — location saved to device only)',
        data: {
          latitude,
          longitude,
          accuracy,
          savedToTree: false,
        },
      });
    }

    // Verify plantation belongs to device owner
    const plantation = await Plantation.findOne({
      _id: targetPlantation,
      owner: device.owner,
    });

    if (!plantation) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    // Create tree with IoT GPS data
    const label = treeLabel || `IoT-${Date.now().toString(36).toUpperCase()}`;

    const tree = await Tree.create({
      plantation: targetPlantation,
      owner: device.owner,
      label,
      latitude,
      longitude,
      location: {
        type: 'Point',
        coordinates: [longitude, latitude],
      },
      accuracy: accuracy || null,
      locationName: `IoT Device (${device.name})`,
      notes: [
        `Captured by IoT device: ${device.deviceId}`,
        satellites ? `Satellites: ${satellites}` : null,
        hdop ? `HDOP: ${hdop}` : null,
        altitude ? `Altitude: ${altitude}m` : null,
      ]
        .filter(Boolean)
        .join(' | '),
      gpsSource: 'iot_device',
      gpsDeviceId: device.deviceId,
      gpsSatellites: satellites,
      gpsHdop: hdop,
    });

    // Update device stats
    device.lastSeenAt = new Date();
    device.lastLocation = { latitude, longitude, accuracy };
    device.totalReadings += 1;
    await device.save();

    res.status(201).json({
      success: true,
      message: 'Tree location saved successfully',
      data: {
        treeId: tree._id,
        label: tree.label,
        latitude,
        longitude,
        accuracy,
        satellites,
        plantation: plantation.name,
        savedToTree: true,
      },
    });
  } catch (error) {
    console.error('Receive location error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to save location',
    });
  }
};

/**
 * @desc    Receive live GPS location from IoT device (continuous tracking)
 * @route   POST /api/iot/live-location
 * @access  Device Auth (X-Device-Id + X-Device-Key headers)
 */
exports.receiveLiveLocation = async (req, res) => {
  try {
    const deviceId = req.headers['x-device-id'];
    const deviceKey = req.headers['x-device-key'];

    if (!deviceId || !deviceKey) {
      return res.status(401).json({
        success: false,
        message: 'Device credentials required',
      });
    }

    const device = await IoTDevice.findOne({
      deviceId,
      deviceKey,
      isActive: true,
    });

    if (!device) {
      return res.status(401).json({
        success: false,
        message: 'Invalid device credentials',
      });
    }

    const { latitude, longitude, accuracy, satellites, hdop, altitude } = req.body;

    if (latitude === undefined || longitude === undefined) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required',
      });
    }

    // Update device live location — no tree creation, just tracking
    device.lastSeenAt = new Date();
    device.lastLocation = {
      latitude,
      longitude,
      accuracy: accuracy || null,
    };
    device.liveData = {
      satellites: satellites || 0,
      hdop: hdop || 99,
      altitude: altitude || 0,
      updatedAt: new Date(),
    };
    await device.save();

    res.json({
      success: true,
      message: 'Live location updated',
    });
  } catch (error) {
    console.error('Live location error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to update live location',
    });
  }
};

/**
 * @desc    Get live location of a device
 * @route   GET /api/iot/devices/:id/live
 * @access  Private (authenticated user)
 */
exports.getDeviceLiveLocation = async (req, res) => {
  try {
    let device = await IoTDevice.findOne({ _id: req.params.id, owner: req.user._id });
    if (!device) {
      device = await IoTDevice.findOne({ _id: req.params.id, isActive: true });
    }

    if (!device) {
      return res.status(404).json({
        success: false,
        message: 'Device not found',
      });
    }

    // Check if device data is fresh (within last 30 seconds)
    const isOnline = device.lastSeenAt &&
      (Date.now() - new Date(device.lastSeenAt).getTime()) < 30000;

    res.json({
      success: true,
      data: {
        deviceId: device.deviceId,
        name: device.name,
        isOnline,
        location: device.lastLocation || null,
        liveData: device.liveData || null,
        lastSeenAt: device.lastSeenAt,
      },
    });
  } catch (error) {
    console.error('Get live location error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get live location',
    });
  }
};

/**
 * @desc    Get recent locations from a device
 * @route   GET /api/iot/devices/:id/locations
 * @access  Private
 */
exports.getDeviceLocations = async (req, res) => {
  try {
    let device = await IoTDevice.findOne({ _id: req.params.id, owner: req.user._id });
    if (!device) {
      device = await IoTDevice.findOne({ _id: req.params.id, isActive: true });
    }

    if (!device) {
      return res.status(404).json({
        success: false,
        message: 'Device not found',
      });
    }

    // Get trees added by this device
    const trees = await Tree.find({
      owner: req.user._id,
      gpsDeviceId: device.deviceId,
      isActive: true,
    })
      .populate('plantation', 'name')
      .sort('-createdAt')
      .limit(50);

    res.json({
      success: true,
      data: trees,
      device: {
        deviceId: device.deviceId,
        name: device.name,
        lastSeenAt: device.lastSeenAt,
        totalReadings: device.totalReadings,
      },
    });
  } catch (error) {
    console.error('Get device locations error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get locations',
    });
  }
};
