const mongoose = require('mongoose');

const iotDeviceSchema = new mongoose.Schema(
  {
    // Device identification
    deviceId: {
      type: String,
      required: [true, 'Device ID is required'],
      unique: true,
      trim: true,
      maxlength: 50,
    },
    name: {
      type: String,
      trim: true,
      maxlength: 100,
      default: 'GPS Device',
    },

    // Owner
    owner: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Owner is required'],
      index: true,
    },

    // Device secret key for authentication
    deviceKey: {
      type: String,
      required: [true, 'Device key is required'],
    },

    // Default plantation to add trees to
    defaultPlantation: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Plantation',
    },

    // Device status
    isActive: {
      type: Boolean,
      default: true,
    },
    lastSeenAt: {
      type: Date,
    },
    lastLocation: {
      latitude: Number,
      longitude: Number,
      accuracy: Number,
    },

    // Live tracking data
    liveData: {
      satellites: Number,
      hdop: Number,
      altitude: Number,
      updatedAt: Date,
    },

    // Stats
    totalReadings: {
      type: Number,
      default: 0,
    },
  },
  {
    timestamps: true,
  }
);

iotDeviceSchema.index({ deviceId: 1, deviceKey: 1 });
iotDeviceSchema.index({ owner: 1, isActive: 1 });

const IoTDevice = mongoose.model('IoTDevice', iotDeviceSchema);

module.exports = IoTDevice;
