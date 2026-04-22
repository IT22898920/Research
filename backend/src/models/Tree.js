const mongoose = require('mongoose');

// Health record sub-schema
const healthRecordSchema = new mongoose.Schema(
  {
    status: {
      type: String,
      enum: ['healthy', 'unhealthy', 'infected'],
      required: true,
    },
    scanType: {
      type: String,
      trim: true,
    },
    confidence: {
      type: Number,
      min: 0,
      max: 1,
    },
    details: {
      type: mongoose.Schema.Types.Mixed,
    },
    scannedAt: {
      type: Date,
      default: Date.now,
    },
  },
  { _id: true }
);

// Yield record sub-schema
const yieldRecordSchema = new mongoose.Schema(
  {
    count: {
      type: Number,
      min: 0,
    },
    estimatedAt: {
      type: Date,
      default: Date.now,
    },
    notes: {
      type: String,
      maxlength: 200,
    },
  },
  { _id: true }
);

const treeSchema = new mongoose.Schema(
  {
    // Relationships
    plantation: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Plantation',
      required: [true, 'Plantation is required'],
      index: true,
    },
    owner: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Owner is required'],
      index: true,
    },

    // Tree identification
    label: {
      type: String,
      required: [true, 'Tree label is required'],
      trim: true,
      maxlength: [50, 'Label cannot exceed 50 characters'],
    },

    // GPS Location (GeoJSON format for geospatial queries)
    location: {
      type: {
        type: String,
        enum: ['Point'],
        default: 'Point',
      },
      coordinates: {
        type: [Number], // [longitude, latitude]
        required: true,
      },
    },

    // Explicit lat/lng for easier access
    latitude: {
      type: Number,
      required: [true, 'Latitude is required'],
      min: -90,
      max: 90,
    },
    longitude: {
      type: Number,
      required: [true, 'Longitude is required'],
      min: -180,
      max: 180,
    },

    // GPS accuracy in meters
    accuracy: {
      type: Number,
      min: 0,
    },

    // Human-readable location name
    locationName: {
      type: String,
      trim: true,
      maxlength: 100,
    },

    // Notes
    notes: {
      type: String,
      maxlength: [500, 'Notes cannot exceed 500 characters'],
    },

    // Health tracking
    lastHealthStatus: {
      type: String,
      enum: ['healthy', 'unhealthy', 'infected', null],
      default: null,
    },
    healthHistory: [healthRecordSchema],

    // Scan results
    nutCount: {
      type: Number,
      min: 0,
      default: null,
    },
    bunchCount: {
      type: Number,
      min: 0,
      default: null,
    },
    pestCount: {
      type: Number,
      min: 0,
      default: null,
    },
    detectedIssues: [
      {
        type: String,
        trim: true,
      },
    ],
    lastScanAt: {
      type: Date,
    },

    // Yield tracking
    yieldHistory: [yieldRecordSchema],

    // IoT Device tracking
    gpsSource: {
      type: String,
      enum: ['phone', 'iot_device', 'manual', null],
      default: null,
    },
    gpsDeviceId: {
      type: String,
      trim: true,
    },
    gpsSatellites: {
      type: Number,
      min: 0,
    },
    gpsHdop: {
      type: Number,
      min: 0,
    },

    // Status
    isActive: {
      type: Boolean,
      default: true,
    },
  },
  {
    timestamps: true,
  }
);

// Indexes for performance
treeSchema.index({ 'location.coordinates': '2dsphere' });
treeSchema.index({ plantation: 1, label: 1 });
treeSchema.index({ owner: 1, createdAt: -1 });
treeSchema.index({ lastHealthStatus: 1 });
treeSchema.index({ plantation: 1, lastHealthStatus: 1 });
treeSchema.index({ plantation: 1, isActive: 1 });

// Pre-save hook to sync GeoJSON coordinates from lat/lng
treeSchema.pre('save', function () {
  if (this.latitude !== undefined && this.longitude !== undefined) {
    this.location = {
      type: 'Point',
      coordinates: [this.longitude, this.latitude],
    };
  }
});

// Pre-update hook for findOneAndUpdate
treeSchema.pre('findOneAndUpdate', function () {
  const update = this.getUpdate();
  if (update.latitude !== undefined && update.longitude !== undefined) {
    update.location = {
      type: 'Point',
      coordinates: [update.longitude, update.latitude],
    };
  }
});

// Static method to update plantation stats after tree changes
treeSchema.statics.updatePlantationStats = async function (plantationId) {
  const Plantation = mongoose.model('Plantation');

  const stats = await this.aggregate([
    { $match: { plantation: plantationId, isActive: true } },
    {
      $group: {
        _id: null,
        totalTrees: { $sum: 1 },
        healthyTrees: {
          $sum: { $cond: [{ $eq: ['$lastHealthStatus', 'healthy'] }, 1, 0] },
        },
        unhealthyTrees: {
          $sum: { $cond: [{ $eq: ['$lastHealthStatus', 'unhealthy'] }, 1, 0] },
        },
        infectedTrees: {
          $sum: { $cond: [{ $eq: ['$lastHealthStatus', 'infected'] }, 1, 0] },
        },
        notScannedTrees: {
          $sum: { $cond: [{ $eq: ['$lastHealthStatus', null] }, 1, 0] },
        },
      },
    },
  ]);

  const defaultStats = {
    totalTrees: 0,
    healthyTrees: 0,
    unhealthyTrees: 0,
    infectedTrees: 0,
    notScannedTrees: 0,
  };

  await Plantation.findByIdAndUpdate(plantationId, {
    stats: stats.length > 0 ? stats[0] : defaultStats,
  });
};

// Post-save hook to update plantation stats
treeSchema.post('save', async function () {
  await this.constructor.updatePlantationStats(this.plantation);
});

// Post-remove hook to update plantation stats
treeSchema.post('findOneAndDelete', async function (doc) {
  if (doc) {
    await mongoose.model('Tree').updatePlantationStats(doc.plantation);
  }
});

const Tree = mongoose.model('Tree', treeSchema);

module.exports = Tree;
