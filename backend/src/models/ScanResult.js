const mongoose = require('mongoose');

const scanResultSchema = new mongoose.Schema(
  {
    // User who performed the scan
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },

    // Scan type: 'mite', 'caterpillar', 'white_fly', or 'all'
    scanType: {
      type: String,
      enum: ['mite', 'caterpillar', 'white_fly', 'all'],
      required: true,
    },

    // Overall scan result
    isInfected: {
      type: Boolean,
      required: true,
    },

    // Whether the image was valid (coconut image)
    isValidImage: {
      type: Boolean,
      default: true,
    },

    // Individual pest results
    results: {
      mite: {
        detected: Boolean,
        confidence: Number,
        class: String, // 'coconut_mite', 'healthy', 'not_coconut'
      },
      caterpillar: {
        detected: Boolean,
        confidence: Number,
        class: String, // 'caterpillar', 'healthy', 'not_coconut'
      },
      white_fly: {
        detected: Boolean,
        confidence: Number,
        class: String, // 'white_fly', 'healthy', 'not_coconut'
      },
    },

    // Severity information
    severity: {
      level: {
        type: String,
        enum: ['mild', 'moderate', 'severe', null],
      },
      percent: Number,
    },

    // Pests detected (array for 'all' scan type)
    pestsDetected: [
      {
        type: String,
        enum: ['coconut_mite', 'caterpillar', 'white_fly'],
      },
    ],

    // Treatment was requested
    treatmentRequested: {
      type: Boolean,
      default: false,
    },

    // Device information
    deviceInfo: {
      platform: String,
      model: String,
    },

    // Image stored in Cloudinary
    imageUrl: {
      type: String,
      default: null,
    },
    imagePublicId: {
      type: String,
      default: null,
    },
  },
  {
    timestamps: true, // Adds createdAt and updatedAt
  }
);

// Indexes for efficient queries
scanResultSchema.index({ createdAt: -1 });
scanResultSchema.index({ userId: 1, createdAt: -1 });
scanResultSchema.index({ isInfected: 1, createdAt: -1 });
scanResultSchema.index({ pestsDetected: 1 });

const ScanResult = mongoose.model('ScanResult', scanResultSchema);

module.exports = ScanResult;
