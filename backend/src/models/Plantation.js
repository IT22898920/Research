const mongoose = require('mongoose');

const plantationSchema = new mongoose.Schema(
  {
    // Owner relationship to User
    owner: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Owner is required'],
      index: true,
    },

    // Plantation details
    name: {
      type: String,
      required: [true, 'Plantation name is required'],
      trim: true,
      maxlength: [100, 'Name cannot exceed 100 characters'],
    },

    description: {
      type: String,
      trim: true,
      maxlength: [500, 'Description cannot exceed 500 characters'],
    },

    // Location info (center point of plantation)
    location: {
      type: {
        type: String,
        enum: ['Point'],
        default: 'Point',
      },
      coordinates: {
        type: [Number], // [longitude, latitude]
        default: [0, 0],
      },
      address: {
        type: String,
        trim: true,
      },
      district: {
        type: String,
        trim: true,
      },
      province: {
        type: String,
        trim: true,
      },
    },

    // Area in hectares (optional)
    area: {
      type: Number,
      min: [0, 'Area cannot be negative'],
    },

    // Statistics (cached for performance, updated when trees change)
    stats: {
      totalTrees: {
        type: Number,
        default: 0,
      },
      healthyTrees: {
        type: Number,
        default: 0,
      },
      unhealthyTrees: {
        type: Number,
        default: 0,
      },
      infectedTrees: {
        type: Number,
        default: 0,
      },
      notScannedTrees: {
        type: Number,
        default: 0,
      },
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

// Geospatial index for location-based queries
plantationSchema.index({ 'location.coordinates': '2dsphere' });

// Compound index for owner queries
plantationSchema.index({ owner: 1, createdAt: -1 });

// Index for active plantations
plantationSchema.index({ owner: 1, isActive: 1 });

// Virtual for getting plantation's trees
plantationSchema.virtual('trees', {
  ref: 'Tree',
  localField: '_id',
  foreignField: 'plantation',
});

// Ensure virtuals are included in JSON
plantationSchema.set('toJSON', { virtuals: true });
plantationSchema.set('toObject', { virtuals: true });

const Plantation = mongoose.model('Plantation', plantationSchema);

module.exports = Plantation;
