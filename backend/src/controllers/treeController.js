const Tree = require('../models/Tree');
const Plantation = require('../models/Plantation');
const mongoose = require('mongoose');

/**
 * @desc    Create new tree
 * @route   POST /api/trees
 * @access  Private
 */
exports.createTree = async (req, res) => {
  try {
    const {
      plantation,
      label,
      latitude,
      longitude,
      accuracy,
      locationName,
      notes,
    } = req.body;

    // Verify plantation exists and belongs to user
    const plantationDoc = await Plantation.findOne({
      _id: plantation,
      owner: req.user._id,
    });

    if (!plantationDoc) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    const tree = await Tree.create({
      plantation,
      owner: req.user._id,
      label,
      latitude,
      longitude,
      // GeoJSON location (required by schema)
      location: {
        type: 'Point',
        coordinates: [longitude, latitude], // GeoJSON uses [lng, lat]
      },
      accuracy,
      locationName,
      notes,
    });

    res.status(201).json({
      success: true,
      message: 'Tree added successfully',
      data: tree,
    });
  } catch (error) {
    console.error('Create tree error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to create tree',
    });
  }
};

/**
 * @desc    Get all trees for current user
 * @route   GET /api/trees
 * @access  Private
 */
exports.getAllMyTrees = async (req, res) => {
  try {
    const { page = 1, limit = 100, sort = '-createdAt' } = req.query;

    const query = { owner: req.user._id, isActive: true };

    const trees = await Tree.find(query)
      .populate('plantation', 'name')
      .sort(sort)
      .skip((page - 1) * limit)
      .limit(parseInt(limit));

    const total = await Tree.countDocuments(query);

    res.json({
      success: true,
      data: trees,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error('Get all trees error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get trees',
    });
  }
};

/**
 * @desc    Get trees by plantation
 * @route   GET /api/trees/plantation/:plantationId
 * @access  Private
 */
exports.getTreesByPlantation = async (req, res) => {
  try {
    const { plantationId } = req.params;
    const { page = 1, limit = 100, sort = 'label' } = req.query;

    // Verify plantation belongs to user
    const plantation = await Plantation.findOne({
      _id: plantationId,
      owner: req.user._id,
    });

    if (!plantation) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    const query = {
      plantation: plantationId,
      isActive: true,
    };

    const trees = await Tree.find(query)
      .sort(sort)
      .skip((page - 1) * limit)
      .limit(parseInt(limit));

    const total = await Tree.countDocuments(query);

    res.json({
      success: true,
      data: trees,
      plantation: {
        _id: plantation._id,
        name: plantation.name,
        stats: plantation.stats,
      },
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error('Get trees by plantation error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get trees',
    });
  }
};

/**
 * @desc    Get single tree
 * @route   GET /api/trees/:id
 * @access  Private
 */
exports.getTree = async (req, res) => {
  try {
    const tree = await Tree.findOne({
      _id: req.params.id,
      owner: req.user._id,
    }).populate('plantation', 'name');

    if (!tree) {
      return res.status(404).json({
        success: false,
        message: 'Tree not found',
      });
    }

    res.json({
      success: true,
      data: tree,
    });
  } catch (error) {
    console.error('Get tree error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get tree',
    });
  }
};

/**
 * @desc    Update tree
 * @route   PUT /api/trees/:id
 * @access  Private
 */
exports.updateTree = async (req, res) => {
  try {
    const {
      label,
      latitude,
      longitude,
      accuracy,
      locationName,
      notes,
      lastHealthStatus,
    } = req.body;

    const tree = await Tree.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!tree) {
      return res.status(404).json({
        success: false,
        message: 'Tree not found',
      });
    }

    // Update fields
    if (label) tree.label = label;
    if (latitude !== undefined) tree.latitude = latitude;
    if (longitude !== undefined) tree.longitude = longitude;
    // Update GeoJSON location if lat/lng changed
    if (latitude !== undefined || longitude !== undefined) {
      tree.location = {
        type: 'Point',
        coordinates: [
          longitude !== undefined ? longitude : tree.longitude,
          latitude !== undefined ? latitude : tree.latitude,
        ],
      };
    }
    if (accuracy !== undefined) tree.accuracy = accuracy;
    if (locationName !== undefined) tree.locationName = locationName;
    if (notes !== undefined) tree.notes = notes;
    if (lastHealthStatus !== undefined) tree.lastHealthStatus = lastHealthStatus;

    await tree.save();

    res.json({
      success: true,
      message: 'Tree updated successfully',
      data: tree,
    });
  } catch (error) {
    console.error('Update tree error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to update tree',
    });
  }
};

/**
 * @desc    Add health scan to tree
 * @route   POST /api/trees/:id/health-scan
 * @access  Private
 */
exports.addHealthScan = async (req, res) => {
  try {
    const { status, scanType, confidence, details } = req.body;

    const tree = await Tree.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!tree) {
      return res.status(404).json({
        success: false,
        message: 'Tree not found',
      });
    }

    // Add to health history
    tree.healthHistory.push({
      status,
      scanType,
      confidence,
      details,
      scannedAt: new Date(),
    });

    // Update last health status
    tree.lastHealthStatus = status;
    tree.lastScanAt = new Date();

    await tree.save();

    res.json({
      success: true,
      message: 'Health scan added successfully',
      data: tree,
    });
  } catch (error) {
    console.error('Add health scan error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to add health scan',
    });
  }
};

/**
 * @desc    Update scan results (nuts, bunches, pests, issues)
 * @route   PUT /api/trees/:id/scan-results
 * @access  Private
 */
exports.updateScanResults = async (req, res) => {
  try {
    const {
      nutCount,
      bunchCount,
      pestCount,
      detectedIssues,
      healthStatus,
    } = req.body;

    const tree = await Tree.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!tree) {
      return res.status(404).json({
        success: false,
        message: 'Tree not found',
      });
    }

    // Update scan results
    if (nutCount !== undefined) tree.nutCount = nutCount;
    if (bunchCount !== undefined) tree.bunchCount = bunchCount;
    if (pestCount !== undefined) tree.pestCount = pestCount;
    if (detectedIssues !== undefined) tree.detectedIssues = detectedIssues;
    if (healthStatus !== undefined) tree.lastHealthStatus = healthStatus;

    tree.lastScanAt = new Date();

    await tree.save();

    res.json({
      success: true,
      message: 'Scan results updated successfully',
      data: tree,
    });
  } catch (error) {
    console.error('Update scan results error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to update scan results',
    });
  }
};

/**
 * @desc    Clear detected issues
 * @route   DELETE /api/trees/:id/issues
 * @access  Private
 */
exports.clearDetectedIssues = async (req, res) => {
  try {
    const tree = await Tree.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!tree) {
      return res.status(404).json({
        success: false,
        message: 'Tree not found',
      });
    }

    tree.detectedIssues = [];
    await tree.save();

    res.json({
      success: true,
      message: 'Issues cleared successfully',
      data: tree,
    });
  } catch (error) {
    console.error('Clear issues error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to clear issues',
    });
  }
};

/**
 * @desc    Delete tree
 * @route   DELETE /api/trees/:id
 * @access  Private
 */
exports.deleteTree = async (req, res) => {
  try {
    const tree = await Tree.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!tree) {
      return res.status(404).json({
        success: false,
        message: 'Tree not found',
      });
    }

    const plantationId = tree.plantation;

    await Tree.findByIdAndDelete(tree._id);

    // Update plantation stats
    await Tree.updatePlantationStats(plantationId);

    res.json({
      success: true,
      message: 'Tree deleted successfully',
    });
  } catch (error) {
    console.error('Delete tree error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to delete tree',
    });
  }
};

/**
 * @desc    Bulk import trees
 * @route   POST /api/trees/bulk
 * @access  Private
 */
exports.bulkImportTrees = async (req, res) => {
  try {
    const { trees, plantationId } = req.body;

    if (!trees || !Array.isArray(trees) || trees.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Trees array is required',
      });
    }

    // Verify plantation belongs to user
    const plantation = await Plantation.findOne({
      _id: plantationId,
      owner: req.user._id,
    });

    if (!plantation) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    // Prepare trees with owner, plantation, and GeoJSON location
    const treesToInsert = trees.map((tree) => ({
      ...tree,
      plantation: plantationId,
      owner: req.user._id,
      // Ensure GeoJSON location is set
      location: tree.location || {
        type: 'Point',
        coordinates: [tree.longitude, tree.latitude],
      },
    }));

    const insertedTrees = await Tree.insertMany(treesToInsert);

    // Update plantation stats
    await Tree.updatePlantationStats(plantationId);

    res.status(201).json({
      success: true,
      message: `${insertedTrees.length} trees imported successfully`,
      data: insertedTrees,
    });
  } catch (error) {
    console.error('Bulk import error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to import trees',
    });
  }
};

/**
 * @desc    Get nearby trees
 * @route   GET /api/trees/nearby
 * @access  Private
 */
exports.getNearbyTrees = async (req, res) => {
  try {
    const { lat, lng, radius = 1000 } = req.query; // radius in meters

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required',
      });
    }

    const trees = await Tree.find({
      owner: req.user._id,
      isActive: true,
      location: {
        $near: {
          $geometry: {
            type: 'Point',
            coordinates: [parseFloat(lng), parseFloat(lat)],
          },
          $maxDistance: parseInt(radius),
        },
      },
    }).populate('plantation', 'name');

    res.json({
      success: true,
      data: trees,
      center: {
        lat: parseFloat(lat),
        lng: parseFloat(lng),
      },
      radius: parseInt(radius),
    });
  } catch (error) {
    console.error('Get nearby trees error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get nearby trees',
    });
  }
};

/**
 * @desc    Get tree statistics
 * @route   GET /api/trees/stats
 * @access  Private
 */
exports.getTreeStats = async (req, res) => {
  try {
    const { plantationId } = req.query;

    const matchQuery = {
      owner: req.user._id,
      isActive: true,
    };

    if (plantationId) {
      matchQuery.plantation = new mongoose.Types.ObjectId(plantationId);
    }

    const stats = await Tree.aggregate([
      { $match: matchQuery },
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
          totalNuts: { $sum: { $ifNull: ['$nutCount', 0] } },
          totalBunches: { $sum: { $ifNull: ['$bunchCount', 0] } },
        },
      },
    ]);

    const defaultStats = {
      totalTrees: 0,
      healthyTrees: 0,
      unhealthyTrees: 0,
      infectedTrees: 0,
      notScannedTrees: 0,
      totalNuts: 0,
      totalBunches: 0,
    };

    res.json({
      success: true,
      data: stats.length > 0 ? stats[0] : defaultStats,
    });
  } catch (error) {
    console.error('Get tree stats error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get tree stats',
    });
  }
};
