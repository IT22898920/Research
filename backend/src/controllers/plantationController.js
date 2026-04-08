const Plantation = require('../models/Plantation');
const Tree = require('../models/Tree');

/**
 * @desc    Create new plantation
 * @route   POST /api/plantations
 * @access  Private
 */
exports.createPlantation = async (req, res) => {
  try {
    const { name, description, location, area } = req.body;

    const plantation = await Plantation.create({
      owner: req.user._id,
      name,
      description,
      location: location || undefined,
      area: area || undefined,
    });

    res.status(201).json({
      success: true,
      message: 'Plantation created successfully',
      data: plantation,
    });
  } catch (error) {
    console.error('Create plantation error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to create plantation',
    });
  }
};

/**
 * @desc    Get all plantations for current user
 * @route   GET /api/plantations
 * @access  Private
 */
exports.getMyPlantations = async (req, res) => {
  try {
    const { page = 1, limit = 20, sort = '-createdAt' } = req.query;

    const query = { owner: req.user._id, isActive: true };

    const plantations = await Plantation.find(query)
      .sort(sort)
      .skip((page - 1) * limit)
      .limit(parseInt(limit));

    const total = await Plantation.countDocuments(query);

    res.json({
      success: true,
      data: plantations,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error('Get plantations error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get plantations',
    });
  }
};

/**
 * @desc    Get single plantation with stats
 * @route   GET /api/plantations/:id
 * @access  Private
 */
exports.getPlantation = async (req, res) => {
  try {
    const plantation = await Plantation.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!plantation) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    res.json({
      success: true,
      data: plantation,
    });
  } catch (error) {
    console.error('Get plantation error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get plantation',
    });
  }
};

/**
 * @desc    Update plantation
 * @route   PUT /api/plantations/:id
 * @access  Private
 */
exports.updatePlantation = async (req, res) => {
  try {
    const { name, description, location, area } = req.body;

    const plantation = await Plantation.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!plantation) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    // Update fields
    if (name) plantation.name = name;
    if (description !== undefined) plantation.description = description;
    if (location) plantation.location = location;
    if (area !== undefined) plantation.area = area;

    await plantation.save();

    res.json({
      success: true,
      message: 'Plantation updated successfully',
      data: plantation,
    });
  } catch (error) {
    console.error('Update plantation error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to update plantation',
    });
  }
};

/**
 * @desc    Delete plantation and all its trees
 * @route   DELETE /api/plantations/:id
 * @access  Private
 */
exports.deletePlantation = async (req, res) => {
  try {
    const plantation = await Plantation.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!plantation) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    // Delete all trees in this plantation
    const deletedTrees = await Tree.deleteMany({
      plantation: plantation._id,
    });

    // Delete the plantation
    await Plantation.findByIdAndDelete(plantation._id);

    res.json({
      success: true,
      message: 'Plantation and all trees deleted successfully',
      data: {
        deletedTreesCount: deletedTrees.deletedCount,
      },
    });
  } catch (error) {
    console.error('Delete plantation error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to delete plantation',
    });
  }
};

/**
 * @desc    Get plantation statistics
 * @route   GET /api/plantations/:id/stats
 * @access  Private
 */
exports.getPlantationStats = async (req, res) => {
  try {
    const plantation = await Plantation.findOne({
      _id: req.params.id,
      owner: req.user._id,
    });

    if (!plantation) {
      return res.status(404).json({
        success: false,
        message: 'Plantation not found',
      });
    }

    // Get fresh stats from trees
    const stats = await Tree.aggregate([
      {
        $match: {
          plantation: plantation._id,
          isActive: true,
        },
      },
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
          avgPestCount: { $avg: { $ifNull: ['$pestCount', 0] } },
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
      avgPestCount: 0,
    };

    res.json({
      success: true,
      data: {
        plantation: {
          _id: plantation._id,
          name: plantation.name,
        },
        stats: stats.length > 0 ? stats[0] : defaultStats,
      },
    });
  } catch (error) {
    console.error('Get plantation stats error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get plantation stats',
    });
  }
};

/**
 * @desc    Get overall analytics across all plantations
 * @route   GET /api/plantations/analytics
 * @access  Private
 */
exports.getOverallAnalytics = async (req, res) => {
  try {
    // Get plantation count
    const plantationCount = await Plantation.countDocuments({
      owner: req.user._id,
      isActive: true,
    });

    // Get aggregated tree stats across all plantations
    const treeStats = await Tree.aggregate([
      {
        $match: {
          owner: req.user._id,
          isActive: true,
        },
      },
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

    // Get plantation breakdown
    const plantationBreakdown = await Plantation.aggregate([
      {
        $match: {
          owner: req.user._id,
          isActive: true,
        },
      },
      {
        $project: {
          name: 1,
          stats: 1,
          createdAt: 1,
        },
      },
      {
        $sort: { createdAt: -1 },
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
      data: {
        totalPlantations: plantationCount,
        treeStats: treeStats.length > 0 ? treeStats[0] : defaultStats,
        plantations: plantationBreakdown,
      },
    });
  } catch (error) {
    console.error('Get overall analytics error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get analytics',
    });
  }
};
