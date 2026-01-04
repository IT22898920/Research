const ScanResult = require('../models/ScanResult');
const mongoose = require('mongoose');
const { uploadImage, deleteImage } = require('../config/cloudinary');
const { sendPestDetectionNotification } = require('../services/pushNotificationService');

/**
 * @desc    Save a new scan result
 * @route   POST /api/scans
 * @access  Private
 */
exports.saveScanResult = async (req, res) => {
  try {
    const {
      scanType,
      isInfected,
      isValidImage,
      results,
      severity,
      pestsDetected,
      treatmentRequested,
      deviceInfo,
      imageBase64, // Base64 encoded image
    } = req.body;

    // Upload image to Cloudinary if provided
    let imageUrl = null;
    let imagePublicId = null;

    if (imageBase64) {
      try {
        const uploadResult = await uploadImage(imageBase64, 'coconut-scans');
        imageUrl = uploadResult.url;
        imagePublicId = uploadResult.publicId;
      } catch (uploadError) {
        console.error('Image upload failed:', uploadError.message);
        // Continue without image - don't fail the whole request
      }
    }

    // Normalize pestsDetected to lowercase (API may return 'Caterpillar', 'Coconut Mite', etc.)
    const normalizedPests = (pestsDetected || []).map(pest => {
      const lower = pest.toLowerCase().replace(/\s+/g, '_'); // Replace spaces with underscores
      if (lower === 'caterpillar') return 'caterpillar';
      if (lower === 'coconut_mite' || lower === 'mite') return 'coconut_mite';
      return lower;
    }).filter(p => ['coconut_mite', 'caterpillar'].includes(p));

    console.log('📊 Scan data - isInfected:', isInfected, 'pestsDetected:', pestsDetected, 'normalized:', normalizedPests);

    const scanResult = await ScanResult.create({
      userId: req.user._id,
      scanType,
      isInfected,
      isValidImage: isValidImage !== false,
      results,
      severity,
      pestsDetected: normalizedPests,
      treatmentRequested: treatmentRequested || false,
      deviceInfo,
      imageUrl,
      imagePublicId,
    });

    console.log('✅ Scan saved successfully:', scanResult._id);

    // Send push notification if pest was detected
    if (isInfected && normalizedPests.length > 0) {
      sendPestDetectionNotification(req.user._id, {
        pestsDetected: normalizedPests,
        severity,
        scanId: scanResult._id,
      }).then(result => {
        if (result.success) {
          console.log('📱 Push notification sent for scan:', scanResult._id);
        } else {
          console.log('📱 Push notification skipped:', result.reason);
        }
      }).catch(err => {
        console.error('📱 Push notification error:', err.message);
      });
    }

    res.status(201).json({
      success: true,
      data: scanResult,
    });
  } catch (error) {
    console.error('Save scan error:', error.message);
    console.error('Full error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Error saving scan result',
    });
  }
};

/**
 * @desc    Get user's scan history with pagination
 * @route   GET /api/scans/my-scans
 * @access  Private
 */
exports.getUserScans = async (req, res) => {
  try {
    console.log('=== GET USER SCANS ===');
    console.log('User ID:', req.user._id);

    const {
      page = 1,
      limit = 20,
      scanType,
      isInfected,
      startDate,
      endDate,
    } = req.query;

    const query = { userId: req.user._id };

    // Filters
    if (scanType) query.scanType = scanType;
    if (isInfected !== undefined) query.isInfected = isInfected === 'true';
    if (startDate || endDate) {
      query.createdAt = {};
      if (startDate) query.createdAt.$gte = new Date(startDate);
      if (endDate) query.createdAt.$lte = new Date(endDate);
    }

    const pageNum = parseInt(page, 10);
    const limitNum = parseInt(limit, 10);
    const skip = (pageNum - 1) * limitNum;

    const [scans, totalScans] = await Promise.all([
      ScanResult.find(query).sort({ createdAt: -1 }).skip(skip).limit(limitNum),
      ScanResult.countDocuments(query),
    ]);

    console.log('Found', totalScans, 'scans for user');
    console.log('Scan IDs:', scans.map(s => s._id.toString()));

    res.status(200).json({
      success: true,
      data: {
        scans,
        pagination: {
          currentPage: pageNum,
          totalPages: Math.ceil(totalScans / limitNum),
          totalScans,
          hasMore: pageNum < Math.ceil(totalScans / limitNum),
        },
      },
    });
  } catch (error) {
    console.error('Get user scans error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching scan history',
    });
  }
};

/**
 * @desc    Get user's scan statistics
 * @route   GET /api/scans/my-stats
 * @access  Private
 */
exports.getScanStats = async (req, res) => {
  try {
    const userId = req.user._id;

    const stats = await ScanResult.aggregate([
      { $match: { userId: new mongoose.Types.ObjectId(userId) } },
      {
        $group: {
          _id: null,
          totalScans: { $sum: 1 },
          infectedScans: {
            $sum: { $cond: ['$isInfected', 1, 0] },
          },
          healthyScans: {
            $sum: { $cond: ['$isInfected', 0, 1] },
          },
          miteDetections: {
            $sum: { $cond: ['$results.mite.detected', 1, 0] },
          },
          caterpillarDetections: {
            $sum: { $cond: ['$results.caterpillar.detected', 1, 0] },
          },
        },
      },
    ]);

    const result = stats[0] || {
      totalScans: 0,
      infectedScans: 0,
      healthyScans: 0,
      miteDetections: 0,
      caterpillarDetections: 0,
    };

    result.infectionRate =
      result.totalScans > 0
        ? ((result.infectedScans / result.totalScans) * 100).toFixed(1)
        : 0;

    res.status(200).json({
      success: true,
      data: result,
    });
  } catch (error) {
    console.error('Get scan stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching statistics',
    });
  }
};

/**
 * @desc    Get single scan by ID
 * @route   GET /api/scans/:id
 * @access  Private
 */
exports.getScanById = async (req, res) => {
  try {
    const scan = await ScanResult.findOne({
      _id: req.params.id,
      userId: req.user._id,
    });

    if (!scan) {
      return res.status(404).json({
        success: false,
        message: 'Scan not found',
      });
    }

    res.status(200).json({
      success: true,
      data: scan,
    });
  } catch (error) {
    console.error('Get scan by ID error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching scan',
    });
  }
};

/**
 * @desc    Delete a scan
 * @route   DELETE /api/scans/:id
 * @access  Private
 */
exports.deleteScan = async (req, res) => {
  try {
    console.log('Delete request for scan:', req.params.id, 'by user:', req.user._id);

    // Validate ObjectId format
    if (!mongoose.Types.ObjectId.isValid(req.params.id)) {
      console.log('Invalid ObjectId format:', req.params.id);
      return res.status(400).json({
        success: false,
        message: 'Invalid scan ID format',
      });
    }

    // First check if scan exists at all
    const existingScan = await ScanResult.findById(req.params.id);
    if (!existingScan) {
      console.log('Scan does not exist in database');
      return res.status(404).json({
        success: false,
        message: 'Scan not found in database',
      });
    }

    // Check if user owns the scan
    console.log('Scan owner:', existingScan.userId, 'Current user:', req.user._id);
    if (existingScan.userId.toString() !== req.user._id.toString()) {
      console.log('User does not own this scan');
      return res.status(403).json({
        success: false,
        message: 'Not authorized to delete this scan',
      });
    }

    const scan = await ScanResult.findOneAndDelete({
      _id: req.params.id,
      userId: req.user._id,
    });

    if (!scan) {
      console.log('Scan not found or not owned by user');
      return res.status(404).json({
        success: false,
        message: 'Scan not found',
      });
    }

    // Delete image from Cloudinary if exists
    if (scan.imagePublicId) {
      try {
        await deleteImage(scan.imagePublicId);
      } catch (cloudErr) {
        console.error('Cloudinary delete error (non-fatal):', cloudErr.message);
      }
    }

    console.log('Scan deleted successfully:', scan._id);
    res.status(200).json({
      success: true,
      message: 'Scan deleted successfully',
    });
  } catch (error) {
    console.error('Delete scan error:', error.message);
    res.status(500).json({
      success: false,
      message: error.message || 'Error deleting scan',
    });
  }
};

/**
 * @desc    Get admin analytics (all users)
 * @route   GET /api/scans/admin/analytics
 * @access  Private/Admin
 */
exports.getAnalytics = async (req, res) => {
  try {
    const { period = '30' } = req.query; // days
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(period));

    const [overview, dailyStats] = await Promise.all([
      // Overall statistics
      ScanResult.aggregate([
        { $match: { createdAt: { $gte: startDate } } },
        {
          $group: {
            _id: null,
            totalScans: { $sum: 1 },
            infectedScans: { $sum: { $cond: ['$isInfected', 1, 0] } },
            uniqueUsers: { $addToSet: '$userId' },
          },
        },
        {
          $project: {
            totalScans: 1,
            infectedScans: 1,
            healthyScans: { $subtract: ['$totalScans', '$infectedScans'] },
            uniqueUsers: { $size: '$uniqueUsers' },
            infectionRate: {
              $cond: [
                { $eq: ['$totalScans', 0] },
                0,
                {
                  $multiply: [
                    { $divide: ['$infectedScans', '$totalScans'] },
                    100,
                  ],
                },
              ],
            },
          },
        },
      ]),

      // Daily breakdown
      ScanResult.aggregate([
        { $match: { createdAt: { $gte: startDate } } },
        {
          $group: {
            _id: {
              $dateToString: { format: '%Y-%m-%d', date: '$createdAt' },
            },
            scans: { $sum: 1 },
            infected: { $sum: { $cond: ['$isInfected', 1, 0] } },
          },
        },
        { $sort: { _id: 1 } },
      ]),
    ]);

    res.status(200).json({
      success: true,
      data: {
        overview: overview[0] || {
          totalScans: 0,
          infectedScans: 0,
          healthyScans: 0,
          uniqueUsers: 0,
          infectionRate: 0,
        },
        dailyStats,
        period: parseInt(period),
      },
    });
  } catch (error) {
    console.error('Get analytics error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching analytics',
    });
  }
};

/**
 * @desc    Get infection trends over time
 * @route   GET /api/scans/admin/trends
 * @access  Private/Admin
 */
exports.getInfectionTrends = async (req, res) => {
  try {
    const { period = '30', groupBy = 'day' } = req.query;
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(period));

    const dateFormat = groupBy === 'week' ? '%Y-W%V' : '%Y-%m-%d';

    const trends = await ScanResult.aggregate([
      { $match: { createdAt: { $gte: startDate } } },
      {
        $group: {
          _id: {
            date: { $dateToString: { format: dateFormat, date: '$createdAt' } },
          },
          totalScans: { $sum: 1 },
          infectedScans: { $sum: { $cond: ['$isInfected', 1, 0] } },
          miteCount: { $sum: { $cond: ['$results.mite.detected', 1, 0] } },
          caterpillarCount: {
            $sum: { $cond: ['$results.caterpillar.detected', 1, 0] },
          },
        },
      },
      {
        $project: {
          date: '$_id.date',
          totalScans: 1,
          infectedScans: 1,
          healthyScans: { $subtract: ['$totalScans', '$infectedScans'] },
          miteCount: 1,
          caterpillarCount: 1,
          infectionRate: {
            $cond: [
              { $eq: ['$totalScans', 0] },
              0,
              {
                $multiply: [
                  { $divide: ['$infectedScans', '$totalScans'] },
                  100,
                ],
              },
            ],
          },
        },
      },
      { $sort: { date: 1 } },
    ]);

    res.status(200).json({
      success: true,
      data: trends,
    });
  } catch (error) {
    console.error('Get trends error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching trends',
    });
  }
};

/**
 * @desc    Get pest distribution stats
 * @route   GET /api/scans/admin/pest-distribution
 * @access  Private/Admin
 */
exports.getPestDistribution = async (req, res) => {
  try {
    const distribution = await ScanResult.aggregate([
      { $match: { isInfected: true } },
      {
        $group: {
          _id: null,
          miteOnly: {
            $sum: {
              $cond: [
                {
                  $and: [
                    { $eq: ['$results.mite.detected', true] },
                    { $ne: ['$results.caterpillar.detected', true] },
                  ],
                },
                1,
                0,
              ],
            },
          },
          caterpillarOnly: {
            $sum: {
              $cond: [
                {
                  $and: [
                    { $eq: ['$results.caterpillar.detected', true] },
                    { $ne: ['$results.mite.detected', true] },
                  ],
                },
                1,
                0,
              ],
            },
          },
          both: {
            $sum: {
              $cond: [
                {
                  $and: [
                    { $eq: ['$results.mite.detected', true] },
                    { $eq: ['$results.caterpillar.detected', true] },
                  ],
                },
                1,
                0,
              ],
            },
          },
          total: { $sum: 1 },
        },
      },
    ]);

    res.status(200).json({
      success: true,
      data: distribution[0] || {
        miteOnly: 0,
        caterpillarOnly: 0,
        both: 0,
        total: 0,
      },
    });
  } catch (error) {
    console.error('Get pest distribution error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching pest distribution',
    });
  }
};
