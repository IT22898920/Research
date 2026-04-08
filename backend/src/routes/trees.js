const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const {
  createTree,
  getAllMyTrees,
  getTreesByPlantation,
  getTree,
  updateTree,
  addHealthScan,
  updateScanResults,
  clearDetectedIssues,
  deleteTree,
  bulkImportTrees,
  getNearbyTrees,
  getTreeStats,
} = require('../controllers/treeController');

// All routes require authentication
router.use(protect);

// Special routes (must be before /:id to avoid conflict)
router.post('/bulk', bulkImportTrees);           // POST /api/trees/bulk
router.get('/nearby', getNearbyTrees);           // GET /api/trees/nearby?lat=X&lng=Y&radius=Z
router.get('/stats', getTreeStats);              // GET /api/trees/stats
router.get('/plantation/:plantationId', getTreesByPlantation); // GET /api/trees/plantation/:id

// CRUD routes
router.route('/')
  .get(getAllMyTrees)         // GET /api/trees
  .post(createTree);          // POST /api/trees

router.route('/:id')
  .get(getTree)               // GET /api/trees/:id
  .put(updateTree)            // PUT /api/trees/:id
  .delete(deleteTree);        // DELETE /api/trees/:id

// Tree-specific actions
router.post('/:id/health-scan', addHealthScan);       // POST /api/trees/:id/health-scan
router.put('/:id/scan-results', updateScanResults);   // PUT /api/trees/:id/scan-results
router.delete('/:id/issues', clearDetectedIssues);    // DELETE /api/trees/:id/issues

module.exports = router;
