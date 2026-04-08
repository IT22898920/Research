const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const {
  createPlantation,
  getMyPlantations,
  getPlantation,
  updatePlantation,
  deletePlantation,
  getPlantationStats,
  getOverallAnalytics,
} = require('../controllers/plantationController');

// All routes require authentication
router.use(protect);

// Analytics route (must be before /:id to avoid conflict)
router.get('/analytics', getOverallAnalytics);

// CRUD routes
router.route('/')
  .get(getMyPlantations)      // GET /api/plantations
  .post(createPlantation);    // POST /api/plantations

router.route('/:id')
  .get(getPlantation)         // GET /api/plantations/:id
  .put(updatePlantation)      // PUT /api/plantations/:id
  .delete(deletePlantation);  // DELETE /api/plantations/:id

// Stats route
router.get('/:id/stats', getPlantationStats);

module.exports = router;
