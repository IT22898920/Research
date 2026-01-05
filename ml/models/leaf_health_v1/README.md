# Leaf Health Detection Model v1

A binary classifier to detect **healthy** vs **unhealthy (yellowing)** coconut leaves.

## Quick Stats

- **Accuracy:** 93.70%
- **F1-Score:** 93.24%
- **Architecture:** MobileNetV2
- **Training Time:** 5.7 hours
- **Status:** ✅ Production Ready

## Performance by Class

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Healthy | 87.76% | 95.56% | 91.49% |
| Unhealthy | 97.44% | 92.68% | 95.00% |

## Files

- `best_model.keras` - Trained model
- `model_info.json` - Model metadata
- `RESULTS_SUMMARY.md` - Detailed results
- `*.png` - Visualizations

## Usage

```python
from tensorflow import keras
model = keras.models.load_model('ml/models/leaf_health_v1/best_model.keras')
```

See `RESULTS_SUMMARY.md` for complete details.
