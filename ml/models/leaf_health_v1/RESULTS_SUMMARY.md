# Leaf Health Detection Model v1 - Results Summary

**Date:** 2025-12-30
**Model:** MobileNetV2 with Transfer Learning
**Task:** Binary Classification (Healthy vs Unhealthy Leaves)

---

## Model Configuration

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Loss Function:** Focal Loss (gamma=2.0, alpha=0.25)
- **Optimizer:** Adam
- **Input Size:** 224x224x3
- **Classes:** healthy, unhealthy
- **Training Strategy:** 2-Phase Training
  - Phase 1: Frozen base (15 epochs, LR=0.001)
  - Phase 2: Fine-tuning (15 epochs, LR=0.0001)

---

## Dataset Information

### Training Data
- **Total Samples:** 8,438
  - Healthy: 4,410 images
  - Unhealthy: 4,028 images
  - Class Imbalance Ratio: 1.09 (well balanced)

### Validation Data
- **Total Samples:** 116
  - Healthy: 45 images
  - Unhealthy: 71 images

### Test Data
- **Total Samples:** 127
  - Healthy: 45 images
  - Unhealthy: 82 images

### Data Augmentation
- Rotation Range: 20°
- Width/Height Shift: 20%
- Horizontal Flip: Yes
- Zoom Range: 20%

---

## Training Results

### Training Time
- **Phase 1:** 329.0 minutes (~5.5 hours)
- **Phase 2:** 11.2 minutes
- **Total:** 340.2 minutes (~5.7 hours)

### Final Training Metrics
- **Training Accuracy:** 99.88%
- **Validation Accuracy:** 98.28%
- **Train-Val Gap:** 1.61% ✅ (No significant overfitting)

---

## Test Set Performance

### Overall Metrics
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **93.70%** |
| **Macro Precision** | **92.60%** |
| **Macro Recall** | **94.12%** |
| **Macro F1-Score** | **93.24%** |

### Class-wise Performance

#### HEALTHY Class
| Metric | Value |
|--------|-------|
| Precision | 87.76% |
| Recall | 95.56% |
| F1-Score | 91.49% |
| Support | 45 images |
| Correctly Classified | 43/45 (95.6%) |
| Misclassified | 2/45 (4.4%) |

#### UNHEALTHY Class
| Metric | Value |
|--------|-------|
| Precision | 97.44% |
| Recall | 92.68% |
| F1-Score | 95.00% |
| Support | 82 images |
| Correctly Classified | 76/82 (92.7%) |
| Misclassified | 6/82 (7.3%) |

---

## Quality Checks ✅

### 1. Metric Balance (Precision, Recall, F1)
- **Healthy Class:**
  - P=87.76%, R=95.56%, F1=91.49%
  - Max difference: 7.8% (acceptable)
  - **Status:** ✅ Well balanced

- **Unhealthy Class:**
  - P=97.44%, R=92.68%, F1=95.00%
  - Max difference: 4.76% (very good)
  - **Status:** ✅ Well balanced

### 2. Accuracy-F1 Alignment
- Accuracy: 93.70%
- Macro F1: 93.24%
- Difference: **0.46%** (excellent!)
- **Status:** ✅ Very close alignment

### 3. Overfitting Check
- Training Accuracy: 99.88%
- Validation Accuracy: 98.28%
- Gap: **1.61%**
- **Status:** ✅ No significant overfitting

### 4. Data Leakage Prevention
- Proper train/val/test split maintained
- No data augmentation on val/test sets
- Independent test set evaluation
- **Status:** ✅ No data leakage

---

## Confusion Matrix

### Counts
```
              Predicted
              healthy  unhealthy
Actual healthy     43         2
       unhealthy    6        76
```

### Percentages
```
              Predicted
              healthy  unhealthy
Actual healthy  95.6%      4.4%
       unhealthy  7.3%     92.7%
```

### Analysis
- **Healthy leaves:** 95.6% correctly identified, 4.4% false negatives
- **Unhealthy leaves:** 92.7% correctly identified, 7.3% false positives
- Model performs slightly better at detecting healthy leaves (higher recall)
- Very low false positive rate for healthy class (97.44% precision for unhealthy)

---

## Key Findings

### Strengths
1. **High Overall Accuracy:** 93.70% on unseen test data
2. **Balanced Performance:** Both classes have >91% F1-scores
3. **No Overfitting:** Only 1.61% gap between train and validation
4. **Excellent Unhealthy Detection:** 97.44% precision for unhealthy class
5. **High Recall for Healthy:** 95.56% - rarely misses healthy leaves
6. **Well-Aligned Metrics:** Accuracy very close to F1-score (0.46% diff)

### Areas for Improvement
1. **Healthy Precision:** 87.76% - could be improved
   - Currently 12.24% of predicted healthy are actually unhealthy
   - Consider collecting more healthy samples or adjusting classification threshold

2. **Unhealthy Recall:** 92.68% - good but could be better
   - 7.32% of unhealthy leaves are missed
   - Important for early detection of plant health issues

### Recommendations
- Model is **production-ready** for healthy/unhealthy leaf classification
- Consider threshold tuning if precision/recall trade-off needs adjustment
- For critical applications, may want to err on side of flagging as unhealthy
- Could benefit from more diverse unhealthy leaf samples if available

---

## Files Generated

All files saved in: `ml/models/leaf_health_v1/`

1. **best_model.keras** - Trained model (ready for deployment)
2. **model_info.json** - Complete model metadata and metrics
3. **training_history.png** - Training/validation curves over epochs
4. **confusion_matrix.png** - Visual confusion matrix (counts & percentages)
5. **class_distribution.png** - Dataset distribution across splits
6. **sample_images.png** - Sample images from each class
7. **correct_predictions.png** - Examples of correct predictions
8. **wrong_predictions.png** - Examples of misclassifications (for analysis)
9. **RESULTS_SUMMARY.md** - This file

---

## Usage Example

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the model
model = keras.models.load_model('ml/models/leaf_health_v1/best_model.keras')

# Load and preprocess image
img = Image.open('path/to/leaf_image.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
class_names = ['healthy', 'unhealthy']

# Get result
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

---

## Conclusion

✅ **Model Successfully Trained!**

This leaf health detection model achieves **93.70% accuracy** with well-balanced metrics across both classes. The model shows:
- No overfitting (1.61% train-val gap)
- No data leakage (proper data splits)
- Balanced precision/recall/F1 scores
- Excellent alignment between accuracy and F1-score

The model is **ready for deployment** and can reliably distinguish between healthy and unhealthy (yellowing) coconut leaves.

---

**Training Completed:** 2025-12-30
**Total Training Time:** 5.7 hours
**Framework:** TensorFlow 2.20.0
**Architecture:** MobileNetV2 (Transfer Learning)
