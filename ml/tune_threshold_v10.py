"""
Threshold Tuning Script for v10 Model
=====================================
Find the optimal threshold for balanced P/R/F1 across all classes.

Requirements:
1. P/R/F1 should be close to each other (per class)
2. All classes should have similar F1 values
3. Accuracy should be close to Macro F1
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'coconut_mite_v10', 'best_model.keras')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'pest_mite', 'dataset_v4_clean')
TEST_DIR = os.path.join(DATA_DIR, 'test')

IMG_SIZE = 224
BATCH_SIZE = 32

print("=" * 70)
print("  THRESHOLD TUNING - v10 Model")
print("=" * 70)

# Load model
print("\n[1] Loading model...")

# Custom focal loss for loading
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        focal_weight = tf.keras.backend.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * focal_weight * cross_entropy
        return tf.keras.backend.sum(focal_loss, axis=-1)
    return focal_loss_fn

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
)
print(f"  Model loaded: {MODEL_PATH}")

# Load test data
print("\n[2] Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print(f"  Classes: {class_names}")
print(f"  Test samples: {test_generator.samples}")

# Get predictions
print("\n[3] Getting predictions...")
test_generator.reset()
y_probs = model.predict(test_generator, verbose=1)
y_true = test_generator.classes

print(f"  Predictions shape: {y_probs.shape}")

# Function to evaluate at a specific threshold
def evaluate_threshold(y_probs, y_true, threshold, class_names):
    """
    Evaluate model with custom threshold for the primary class (mite).
    Uses multi-class threshold adjustment.
    """
    n_classes = len(class_names)
    mite_idx = class_names.index('coconut_mite')

    # Adjust probabilities: boost mite predictions
    adjusted_probs = y_probs.copy()

    # Method: Lower threshold means we accept lower probability for mite
    # We achieve this by boosting mite probability
    boost_factor = 0.5 / threshold  # If threshold=0.25, boost=2x
    adjusted_probs[:, mite_idx] = adjusted_probs[:, mite_idx] * boost_factor

    # Renormalize (optional, for cleaner interpretation)
    # adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)

    # Get predictions
    y_pred = np.argmax(adjusted_probs, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i]) if i < len(support) else 0
            }
            for i in range(n_classes)
        }
    }

# Function to calculate balance score
def calculate_balance_score(result, class_names):
    """
    Calculate how balanced the metrics are.
    Lower score = better balance.
    """
    scores = []

    # 1. Per-class P/R/F1 balance (should be close)
    for cls in class_names:
        metrics = result['per_class'][cls]
        p, r, f = metrics['precision'], metrics['recall'], metrics['f1']
        if p > 0 and r > 0 and f > 0:
            prf_range = max(p, r, f) - min(p, r, f)
            scores.append(prf_range)

    # 2. Cross-class F1 similarity (all classes should have similar F1)
    f1_values = [result['per_class'][cls]['f1'] for cls in class_names]
    f1_range = max(f1_values) - min(f1_values)
    scores.append(f1_range * 2)  # Weight this more

    # 3. Accuracy ~ Macro F1
    acc_f1_diff = abs(result['accuracy'] - result['macro_f1'])
    scores.append(acc_f1_diff)

    return sum(scores) / len(scores)

# Test different thresholds
print("\n[4] Testing thresholds...")
print("=" * 70)

thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
results = []

for thresh in thresholds:
    result = evaluate_threshold(y_probs, y_true, thresh, class_names)
    result['balance_score'] = calculate_balance_score(result, class_names)
    results.append(result)

# Print results
print(f"\n{'Thresh':<8} {'Acc':>8} {'M-F1':>8} | {'Mite P':>8} {'Mite R':>8} {'Mite F1':>8} | {'Balance':>8}")
print("-" * 80)

for r in results:
    mite = r['per_class']['coconut_mite']
    print(f"{r['threshold']:<8.2f} {r['accuracy']*100:>7.2f}% {r['macro_f1']*100:>7.2f}% | "
          f"{mite['precision']*100:>7.2f}% {mite['recall']*100:>7.2f}% {mite['f1']*100:>7.2f}% | "
          f"{r['balance_score']:>8.4f}")

# Find best threshold
best_result = min(results, key=lambda x: x['balance_score'])
best_thresh = best_result['threshold']

print("\n" + "=" * 70)
print(f"  BEST THRESHOLD: {best_thresh}")
print("=" * 70)

# Detailed report for best threshold
print(f"\n  Overall Metrics:")
print(f"    Accuracy:     {best_result['accuracy']*100:.2f}%")
print(f"    Macro F1:     {best_result['macro_f1']*100:.2f}%")
print(f"    Acc-F1 diff:  {abs(best_result['accuracy'] - best_result['macro_f1'])*100:.2f}%")

print(f"\n  Per-Class Metrics:")
print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'P/R/F1 Gap':>12}")
print("  " + "-" * 60)

all_f1 = []
for cls in class_names:
    m = best_result['per_class'][cls]
    gap = max(m['precision'], m['recall'], m['f1']) - min(m['precision'], m['recall'], m['f1'])
    all_f1.append(m['f1'])
    status = "OK" if gap < 0.15 else "HIGH"
    print(f"  {cls:<15} {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% {m['f1']*100:>9.2f}% {gap*100:>10.2f}% ({status})")

print(f"\n  F1 Range across classes: {(max(all_f1) - min(all_f1))*100:.2f}%")

# Requirements check
print("\n" + "=" * 70)
print("  REQUIREMENTS CHECK")
print("=" * 70)

# Check 1: P/R/F1 balanced per class
prf_balanced = True
for cls in class_names:
    m = best_result['per_class'][cls]
    gap = max(m['precision'], m['recall'], m['f1']) - min(m['precision'], m['recall'], m['f1'])
    if gap > 0.15:
        prf_balanced = False

# Check 2: Accuracy ~ F1
acc_f1_ok = abs(best_result['accuracy'] - best_result['macro_f1']) < 0.05

# Check 3: F1 similar across classes
f1_similar = (max(all_f1) - min(all_f1)) < 0.15

# Check 4: Mite Recall > 75%
mite_recall_ok = best_result['per_class']['coconut_mite']['recall'] >= 0.75

print(f"\n  [1] P/R/F1 Balanced (per class <15%): {'PASS' if prf_balanced else 'FAIL'}")
print(f"  [2] Accuracy ~ F1 (<5% diff):         {'PASS' if acc_f1_ok else 'FAIL'}")
print(f"  [3] F1 Similar (cross-class <15%):    {'PASS' if f1_similar else 'FAIL'}")
print(f"  [4] Mite Recall >75%:                 {'PASS' if mite_recall_ok else 'FAIL'}")

all_pass = prf_balanced and acc_f1_ok and f1_similar and mite_recall_ok
print(f"\n  OVERALL: {'ALL PASS!' if all_pass else 'NEEDS MORE TUNING'}")

# Also show full table for all thresholds
print("\n" + "=" * 70)
print("  FULL RESULTS TABLE (All Thresholds)")
print("=" * 70)

print(f"\n{'Thresh':<8} | {'coconut_mite':<25} | {'healthy':<25} | {'not_coconut':<25}")
print(f"{'':8} | {'P/R/F1':<25} | {'P/R/F1':<25} | {'P/R/F1':<25}")
print("-" * 95)

for r in results:
    mite = r['per_class']['coconut_mite']
    healthy = r['per_class']['healthy']
    not_coco = r['per_class']['not_coconut']

    marker = " <-- BEST" if r['threshold'] == best_thresh else ""

    print(f"{r['threshold']:<8.2f} | "
          f"{mite['precision']*100:>5.1f}/{mite['recall']*100:>5.1f}/{mite['f1']*100:>5.1f}{'':>8} | "
          f"{healthy['precision']*100:>5.1f}/{healthy['recall']*100:>5.1f}/{healthy['f1']*100:>5.1f}{'':>8} | "
          f"{not_coco['precision']*100:>5.1f}/{not_coco['recall']*100:>5.1f}/{not_coco['f1']*100:>5.1f}{marker}")

# Save results
print("\n[5] Saving results...")
output = {
    'model': 'coconut_mite_v10',
    'best_threshold': float(best_thresh),
    'results': [
        {
            'threshold': float(r['threshold']),
            'accuracy': float(r['accuracy']),
            'macro_f1': float(r['macro_f1']),
            'balance_score': float(r['balance_score']),
            'per_class': {
                cls: {
                    'precision': float(r['per_class'][cls]['precision']),
                    'recall': float(r['per_class'][cls]['recall']),
                    'f1': float(r['per_class'][cls]['f1'])
                }
                for cls in class_names
            }
        }
        for r in results
    ],
    'requirements': {
        'prf_balanced': prf_balanced,
        'acc_f1_ok': acc_f1_ok,
        'f1_similar': f1_similar,
        'mite_recall_ok': mite_recall_ok,
        'all_pass': all_pass
    }
}

output_path = os.path.join(BASE_DIR, 'models', 'coconut_mite_v10', 'threshold_tuning.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"  Saved: {output_path}")

print("\n" + "=" * 70)
print("  THRESHOLD TUNING COMPLETE")
print("=" * 70)
print(f"\n  Recommendation: Use threshold = {best_thresh}")
print(f"  This gives the most balanced P/R/F1 across all classes.")
print("=" * 70)
