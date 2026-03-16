"""
Coconut Leaf Health Detection Model v3
Color-Scale Classification: Healthy (Green) vs Unhealthy (Yellow/Brown)
"""

import os
import shutil
import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

print(f"TensorFlow: {tf.__version__}")
print(f"GPU:        {tf.config.list_physical_devices('GPU')}")

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

HEALTHY_SRC   = os.path.join(BASE_DIR, 'data', 'raw', 'healthy-leaves')
UNHEALTHY_SRC = os.path.join(BASE_DIR, 'data', 'raw', 'unhealthy-yellowing')
DATASET_DIR   = os.path.join(BASE_DIR, 'data', 'raw', 'leaf_health_v3', 'dataset')
MODEL_DIR     = os.path.join(BASE_DIR, 'models', 'leaf_health_v3')
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

IMG_SIZE      = 224
BATCH_SIZE    = 32
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 20
LR_PHASE1     = 1e-3
LR_PHASE2     = 3e-5
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
CLASS_NAMES   = ['healthy', 'unhealthy']
VALID_EXT     = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

print(f"\nHealthy source   : {HEALTHY_SRC}")
print(f"Unhealthy source : {UNHEALTHY_SRC}")
print(f"Dataset dir      : {DATASET_DIR}")
print(f"Model dir        : {MODEL_DIR}")

# ─── Collect & Re-split Data ──────────────────────────────────────────────────
def collect_all_images(src_dir):
    images = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                images.append(os.path.join(root, f))
    return images

healthy_all   = collect_all_images(HEALTHY_SRC)
unhealthy_all = collect_all_images(UNHEALTHY_SRC)
random.shuffle(healthy_all)
random.shuffle(unhealthy_all)

print(f"\nHealthy images   : {len(healthy_all)}")
print(f"Unhealthy images : {len(unhealthy_all)}")
print(f"Total            : {len(healthy_all) + len(unhealthy_all)}")

def split_list(lst, train_r, val_r):
    n = len(lst)
    t = int(n * train_r)
    v = int(n * (train_r + val_r))
    return lst[:t], lst[t:v], lst[v:]

h_train, h_val, h_test = split_list(healthy_all,   TRAIN_RATIO, VAL_RATIO)
u_train, u_val, u_test = split_list(unhealthy_all, TRAIN_RATIO, VAL_RATIO)
test_total = len(h_test) + len(u_test)

print(f"\n{'Split':<8} {'Healthy':>10} {'Unhealthy':>12} {'Total':>8}")
print("-" * 44)
print(f"{'Train':<8} {len(h_train):>10} {len(u_train):>12} {len(h_train)+len(u_train):>8}")
print(f"{'Val':<8} {len(h_val):>10} {len(u_val):>12} {len(h_val)+len(u_val):>8}")
print(f"{'Test':<8} {len(h_test):>10} {len(u_test):>12} {test_total:>8}")
print(f"\nTest >= 500: {test_total}  {'OK' if test_total >= 500 else 'NEED MORE DATA'}")

def build_dataset_folders(splits_dict, dataset_dir):
    for split, classes in splits_dict.items():
        for cls, file_list in classes.items():
            dest = os.path.join(dataset_dir, split, cls)
            os.makedirs(dest, exist_ok=True)
            existing = len([f for f in os.listdir(dest) if f.lower().endswith(VALID_EXT)])
            if existing == len(file_list):
                print(f"  {split}/{cls}: {existing} files already present (skipping)")
                continue
            for f in os.listdir(dest):
                if f.lower().endswith(VALID_EXT):
                    os.remove(os.path.join(dest, f))
            for i, src in enumerate(file_list):
                ext = os.path.splitext(src)[1].lower() or '.jpg'
                shutil.copy2(src, os.path.join(dest, f"{cls}_{split}_{i:05d}{ext}"))
            print(f"  {split}/{cls}: copied {len(file_list)} files")

print("\nBuilding dataset structure...")
build_dataset_folders(
    {
        'train': {'healthy': h_train, 'unhealthy': u_train},
        'val':   {'healthy': h_val,   'unhealthy': u_val},
        'test':  {'healthy': h_test,  'unhealthy': u_test},
    },
    DATASET_DIR
)
print("Done!")

# ─── Dataset Summary ──────────────────────────────────────────────────────────
data_summary = {}
print("\n" + "=" * 55)
print("DATASET SUMMARY")
print("=" * 55)
for split in ['train', 'val', 'test']:
    data_summary[split] = {}
    total = 0
    print(f"\n{split.upper()}:")
    for cls in CLASS_NAMES:
        path  = os.path.join(DATASET_DIR, split, cls)
        count = len([f for f in os.listdir(path) if f.lower().endswith(VALID_EXT)])
        data_summary[split][cls] = count
        total += count
        print(f"  {cls:<15} {count:>6}")
    print(f"  {'TOTAL':<15} {total:>6}")

# ─── Color Health Score ───────────────────────────────────────────────────────
def rgb_to_hsv_np(rgb_float):
    r, g, b = rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2]
    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    h = np.zeros_like(r)
    m  = delta > 0
    mr = m & (cmax == r)
    mg = m & (cmax == g)
    mb = m & (cmax == b)
    h[mr] = 60 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    h[mg] = 60 * ((b[mg] - r[mg]) / delta[mg] + 2)
    h[mb] = 60 * ((r[mb] - g[mb]) / delta[mb] + 4)
    h = h / 2.0
    s = np.where(cmax > 0, delta / cmax, 0) * 255
    v = cmax * 255
    return h, s, v

def compute_color_health_score(img_path, resize=(112, 112)):
    try:
        img = Image.open(img_path).convert('RGB').resize(resize)
        rgb = np.array(img, dtype=np.float32) / 255.0
        h, s, v = rgb_to_hsv_np(rgb)
        sat_mask  = s > 40
        total_sat = np.sum(sat_mask)
        if total_sat == 0:
            return 50.0
        green_mask  = sat_mask & (h >= 35) & (h <= 85)
        green_ratio = np.sum(green_mask) / total_sat
        return float(round(green_ratio * 100, 2))
    except Exception:
        return 50.0

# ─── Data Generators ─────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.20,
    shear_range=0.10,
    fill_mode='nearest'
)
eval_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True, seed=42
)
val_gen = eval_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)
test_gen = eval_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print(f"\nTrain : {train_gen.samples}")
print(f"Val   : {val_gen.samples}")
print(f"Test  : {test_gen.samples}")
print(f"Test >= 500: {'YES' if test_gen.samples >= 500 else 'NO'}")

# ─── Class Weights ────────────────────────────────────────────────────────────
train_labels = train_gen.classes
cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: w for i, w in enumerate(cw)}
print(f"\nClass Weights: {class_weight_dict}")

# ─── Focal Loss + Label Smoothing ────────────────────────────────────────────
def focal_loss_smoothed(gamma=2.0, alpha=0.25, label_smoothing=0.1):
    n_classes = 2
    def loss_fn(y_true, y_pred):
        y_smooth = y_true * (1.0 - label_smoothing) + (label_smoothing / n_classes)
        eps = tf.keras.backend.epsilon()
        y_pred_c = tf.keras.backend.clip(y_pred, eps, 1.0 - eps)
        ce = -y_smooth * tf.keras.backend.log(y_pred_c)
        fw = tf.keras.backend.pow(1.0 - y_pred_c, gamma)
        return tf.keras.backend.sum(alpha * fw * ce, axis=-1)
    return loss_fn

# ─── Build Model ──────────────────────────────────────────────────────────────
def build_model():
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(scale=255.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
    return keras.Model(inputs, outputs), base

model, base_model = build_model()
print(f"\nTotal params: {model.count_params():,}")

# ─── Phase 1: Frozen Base ─────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(LR_PHASE1),
    loss=focal_loss_smoothed(gamma=2.0, alpha=0.25, label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_p1 = [
    ModelCheckpoint(os.path.join(MODEL_DIR, 'phase1_best.keras'),
                    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
]

print("\n" + "=" * 65)
print("PHASE 1 — Frozen EfficientNetB0")
print("=" * 65)
t0 = time.time()
history_p1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_p1,
    verbose=1
)
p1_min = (time.time() - t0) / 60
best_p1 = max(history_p1.history['val_accuracy'])
print(f"\nPhase 1: {p1_min:.1f} min | Best val acc: {best_p1*100:.2f}%")

# ─── Phase 2: Fine-tuning ─────────────────────────────────────────────────────
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR_PHASE2),
    loss=focal_loss_smoothed(gamma=2.0, alpha=0.25, label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_p2 = [
    ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.keras'),
                    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1),
]

print(f"\nFine-tuning from layer {fine_tune_at} / {len(base_model.layers)}")
print("=" * 65)
print("PHASE 2 — Fine-tuning color-sensitive layers")
print("=" * 65)
t0 = time.time()
history_p2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_p2,
    verbose=1
)
p2_min = (time.time() - t0) / 60
total_min = p1_min + p2_min
best_p2 = max(history_p2.history['val_accuracy'])
print(f"\nPhase 2: {p2_min:.1f} min | Total: {total_min:.1f} min | Best val acc: {best_p2*100:.2f}%")

# ─── Training History Plot ────────────────────────────────────────────────────
hist = {
    k: history_p1.history[k] + history_p2.history[k]
    for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']
}
p1_end = len(history_p1.history['accuracy']) - 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History — Leaf Health v3', fontsize=13, fontweight='bold')
for ax, (metric, title) in zip(axes, [('accuracy', 'Accuracy'), ('loss', 'Loss')]):
    ax.plot(hist[metric],          label='Train', linewidth=2)
    ax.plot(hist[f'val_{metric}'], label='Val',   linewidth=2)
    ax.axvline(p1_end, color='red', linestyle='--', alpha=0.7, label='Fine-tune start')
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: training_history.png")

final_train_acc = hist['accuracy'][-1]
final_val_acc   = hist['val_accuracy'][-1]

# ─── Evaluate on Test Set ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EVALUATING ON TEST SET")
print("=" * 65)

best_model = keras.models.load_model(
    os.path.join(MODEL_DIR, 'best_model.keras'),
    custom_objects={'loss_fn': focal_loss_smoothed(gamma=2.0, alpha=0.25, label_smoothing=0.1)}
)

test_gen.reset()
preds    = best_model.predict(test_gen, verbose=1)
y_true   = test_gen.classes
y_pred   = np.argmax(preds, axis=1)
y_conf   = np.max(preds, axis=1)
test_acc = np.mean(y_true == y_pred)

print(f"\nTest samples  : {len(y_true)}")
print(f"Test accuracy : {test_acc*100:.2f}%")

# ─── Class-wise Metrics ───────────────────────────────────────────────────────
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
macro_p = np.mean(precision)
macro_r = np.mean(recall)
macro_f = np.mean(f1)

print("\n" + "=" * 90)
print("CLASS-WISE METRICS")
print("=" * 90)
print(f"\n{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
print("-" * 65)
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls:<15} {precision[i]*100:>11.2f}% {recall[i]*100:>11.2f}%"
          f" {f1[i]*100:>11.2f}% {support[i]:>10}")
print("-" * 65)
print(f"{'Macro Avg':<15} {macro_p*100:>11.2f}% {macro_r*100:>11.2f}% {macro_f*100:>11.2f}%")

print("\n" + "=" * 90)
print("SUPERVISOR REQUIREMENT CHECKS")
print("=" * 90)
print(f"\n[1] P, R, F1 close per class (max diff < 10%):")
for i, cls in enumerate(CLASS_NAMES):
    p, r, f = precision[i], recall[i], f1[i]
    max_d = max(abs(p-r), abs(p-f), abs(r-f))
    print(f"  {cls.upper():12} P={p*100:.2f}%  R={r*100:.2f}%  F1={f*100:.2f}%"
          f"  max_diff={max_d*100:.2f}%  {'OK' if max_d < 0.10 else 'CHECK'}")

f1_diff = abs(f1[0] - f1[1])
print(f"\n[2] Cross-class F1 diff: {f1_diff*100:.2f}%  {'OK' if f1_diff < 0.10 else 'CHECK'}")
print(f"[3] Test set size: {len(y_true)}  {'OK (>=500)' if len(y_true) >= 500 else 'FAIL'}")
print(f"[4] Accuracy: {test_acc*100:.2f}%  {'OK (>=95%)' if test_acc >= 0.95 else 'below 95% target'}")

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrix — Leaf Health v3', fontsize=13, fontweight='bold')
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title('Counts'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title('Percentages (%)'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: confusion_matrix.png")

print("\n" + "=" * 70)
print("FULL CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ─── Color Health Score Validation ───────────────────────────────────────────
print("Computing Color Health Scores for all test images...")
filenames = test_gen.filenames
color_scores = np.array([
    compute_color_health_score(os.path.join(DATASET_DIR, 'test', fn))
    for fn in filenames
])

h_scores_test = color_scores[y_true == 0]
u_scores_test = color_scores[y_true == 1]
score_threshold = (h_scores_test.mean() + u_scores_test.mean()) / 2

print(f"\nColor Health Score (Test Set):")
print(f"  Healthy avg score   : {h_scores_test.mean():.1f} / 100")
print(f"  Unhealthy avg score : {u_scores_test.mean():.1f} / 100")
print(f"  Separation          : {abs(h_scores_test.mean()-u_scores_test.mean()):.1f} pts")
print(f"  Optimal threshold   : {score_threshold:.1f}")

# ─── Save Model Info ──────────────────────────────────────────────────────────
model_info = {
    'model_name'   : 'leaf_health_v3',
    'architecture' : 'EfficientNetB0',
    'version'      : 'v3',
    'classes'      : CLASS_NAMES,
    'num_classes'  : 2,
    'input_shape'  : [IMG_SIZE, IMG_SIZE, 3],
    'color_health_score': {
        'description'    : 'Green pixel ratio (HSV) mapped to 0-100 scale',
        'formula'        : 'score = (green_pixels / total_saturated_pixels) * 100',
        'green_hue_range': '35-85 (OpenCV scale 0-180)',
        'sat_threshold'  : 40,
        'labels': {'70-100': 'Healthy', '30-69': 'Moderate', '0-29': 'Unhealthy'}
    },
    'training': {
        'phase1_epochs'         : PHASE1_EPOCHS,
        'phase2_epochs'         : PHASE2_EPOCHS,
        'batch_size'            : BATCH_SIZE,
        'lr_phase1'             : LR_PHASE1,
        'lr_phase2'             : LR_PHASE2,
        'loss_function'         : 'Focal Loss (gamma=2.0, alpha=0.25) + Label Smoothing 0.1',
        'optimizer'             : 'Adam',
        'augmentation'          : 'Geometric only — NO color changes',
        'class_weights'         : {str(k): float(v) for k, v in class_weight_dict.items()},
        'training_time_minutes' : round(total_min, 1),
        'final_train_accuracy'  : float(final_train_acc),
        'final_val_accuracy'    : float(final_val_acc)
    },
    'data': {
        'healthy_source'   : 'healthy-leaves/',
        'unhealthy_source' : 'unhealthy-yellowing/',
        'split'            : '70% train / 15% val / 15% test',
        'train_samples'    : train_gen.samples,
        'val_samples'      : val_gen.samples,
        'test_samples'     : test_gen.samples
    },
    'test_performance': {
        'accuracy'            : float(test_acc),
        'macro_precision'     : float(macro_p),
        'macro_recall'        : float(macro_r),
        'macro_f1'            : float(macro_f),
        'healthy_precision'   : float(precision[0]),
        'healthy_recall'      : float(recall[0]),
        'healthy_f1'          : float(f1[0]),
        'unhealthy_precision' : float(precision[1]),
        'unhealthy_recall'    : float(recall[1]),
        'unhealthy_f1'        : float(f1[1])
    },
    'color_score_stats': {
        'healthy_mean_score'   : float(h_scores_test.mean()),
        'unhealthy_mean_score' : float(u_scores_test.mean()),
        'optimal_threshold'    : float(score_threshold)
    },
    'supervisor_checks': {
        'test_samples_500_plus'  : bool(test_gen.samples >= 500),
        'accuracy_95_plus'       : bool(test_acc >= 0.95),
        'healthy_prf_balanced'   : bool(max(abs(precision[0]-recall[0]),
                                            abs(precision[0]-f1[0]),
                                            abs(recall[0]-f1[0])) < 0.10),
        'unhealthy_prf_balanced' : bool(max(abs(precision[1]-recall[1]),
                                            abs(precision[1]-f1[1]),
                                            abs(recall[1]-f1[1])) < 0.10),
        'cross_class_balanced'   : bool(abs(f1[0]-f1[1]) < 0.10),
        'no_overconfidence'      : bool(float(y_conf.max()) < 0.999)
    }
}

with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=2)

# ─── Final Summary ────────────────────────────────────────────────────────────
chk = model_info['supervisor_checks']
print("\n" + "=" * 85)
print("  COCONUT LEAF HEALTH MODEL v3 — FINAL SUMMARY")
print("=" * 85)
print(f"\n  Test accuracy   : {test_acc*100:.2f}%  {'OK' if test_acc >= 0.95 else 'BELOW TARGET'}")
print(f"  Macro Precision : {macro_p*100:.2f}%")
print(f"  Macro Recall    : {macro_r*100:.2f}%")
print(f"  Macro F1        : {macro_f*100:.2f}%")
print(f"\n  Class-wise:")
for i, cls in enumerate(CLASS_NAMES):
    mx = max(abs(precision[i]-recall[i]), abs(precision[i]-f1[i]), abs(recall[i]-f1[i]))
    print(f"    {cls.upper():12}  P={precision[i]*100:.2f}%  R={recall[i]*100:.2f}%"
          f"  F1={f1[i]*100:.2f}%  {'OK' if mx < 0.10 else 'CHECK'}")
print(f"\n  Color Score: Healthy avg={h_scores_test.mean():.1f}  Unhealthy avg={u_scores_test.mean():.1f}")
print(f"\n  Supervisor Checks:")
print(f"    [{'OK' if chk['test_samples_500_plus']  else 'FAIL'}] Test >= 500          : {test_gen.samples}")
print(f"    [{'OK' if chk['accuracy_95_plus']       else 'WARN'}] Accuracy >= 95%      : {test_acc*100:.2f}%")
print(f"    [{'OK' if chk['healthy_prf_balanced']   else 'WARN'}] Healthy P/R/F1 balanced")
print(f"    [{'OK' if chk['unhealthy_prf_balanced'] else 'WARN'}] Unhealthy P/R/F1 balanced")
print(f"    [{'OK' if chk['cross_class_balanced']   else 'WARN'}] Cross-class balanced")
print(f"    [{'OK' if chk['no_overconfidence']      else 'WARN'}] No overconfidence (max={y_conf.max()*100:.1f}%)")
print(f"\n  Model: {MODEL_DIR}/best_model.keras")
print(f"  Info : {MODEL_DIR}/model_info.json")
print("\n" + "=" * 85)
print("           TRAINING COMPLETE — leaf_health_v3")
print("=" * 85)
