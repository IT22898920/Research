import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')

import os
import shutil
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import random

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath('..')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATASET_DIR  = os.path.join(RAW_DATA_DIR, 'coconut_tree_health', 'dataset_v2')
MODEL_DIR    = os.path.join(BASE_DIR, 'models', 'coconut_tree_health_v2')

HEALTHY_SRC   = os.path.join(RAW_DATA_DIR, 'Healthy_Coconut_trees')
UNHEALTHY_SRC = os.path.join(RAW_DATA_DIR, 'unhealthy_coconut_trees')

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE           = 224
BATCH_SIZE         = 32
PHASE1_EPOCHS      = 30
PHASE2_EPOCHS      = 25
LEARNING_RATE_P1   = 1e-3
LEARNING_RATE_P2   = 3e-5
HEALTHY_OVERSAMPLE = 4      # repeat healthy images N times to balance classes
TRAIN_RATIO        = 0.65
VAL_RATIO          = 0.15
TEST_RATIO         = 0.20

class_names = ['healthy', 'unhealthy']

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nBase Dir    : {BASE_DIR}")
print(f"Dataset Dir : {DATASET_DIR}")
print(f"Model Dir   : {MODEL_DIR}")
print(f"Oversample  : {HEALTHY_OVERSAMPLE}x healthy images → balance ratio ≈ 1:{1/HEALTHY_OVERSAMPLE*10:.1f}")

# ── Collect images ─────────────────────────────────────────────────────────────
def collect_images(src_dir):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imgs = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(valid_ext):
                imgs.append(os.path.join(root, f))
    return imgs

healthy_images   = collect_images(HEALTHY_SRC)
unhealthy_images = collect_images(UNHEALTHY_SRC)

random.shuffle(healthy_images)
random.shuffle(unhealthy_images)

print(f"\nTotal Healthy   : {len(healthy_images)}")
print(f"Total Unhealthy : {len(unhealthy_images)}")
print(f"Raw ratio       : 1:{len(unhealthy_images)/len(healthy_images):.1f}")

# ── Split ──────────────────────────────────────────────────────────────────────
def split_images(images, train_r, val_r):
    n = len(images)
    t = int(n * train_r)
    v = int(n * (train_r + val_r))
    return images[:t], images[t:v], images[v:]

healthy_train,   healthy_val,   healthy_test   = split_images(healthy_images,   TRAIN_RATIO, VAL_RATIO)
unhealthy_train, unhealthy_val, unhealthy_test = split_images(unhealthy_images, TRAIN_RATIO, VAL_RATIO)

# ── Oversample healthy training set ───────────────────────────────────────────
# Repeat healthy images HEALTHY_OVERSAMPLE times so the model sees more variety
healthy_train_oversampled = (healthy_train * HEALTHY_OVERSAMPLE)
random.shuffle(healthy_train_oversampled)

ratio_after = len(unhealthy_train) / len(healthy_train_oversampled)
print(f"\n── After oversampling (train only) ──────────────────")
print(f"Healthy   (train): {len(healthy_train):>4} → {len(healthy_train_oversampled):>4} (x{HEALTHY_OVERSAMPLE})")
print(f"Unhealthy (train): {len(unhealthy_train):>4}")
print(f"New ratio         : 1:{ratio_after:.2f}  ← much better than 1:10.7")

# ── Build dataset directory structure ─────────────────────────────────────────
def build_dataset(h_train, h_val, h_test, u_train, u_val, u_test, dst):
    splits = {
        'train': {'healthy': h_train,   'unhealthy': u_train},
        'val':   {'healthy': h_val,     'unhealthy': u_val},
        'test':  {'healthy': h_test,    'unhealthy': u_test},
    }
    for split, classes in splits.items():
        for cls, file_list in classes.items():
            dest_dir = os.path.join(dst, split, cls)
            os.makedirs(dest_dir, exist_ok=True)

            existing = len([f for f in os.listdir(dest_dir)
                            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
            if existing == len(file_list):
                print(f"  {split}/{cls}: already has {existing} files (skipping)")
                continue

            for f in os.listdir(dest_dir):
                os.remove(os.path.join(dest_dir, f))

            for i, src in enumerate(file_list):
                ext  = os.path.splitext(src)[1].lower()
                name = f"{cls}_{split}_{i:05d}{ext}"
                shutil.copy2(src, os.path.join(dest_dir, name))

            print(f"  {split}/{cls}: copied {len(file_list)} files")

print("\nBuilding dataset...")
build_dataset(
    healthy_train_oversampled, healthy_val,   healthy_test,
    unhealthy_train,           unhealthy_val, unhealthy_test,
    DATASET_DIR
)
print("Dataset ready!\n")

# ── Count per split ────────────────────────────────────────────────────────────
data_summary = {}
print("=" * 60)
print("DATASET SUMMARY (v2)")
print("=" * 60)
for split in ['train', 'val', 'test']:
    sp = os.path.join(DATASET_DIR, split)
    print(f"\n{split.upper()}:")
    total = 0
    data_summary[split] = {}
    for cls in class_names:
        cp = os.path.join(sp, cls)
        n  = len([f for f in os.listdir(cp)
                  if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
        data_summary[split][cls] = n
        total += n
        print(f"  {cls:<12} {n:>6}")
    print(f"  {'TOTAL':<12} {total:>6}")
print("=" * 60)

# ── Augmentation ───────────────────────────────────────────────────────────────
# Heavy augmentation for training — more aggressive than v1 for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.30,
    height_shift_range=0.30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.30,
    shear_range=0.20,
    brightness_range=[0.6, 1.4],
    channel_shift_range=25.0,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=True,
    seed=42
)
val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

print(f"\nTrain : {train_gen.samples}")
print(f"Val   : {val_gen.samples}")
print(f"Test  : {test_gen.samples} {'✓' if test_gen.samples >= 500 else '✗'}")
print(f"Class indices: {train_gen.class_indices}")

# ── Class weights (computed on oversampled train set) ─────────────────────────
train_labels      = train_gen.classes
class_weights_arr = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

print("\nClass Weights:")
for idx, cls in enumerate(class_names):
    print(f"  Class {idx} ({cls}): {class_weight_dict[idx]:.4f}")

# ── Focal Loss (gamma=3.0 — more focus on hard examples than v1's 2.0) ────────
def focal_loss(gamma=3.0, alpha=0.25):
    """
    Focal Loss for class imbalance.
    gamma=3.0: stronger focus on hard/misclassified examples than v1 (2.0)
    label_smoothing=0.05: prevents overconfidence (fixes v1's val=100% issue)
    """
    def focal_loss_fn(y_true, y_pred):
        # Label smoothing: reduce overconfidence
        smoothing = 0.05
        y_true_smooth = y_true * (1 - smoothing) + smoothing / tf.cast(tf.shape(y_true)[-1], tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred  = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy  = -y_true_smooth * tf.keras.backend.log(y_pred)
        focal_weight   = tf.keras.backend.pow(1.0 - y_pred, gamma)
        loss = alpha * focal_weight * cross_entropy
        return tf.keras.backend.sum(loss, axis=-1)
    return focal_loss_fn

print("\nFocal Loss v2:")
print("  gamma=3.0       → stronger focus on hard examples")
print("  label_smooth=5% → prevents val_accuracy=100% overconfidence")

# ── Model: EfficientNetB0 (better than MobileNetV2 for tree images) ───────────
def build_model():
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # EfficientNetB0 expects [0, 255] input — rescale back from [0, 1]
    x = layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)                                          # v1: 0.4
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(2e-4))(x) # v1: 1e-4
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)                                          # v1: 0.3
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(2e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)                                          # v1: 0.2
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base

model, base_model = build_model()
model.summary()
print(f"\nBase layers  : {len(base_model.layers)}")
print(f"Total params : {model.count_params():,}")

# ── Phase 1: Frozen base ───────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_P1, clipnorm=1.0),
    loss=focal_loss(gamma=3.0, alpha=0.25),
    metrics=['accuracy']
)

callbacks_p1 = [
    ModelCheckpoint(os.path.join(MODEL_DIR, 'phase1_best.keras'),
                    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
]

print("\n" + "=" * 70)
print("PHASE 1: Frozen EfficientNetB0 base")
print(f"LR={LEARNING_RATE_P1}  Epochs={PHASE1_EPOCHS}  Class Weights=ON")
print("=" * 70)

t0 = time.time()
history_p1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_p1,
    verbose=1
)
phase1_min = (time.time() - t0) / 60
print(f"\nPhase 1 done in {phase1_min:.1f} min")
print(f"Best val accuracy: {max(history_p1.history['val_accuracy'])*100:.2f}%")

# ── Phase 2: Fine-tune last 50 layers ─────────────────────────────────────────
# Unfreeze last 50 layers (v1 used 40)
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"\nFine-tuning from layer {fine_tune_at} — trainable layers: {trainable}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_P2, clipnorm=1.0),
    loss=focal_loss(gamma=3.0, alpha=0.25),
    metrics=['accuracy']
)

callbacks_p2 = [
    ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.keras'),
                    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1),
]

print("\n" + "=" * 70)
print("PHASE 2: Fine-tuning last 50 layers")
print(f"LR={LEARNING_RATE_P2}  Epochs={PHASE2_EPOCHS}  Class Weights=ON")
print("=" * 70)

t0 = time.time()
history_p2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_p2,
    verbose=1
)
phase2_min = (time.time() - t0) / 60
total_min  = phase1_min + phase2_min

print(f"\nPhase 2 done in {phase2_min:.1f} min")
print(f"Total training time: {total_min:.1f} min")
print(f"Best val accuracy: {max(history_p2.history['val_accuracy'])*100:.2f}%")

# ── Training history plot ──────────────────────────────────────────────────────
history = {
    'accuracy':     history_p1.history['accuracy']     + history_p2.history['accuracy'],
    'val_accuracy': history_p1.history['val_accuracy'] + history_p2.history['val_accuracy'],
    'loss':         history_p1.history['loss']         + history_p2.history['loss'],
    'val_loss':     history_p1.history['val_loss']     + history_p2.history['val_loss'],
}
p1_end = len(history_p1.history['accuracy']) - 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History — Coconut Tree Health v2 (EfficientNetB0)', fontsize=13, fontweight='bold')

axes[0].plot(history['accuracy'],     label='Train', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Val',   linewidth=2)
axes[0].axvline(x=p1_end, color='red', linestyle='--', label='Fine-tune start', alpha=0.7)
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history['loss'],     label='Train', linewidth=2)
axes[1].plot(history['val_loss'], label='Val',   linewidth=2)
axes[1].axvline(x=p1_end, color='red', linestyle='--', label='Fine-tune start', alpha=0.7)
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
print("training_history.png saved")

# ── Overfitting check ──────────────────────────────────────────────────────────
final_train_acc = history['accuracy'][-1]
final_val_acc   = history['val_accuracy'][-1]
gap = abs(final_train_acc - final_val_acc)

print("\n" + "=" * 60)
print("OVERFITTING CHECK")
print("=" * 60)
print(f"Train Accuracy : {final_train_acc*100:.2f}%")
print(f"Val   Accuracy : {final_val_acc*100:.2f}%")
print(f"Gap            : {gap*100:.2f}%  {'✓ OK' if gap < 0.05 else '⚠ Minor' if gap < 0.10 else '⚠⚠ Overfit'}")
print("=" * 60)

# ── Evaluate on test set ───────────────────────────────────────────────────────
best_model = keras.models.load_model(
    os.path.join(MODEL_DIR, 'best_model.keras'),
    custom_objects={'focal_loss_fn': focal_loss(gamma=3.0, alpha=0.25)}
)
print("\nBest model loaded!")

test_gen.reset()
predictions = best_model.predict(test_gen, verbose=1)

y_true = test_gen.classes
y_pred = np.argmax(predictions, axis=1)
y_conf = np.max(predictions, axis=1)

test_accuracy = np.mean(y_true == y_pred)
print(f"\nTest Set   : {len(y_true)} images")
print(f"Accuracy   : {test_accuracy*100:.2f}%")

precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
macro_p = np.mean(precision)
macro_r = np.mean(recall)
macro_f = np.mean(f1)

print("\n" + "=" * 80)
print("DETAILED CLASS-WISE METRICS")
print("=" * 80)
print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Support':>10}")
print("-" * 80)
for i, cls in enumerate(class_names):
    print(f"{cls:<15} {precision[i]*100:>11.2f}% {recall[i]*100:>11.2f}% {f1[i]*100:>11.2f}% {support[i]:>10}")
print("-" * 80)
print(f"{'Macro Avg':<15} {macro_p*100:>11.2f}% {macro_r*100:>11.2f}% {macro_f*100:>11.2f}%")
print("=" * 80)

print("\n" + "=" * 80)
print("FULL CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# ── Confusion matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrix — Tree Health v2', fontsize=13, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Counts'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Percentages'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
print("confusion_matrix.png saved")

for i, true_cls in enumerate(class_names):
    for j, pred_cls in enumerate(class_names):
        cnt = cm[i, j]
        pct = cm_pct[i, j]
        sym = '✓' if i == j else '✗'
        if cnt > 0:
            print(f"{sym} {true_cls} → {pred_cls}: {cnt} ({pct:.1f}%)")

# ── Confidence distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Confidence Distribution — v2', fontsize=13, fontweight='bold')

correct_conf = y_conf[y_true == y_pred]
wrong_conf   = y_conf[y_true != y_pred]

axes[0].hist(correct_conf, bins=20, color='#2ecc71', edgecolor='black', alpha=0.8)
axes[0].set_title(f'Correct (n={len(correct_conf)})')
axes[0].axvline(np.mean(correct_conf), color='darkgreen', linestyle='--',
                label=f'Mean: {np.mean(correct_conf):.2f}')
axes[0].legend()

if len(wrong_conf) > 0:
    axes[1].hist(wrong_conf, bins=20, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1].set_title(f'Wrong (n={len(wrong_conf)})')
    axes[1].axvline(np.mean(wrong_conf), color='darkred', linestyle='--',
                    label=f'Mean: {np.mean(wrong_conf):.2f}')
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, 'No wrong predictions!', ha='center', va='center',
                fontsize=14, color='green', transform=axes[1].transAxes)
    axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
print("confidence_distribution.png saved")

# ── Wrong predictions visualization ───────────────────────────────────────────
from tensorflow.keras.preprocessing import image as keras_image

filenames  = test_gen.filenames
wrong_idx  = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]

print(f"\nTotal: {len(y_true)} | Correct: {len(correct_idx)} | Wrong: {len(wrong_idx)}")

if len(wrong_idx) > 0:
    n = min(10, len(wrong_idx))
    rows = max(1, (n + 4) // 5)
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3.5 * rows))
    fig.suptitle(f'WRONG Predictions — v2 (total: {len(wrong_idx)})', fontsize=13, color='red', fontweight='bold')
    if rows == 1:
        axes = axes.reshape(1, -1)
    for idx, i in enumerate(wrong_idx[:n]):
        r, c = idx // 5, idx % 5
        img_path = os.path.join(DATASET_DIR, 'test', filenames[i])
        img = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        axes[r, c].imshow(img); axes[r, c].axis('off')
        conf = predictions[i][y_pred[i]] * 100
        axes[r, c].set_title(f'T:{class_names[y_true[i]]}\nP:{class_names[y_pred[i]]} ({conf:.1f}%)',
                              fontsize=8, color='red')
    for idx in range(n, rows * 5):
        axes[idx // 5, idx % 5].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'wrong_predictions.png'), dpi=150, bbox_inches='tight')
    print("wrong_predictions.png saved")
else:
    print("Perfect! No wrong predictions!")

# ── Save model info ────────────────────────────────────────────────────────────
model_info = {
    'model_name':    'coconut_tree_health_v2',
    'architecture':  'EfficientNetB0',
    'version':       'v2',
    'classes':       class_names,
    'num_classes':   len(class_names),
    'input_shape':   [IMG_SIZE, IMG_SIZE, 3],
    'improvements_over_v1': [
        'EfficientNetB0 instead of MobileNetV2',
        f'Healthy class oversampled {HEALTHY_OVERSAMPLE}x (ratio 1:10 → 1:{ratio_after:.1f})',
        'Focal Loss gamma=3.0 (v1: 2.0) — stronger focus on hard examples',
        'Label smoothing 5% — prevents overconfidence (v1 val=100% issue fixed)',
        'Higher dropout: 0.5/0.4/0.3 (v1: 0.4/0.3/0.2)',
        'Stronger L2 regularization: 2e-4 (v1: 1e-4)',
        'Fine-tune last 50 layers (v1: 40)',
        'Gradient clipping clipnorm=1.0',
    ],
    'training': {
        'phase1_epochs':         PHASE1_EPOCHS,
        'phase2_epochs':         PHASE2_EPOCHS,
        'batch_size':            BATCH_SIZE,
        'learning_rate_phase1':  LEARNING_RATE_P1,
        'learning_rate_phase2':  LEARNING_RATE_P2,
        'loss_function':         'Focal Loss (gamma=3.0, alpha=0.25, label_smooth=0.05)',
        'optimizer':             'Adam (clipnorm=1.0)',
        'healthy_oversample':    HEALTHY_OVERSAMPLE,
        'class_weights':         class_weight_dict,
        'training_time_minutes': round(total_min, 1),
        'final_train_accuracy':  float(final_train_acc),
        'final_val_accuracy':    float(final_val_acc),
    },
    'data': {
        'split':              '65% train / 15% val / 20% test',
        'train_samples':      train_gen.samples,
        'val_samples':        val_gen.samples,
        'test_samples':       test_gen.samples,
        'train_healthy':      data_summary['train']['healthy'],
        'train_unhealthy':    data_summary['train']['unhealthy'],
        'original_healthy':   len(healthy_images),
        'original_unhealthy': len(unhealthy_images),
        'imbalance_ratio_v1': round(len(unhealthy_images) / len(healthy_images), 2),
        'imbalance_ratio_v2': round(ratio_after, 2),
    },
    'test_performance': {
        'accuracy':             float(test_accuracy),
        'macro_precision':      float(macro_p),
        'macro_recall':         float(macro_r),
        'macro_f1':             float(macro_f),
        'healthy_precision':    float(precision[0]),
        'healthy_recall':       float(recall[0]),
        'healthy_f1':           float(f1[0]),
        'unhealthy_precision':  float(precision[1]),
        'unhealthy_recall':     float(recall[1]),
        'unhealthy_f1':         float(f1[1]),
    },
    'augmentation': {
        'rotation_range':     45,
        'width_shift_range':  0.30,
        'height_shift_range': 0.30,
        'horizontal_flip':    True,
        'vertical_flip':      True,
        'zoom_range':         0.30,
        'shear_range':        0.20,
        'brightness_range':   [0.6, 1.4],
        'channel_shift_range': 25.0,
    }
}

info_path = os.path.join(MODEL_DIR, 'model_info.json')
with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"\nModel info saved: {info_path}")

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)
for fname in sorted(os.listdir(MODEL_DIR)):
    fpath = os.path.join(MODEL_DIR, fname)
    size  = os.path.getsize(fpath) / 1024
    print(f"  {fname:<45} {size:>8.1f} KB")

print("\n" + "=" * 60)
print("V2 SUMMARY")
print("=" * 60)
print(f"  Architecture   : EfficientNetB0 (improved from MobileNetV2)")
print(f"  Healthy train  : {len(healthy_train)} → {len(healthy_train_oversampled)} (x{HEALTHY_OVERSAMPLE} oversample)")
print(f"  Imbalance ratio: 1:{len(unhealthy_images)/len(healthy_images):.1f} → 1:{ratio_after:.1f}")
print(f"  Test Accuracy  : {test_accuracy*100:.2f}%")
print(f"  Macro F1       : {macro_f*100:.2f}%")
print(f"  Healthy Recall : {recall[0]*100:.2f}%  ← key metric (was {96.72}% in v1)")
print(f"  Train Time     : {total_min:.1f} min")
print("=" * 60)
print("\nNext step: update app.py to load coconut_tree_health_v2/best_model.keras")
