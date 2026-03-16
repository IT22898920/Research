import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no GUI window needed

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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import random

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Paths
BASE_DIR = os.path.abspath('..')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATASET_DIR = os.path.join(RAW_DATA_DIR, 'coconut_tree_health', 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'coconut_tree_health_v1')

# Source data folders
HEALTHY_SRC = os.path.join(RAW_DATA_DIR, 'Healthy_Coconut_trees')
UNHEALTHY_SRC = os.path.join(RAW_DATA_DIR, 'unhealthy_coconut_trees')

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
PHASE1_EPOCHS = 25   # Frozen base
PHASE2_EPOCHS = 20   # Fine-tuning
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 5e-5

# Data split ratios: 65% train, 15% val, 20% test
TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20   # Ensures 500+ test images

# Classes
class_names = ['healthy', 'unhealthy']

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Base Directory: {BASE_DIR}")
print(f"Healthy Source: {HEALTHY_SRC}")
print(f"Unhealthy Source: {UNHEALTHY_SRC}")
print(f"Dataset Directory: {DATASET_DIR}")
print(f"Model Directory: {MODEL_DIR}")
print(f"Classes: {class_names}")
print(f"Split: {TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")

def collect_all_images(src_dir):
    """Collect all image files from a directory and its subdirectories."""
    all_images = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(valid_ext):
                all_images.append(os.path.join(root, f))
    
    return all_images

# Collect all images
healthy_images = collect_all_images(HEALTHY_SRC)
unhealthy_images = collect_all_images(UNHEALTHY_SRC)

# Shuffle with fixed seed
random.shuffle(healthy_images)
random.shuffle(unhealthy_images)

print(f"Total Healthy images:   {len(healthy_images)}")
print(f"Total Unhealthy images: {len(unhealthy_images)}")
print(f"Total images:           {len(healthy_images) + len(unhealthy_images)}")
print(f"Imbalance ratio:        {len(unhealthy_images) / len(healthy_images):.1f}x more unhealthy")

def split_images(images, train_ratio, val_ratio):
    """Split image list into train/val/test."""
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return images[:train_end], images[train_end:val_end], images[val_end:]

# Split each class
healthy_train, healthy_val, healthy_test = split_images(healthy_images, TRAIN_RATIO, VAL_RATIO)
unhealthy_train, unhealthy_val, unhealthy_test = split_images(unhealthy_images, TRAIN_RATIO, VAL_RATIO)

print("Data Split Summary:")
print(f"{'Split':<10} {'Healthy':>10} {'Unhealthy':>12} {'Total':>8}")
print("-" * 44)
print(f"{'Train':<10} {len(healthy_train):>10} {len(unhealthy_train):>12} {len(healthy_train)+len(unhealthy_train):>8}")
print(f"{'Val':<10} {len(healthy_val):>10} {len(unhealthy_val):>12} {len(healthy_val)+len(unhealthy_val):>8}")
print(f"{'Test':<10} {len(healthy_test):>10} {len(unhealthy_test):>12} {len(healthy_test)+len(unhealthy_test):>8}")
print("-" * 44)
total_test = len(healthy_test) + len(unhealthy_test)
print(f"\nTest set total: {total_test} {'✓ (≥500)' if total_test >= 500 else '✗ (<500)'}")

def build_dataset(healthy_train, healthy_val, healthy_test,
                  unhealthy_train, unhealthy_val, unhealthy_test,
                  dataset_dir):
    """Copy images into ImageDataGenerator-compatible directory structure."""
    
    splits = {
        'train':  {'healthy': healthy_train, 'unhealthy': unhealthy_train},
        'val':    {'healthy': healthy_val,   'unhealthy': unhealthy_val},
        'test':   {'healthy': healthy_test,  'unhealthy': unhealthy_test},
    }
    
    for split, classes in splits.items():
        for cls, file_list in classes.items():
            dest_dir = os.path.join(dataset_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Skip if already populated
            existing = len([f for f in os.listdir(dest_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
            if existing == len(file_list):
                print(f"  {split}/{cls}: already has {existing} files (skipping)")
                continue
            
            # Clear and re-copy
            for f in os.listdir(dest_dir):
                os.remove(os.path.join(dest_dir, f))
            
            for i, src_path in enumerate(file_list):
                ext = os.path.splitext(src_path)[1].lower()
                dest_name = f"{cls}_{split}_{i:05d}{ext}"
                shutil.copy2(src_path, os.path.join(dest_dir, dest_name))
            
            print(f"  {split}/{cls}: copied {len(file_list)} files")

print("Building dataset structure...")
build_dataset(healthy_train, healthy_val, healthy_test,
              unhealthy_train, unhealthy_val, unhealthy_test,
              DATASET_DIR)
print("\nDataset ready!")

# Count images in each split and class
data_summary = {}

print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

for split in ['train', 'val', 'test']:
    split_path = os.path.join(DATASET_DIR, split)
    print(f"\n{split.upper()}:")
    print("-" * 50)
    
    split_total = 0
    for cls in class_names:
        cls_path = os.path.join(split_path, cls)
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
        split_total += count
        print(f"  {cls:<15} {count:>6} images")
        
        if split not in data_summary:
            data_summary[split] = {}
        data_summary[split][cls] = count
    
    print(f"  {'TOTAL':<15} {split_total:>6} images")

print("\n" + "=" * 70)

# Visualize class distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Class Distribution Across Splits', fontsize=14, fontweight='bold')

colors = ['#2ecc71', '#e74c3c']
for idx, split in enumerate(['train', 'val', 'test']):
    counts = [data_summary[split][cls] for cls in class_names]
    bars = axes[idx].bar(class_names, counts, color=colors)
    axes[idx].set_title(f'{split.upper()} Split', fontweight='bold')
    axes[idx].set_ylabel('Number of Images')
    axes[idx].set_xlabel('Class')
    for i, count in enumerate(counts):
        axes[idx].text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

# Imbalance check
train_healthy = data_summary['train']['healthy']
train_unhealthy = data_summary['train']['unhealthy']
imbalance_ratio = max(train_healthy, train_unhealthy) / min(train_healthy, train_unhealthy)
print(f"\nClass imbalance ratio (train): {imbalance_ratio:.1f}x")
print("→ Using Focal Loss + Class Weights to handle imbalance")

test_total = data_summary['test']['healthy'] + data_summary['test']['unhealthy']
print(f"\nTest set total: {test_total} {'✓ (≥500)' if test_total >= 500 else '✗ (<500)'}")

from tensorflow.keras.preprocessing import image

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle('Sample Images from Training Set', fontsize=14, fontweight='bold')

label_colors = {'healthy': '#2ecc71', 'unhealthy': '#e74c3c'}

for row, cls in enumerate(class_names):
    cls_dir = os.path.join(DATASET_DIR, 'train', cls)
    images_list = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    sample_imgs = random.sample(images_list, min(5, len(images_list)))
    
    for col, img_name in enumerate(sample_imgs):
        img_path = os.path.join(cls_dir, img_name)
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    axes[row, 0].text(-0.3, 0.5, cls.upper(),
                      transform=axes[row, 0].transAxes,
                      fontsize=12, fontweight='bold', va='center', ha='right',
                      color=label_colors[cls])

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
plt.show()

# Training generator - heavy augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.25,
    shear_range=0.15,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    fill_mode='nearest'
)

# Val / Test - only rescale (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

print(f"Training samples:   {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples:       {test_generator.samples} {'✓ (≥500)' if test_generator.samples >= 500 else '✗'}")
print(f"Class indices:      {train_generator.class_indices}")

# Compute class weights from training labels
train_labels = train_generator.classes

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}

print("Class Weights:")
for idx, cls in enumerate(class_names):
    print(f"  Class {idx} ({cls}): {class_weight_dict[idx]:.4f}")

print(f"\n→ Healthy gets {class_weight_dict[0]:.1f}x more weight during training")
print("→ This helps the model pay equal attention to both classes")

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for handling class imbalance.
    gamma: Focusing parameter (higher = more focus on hard examples)
    alpha: Class balancing parameter
    """
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        focal_weight = tf.keras.backend.pow(1.0 - y_pred, gamma)
        loss = alpha * focal_weight * cross_entropy
        return tf.keras.backend.sum(loss, axis=-1)
    return focal_loss_fn

print("Focal Loss defined!")
print(f"  gamma=2.0  → Focuses training on hard/misclassified examples")
print(f"  alpha=0.25 → Additional class balancing")

def build_model():
    """Build MobileNetV2 model with transfer learning for tree health classification."""
    
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

model, base_model = build_model()

print("Model Architecture:")
model.summary()
print(f"\nBase model layers: {len(base_model.layers)}")
print(f"Total parameters: {model.count_params():,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

checkpoint_p1 = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'phase1_best.keras'),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
early_stop_p1 = EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
)
reduce_lr_p1 = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
)

print("=" * 70)
print("PHASE 1: Training with Frozen Base")
print("=" * 70)
print(f"LR: {LEARNING_RATE_PHASE1} | Epochs: {PHASE1_EPOCHS} | Class Weights: ON")
print("=" * 70)

start = time.time()

history_p1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_p1, early_stop_p1, reduce_lr_p1],
    verbose=1
)

phase1_time = (time.time() - start) / 60
print(f"\nPhase 1 done in {phase1_time:.1f} min")
print(f"Best val accuracy: {max(history_p1.history['val_accuracy'])*100:.2f}%")

# Unfreeze last 40 layers of base model
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 40

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = sum(1 for layer in model.layers if layer.trainable)
print(f"Total base layers: {len(base_model.layers)}")
print(f"Fine-tuning from layer {fine_tune_at} onward")
print(f"Trainable layers in model: {trainable_count}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

checkpoint_p2 = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'best_model.keras'),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
early_stop_p2 = EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
)
reduce_lr_p2 = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8, verbose=1
)

print("\n" + "=" * 70)
print("PHASE 2: Fine-tuning")
print("=" * 70)
print(f"LR: {LEARNING_RATE_PHASE2} | Epochs: {PHASE2_EPOCHS} | Class Weights: ON")
print("=" * 70)

start = time.time()

history_p2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_p2, early_stop_p2, reduce_lr_p2],
    verbose=1
)

phase2_time = (time.time() - start) / 60
total_time = phase1_time + phase2_time

print(f"\nPhase 2 done in {phase2_time:.1f} min")
print(f"Total training time: {total_time:.1f} min")
print(f"Best val accuracy: {max(history_p2.history['val_accuracy'])*100:.2f}%")

history_combined = {
    'accuracy':     history_p1.history['accuracy']     + history_p2.history['accuracy'],
    'val_accuracy': history_p1.history['val_accuracy'] + history_p2.history['val_accuracy'],
    'loss':         history_p1.history['loss']         + history_p2.history['loss'],
    'val_loss':     history_p1.history['val_loss']     + history_p2.history['val_loss'],
}

phase1_end = len(history_p1.history['accuracy']) - 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History - Coconut Tree Health Model v1', fontsize=14, fontweight='bold')

axes[0].plot(history_combined['accuracy'],     label='Train Accuracy', linewidth=2)
axes[0].plot(history_combined['val_accuracy'], label='Val Accuracy',   linewidth=2)
axes[0].axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning starts', alpha=0.7)
axes[0].set_title('Model Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history_combined['loss'],     label='Train Loss', linewidth=2)
axes[1].plot(history_combined['val_loss'], label='Val Loss',   linewidth=2)
axes[1].axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning starts', alpha=0.7)
axes[1].set_title('Model Loss', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()

# Overfitting check
final_train_acc = history_combined['accuracy'][-1]
final_val_acc   = history_combined['val_accuracy'][-1]
acc_gap = abs(final_train_acc - final_val_acc)

print("\n" + "=" * 70)
print("OVERFITTING CHECK")
print("=" * 70)
print(f"Final Train Accuracy: {final_train_acc*100:.2f}%")
print(f"Final Val Accuracy:   {final_val_acc*100:.2f}%")
print(f"Accuracy Gap:         {acc_gap*100:.2f}%")

if acc_gap < 0.05:
    print("✓ No significant overfitting!")
elif acc_gap < 0.10:
    print("⚠ Minor overfitting - acceptable")
else:
    print("⚠⚠ Overfitting detected")
print("=" * 70)

best_model = keras.models.load_model(
    os.path.join(MODEL_DIR, 'best_model.keras'),
    custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
)
print("Best model loaded!")

test_generator.reset()
predictions = best_model.predict(test_generator, verbose=1)

y_true = test_generator.classes
y_pred = np.argmax(predictions, axis=1)
y_conf = np.max(predictions, axis=1)

test_accuracy = np.mean(y_true == y_pred)
print(f"\nTest Set Size: {len(y_true)} images")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)
macro_p = np.mean(precision)
macro_r = np.mean(recall)
macro_f = np.mean(f1)

print("=" * 85)
print("DETAILED CLASS-WISE METRICS")
print("=" * 85)
print(f"\n{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
print("-" * 85)

for i, cls in enumerate(class_names):
    print(f"{cls:<15} {precision[i]*100:>11.2f}% {recall[i]*100:>11.2f}% {f1[i]*100:>11.2f}% {support[i]:>10}")

print("-" * 85)
print(f"{'Macro Avg':<15} {macro_p*100:>11.2f}% {macro_r*100:>11.2f}% {macro_f*100:>11.2f}%")
print("=" * 85)

# --- Metric Balance Check ---
print("\n" + "=" * 85)
print("METRIC BALANCE CHECK (P, R, F1 should be close to each other per class)")
print("=" * 85)

all_balanced = True
for i, cls in enumerate(class_names):
    p, r, f = precision[i], recall[i], f1[i]
    max_diff = max(abs(p-r), abs(p-f), abs(r-f))
    balanced = max_diff < 0.10
    if not balanced:
        all_balanced = False
    status = "✓" if balanced else "⚠"
    print(f"\n{cls.upper()}:")
    print(f"  Precision: {p*100:.2f}%")
    print(f"  Recall:    {r*100:.2f}%")
    print(f"  F1-Score:  {f*100:.2f}%")
    print(f"  Max diff:  {max_diff*100:.2f}%  {status}")

# --- Cross-class balance check ---
print("\n" + "-" * 85)
print("CROSS-CLASS BALANCE CHECK (values should be similar for all classes)")
print("-" * 85)
f1_diff = abs(f1[0] - f1[1])
p_diff  = abs(precision[0] - precision[1])
r_diff  = abs(recall[0] - recall[1])
print(f"  F1-Score difference between classes:  {f1_diff*100:.2f}% {'✓' if f1_diff < 0.10 else '⚠'}")
print(f"  Precision difference between classes: {p_diff*100:.2f}%  {'✓' if p_diff < 0.10 else '⚠'}")
print(f"  Recall difference between classes:    {r_diff*100:.2f}%  {'✓' if r_diff < 0.10 else '⚠'}")

# Accuracy vs F1 alignment
acc_f1_diff = abs(test_accuracy - macro_f)
print(f"\n  Accuracy:  {test_accuracy*100:.2f}%")
print(f"  Macro F1:  {macro_f*100:.2f}%")
print(f"  Difference:{acc_f1_diff*100:.2f}%  {'✓' if acc_f1_diff < 0.05 else '⚠'}")
print("=" * 85)

cm = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrix - Coconut Tree Health Model v1', fontsize=14, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1],
            cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('Confusion Matrix (Percentages)', fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\nConfusion Matrix Details:")
print("-" * 50)
for i, true_cls in enumerate(class_names):
    for j, pred_cls in enumerate(class_names):
        count = cm[i, j]
        pct = cm_pct[i, j]
        if i == j:
            print(f"✓ {true_cls} correctly classified: {count} ({pct:.1f}%)")
        elif count > 0:
            print(f"✗ {true_cls} misclassified as {pred_cls}: {count} ({pct:.1f}%)")

print("\n" + "=" * 85)
print("FULL CLASSIFICATION REPORT")
print("=" * 85)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print("=" * 85)

# Confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Prediction Confidence Distribution', fontsize=13, fontweight='bold')

correct_conf = y_conf[y_true == y_pred]
wrong_conf   = y_conf[y_true != y_pred]

axes[0].hist(correct_conf, bins=20, color='#2ecc71', edgecolor='black', alpha=0.8)
axes[0].set_title(f'Correct Predictions (n={len(correct_conf)})', fontweight='bold')
axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Count')
axes[0].axvline(x=np.mean(correct_conf), color='darkgreen', linestyle='--',
                label=f'Mean: {np.mean(correct_conf):.2f}')
axes[0].legend()

if len(wrong_conf) > 0:
    axes[1].hist(wrong_conf, bins=20, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1].set_title(f'Wrong Predictions (n={len(wrong_conf)})', fontweight='bold')
    axes[1].set_xlabel('Confidence'); axes[1].set_ylabel('Count')
    axes[1].axvline(x=np.mean(wrong_conf), color='darkred', linestyle='--',
                    label=f'Mean: {np.mean(wrong_conf):.2f}')
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, 'No wrong predictions!', ha='center', va='center',
                fontsize=14, color='green', transform=axes[1].transAxes)
    axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

filenames = test_generator.filenames

correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
wrong_idx   = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

print(f"Total: {len(y_true)} | Correct: {len(correct_idx)} | Wrong: {len(wrong_idx)}")
print(f"Accuracy: {len(correct_idx)/len(y_true)*100:.2f}%")

# Correct predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle('CORRECT Predictions (Sample)', fontsize=14, fontweight='bold', color='green')

sample_correct = random.sample(correct_idx, min(10, len(correct_idx)))
for idx, i in enumerate(sample_correct):
    row, col = idx // 5, idx % 5
    img_path = os.path.join(DATASET_DIR, 'test', filenames[i])
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    conf = predictions[i][y_pred[i]] * 100
    axes[row, col].set_title(
        f'T:{class_names[y_true[i]]}\nP:{class_names[y_pred[i]]} ({conf:.1f}%)',
        fontsize=8, color='green'
    )

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'correct_predictions.png'), dpi=150, bbox_inches='tight')
plt.show()

# Wrong predictions
if len(wrong_idx) > 0:
    n_wrong = min(10, len(wrong_idx))
    rows = max(1, (n_wrong + 4) // 5)
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3.5 * rows))
    fig.suptitle(f'WRONG Predictions (All {len(wrong_idx)})', fontsize=14, fontweight='bold', color='red')
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, i in enumerate(wrong_idx[:n_wrong]):
        row, col = idx // 5, idx % 5
        img_path = os.path.join(DATASET_DIR, 'test', filenames[i])
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        conf = predictions[i][y_pred[i]] * 100
        axes[row, col].set_title(
            f'T:{class_names[y_true[i]]}\nP:{class_names[y_pred[i]]} ({conf:.1f}%)',
            fontsize=8, color='red'
        )
    
    for idx in range(n_wrong, rows * 5):
        axes[idx // 5, idx % 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'wrong_predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("\n Perfect! No wrong predictions!")

model_info = {
    'model_name': 'coconut_tree_health_v1',
    'architecture': 'MobileNetV2',
    'version': 'v1',
    'classes': class_names,
    'num_classes': len(class_names),
    'input_shape': [IMG_SIZE, IMG_SIZE, 3],
    'training': {
        'phase1_epochs': PHASE1_EPOCHS,
        'phase2_epochs': PHASE2_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate_phase1': LEARNING_RATE_PHASE1,
        'learning_rate_phase2': LEARNING_RATE_PHASE2,
        'loss_function': 'Focal Loss (gamma=2.0, alpha=0.25)',
        'optimizer': 'Adam',
        'class_weights': class_weight_dict,
        'training_time_minutes': round(total_time, 1),
        'final_train_accuracy': float(final_train_acc),
        'final_val_accuracy': float(final_val_acc)
    },
    'data': {
        'split': '65% train / 15% val / 20% test',
        'train_samples': train_generator.samples,
        'val_samples': val_generator.samples,
        'test_samples': test_generator.samples,
        'train_healthy': data_summary['train']['healthy'],
        'train_unhealthy': data_summary['train']['unhealthy'],
        'imbalance_ratio': float(imbalance_ratio)
    },
    'test_performance': {
        'accuracy': float(test_accuracy),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f),
        'healthy_precision': float(precision[0]),
        'healthy_recall': float(recall[0]),
        'healthy_f1': float(f1[0]),
        'unhealthy_precision': float(precision[1]),
        'unhealthy_recall': float(recall[1]),
        'unhealthy_f1': float(f1[1])
    },
    'augmentation': {
        'rotation_range': 40,
        'width_shift_range': 0.25,
        'height_shift_range': 0.25,
        'horizontal_flip': True,
        'vertical_flip': True,
        'zoom_range': 0.25,
        'shear_range': 0.15,
        'brightness_range': [0.7, 1.3],
        'channel_shift_range': 20.0
    }
}

model_info_path = os.path.join(MODEL_DIR, 'model_info.json')
with open(model_info_path, 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"Model info saved: {model_info_path}")
print("\nFiles created:")
for fname in os.listdir(MODEL_DIR):
    fpath = os.path.join(MODEL_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname:<40} ({size_kb:.1f} KB)")

print("\n" + "=" * 85)
print("  COCONUT TREE HEALTH MODEL v1 — FINAL SUMMARY")
print("=" * 85)
print()
print("  Model Information:")
print("  " + "-" * 65)
print(f"  Name:            Coconut Tree Health Detection Model v1")
print(f"  Architecture:    MobileNetV2 (Transfer Learning)")
print(f"  Loss:            Focal Loss (gamma=2.0) + Class Weights")
print(f"  Input Size:      {IMG_SIZE}x{IMG_SIZE}x3")
print(f"  Classes:         {class_names}")
print()
print("  Training Summary:")
print("  " + "-" * 65)
print(f"  Phase 1 (Frozen):    LR={LEARNING_RATE_PHASE1}")
print(f"  Phase 2 (Fine-tune): LR={LEARNING_RATE_PHASE2}")
print(f"  Training Time:       {total_time:.1f} minutes")
print(f"  Final Train Acc:     {final_train_acc*100:.2f}%")
print(f"  Final Val Acc:       {final_val_acc*100:.2f}%")
print()
print("  Data Split (65/15/20):")
print("  " + "-" * 65)
print(f"  Train: {train_generator.samples} | Val: {val_generator.samples} | Test: {test_generator.samples} ✓")
print()
print("  Test Performance:")
print("  " + "-" * 65)
print(f"  Accuracy:        {test_accuracy*100:.2f}%")
print(f"  Macro Precision: {macro_p*100:.2f}%")
print(f"  Macro Recall:    {macro_r*100:.2f}%")
print(f"  Macro F1-Score:  {macro_f*100:.2f}%")
print()
print("  Class-wise Performance:")
print("  " + "-" * 65)
for i, cls in enumerate(class_names):
    max_diff = max(abs(precision[i]-recall[i]), abs(precision[i]-f1[i]), abs(recall[i]-f1[i]))
    status = "✓" if max_diff < 0.10 else "⚠"
    print(f"  {cls.upper():12} P={precision[i]*100:.2f}% R={recall[i]*100:.2f}% F1={f1[i]*100:.2f}% {status}")
print()
print("  Quality Checks:")
print("  " + "-" * 65)
print(f"  Test set ≥ 500:          {test_generator.samples} {'✓' if test_generator.samples >= 500 else '✗'}")
print(f"  Train-Val gap:           {acc_gap*100:.2f}% {'✓' if acc_gap < 0.05 else '⚠'}")
print(f"  Accuracy-F1 alignment:   {acc_f1_diff*100:.2f}% {'✓' if acc_f1_diff < 0.05 else '⚠'}")
print(f"  F1 cross-class diff:     {f1_diff*100:.2f}% {'✓' if f1_diff < 0.10 else '⚠'}")
print()
print("  Saved to:")
print(f"  {MODEL_DIR}/best_model.keras")
print(f"  {MODEL_DIR}/model_info.json")
print()
print("=" * 85)
print("                         TRAINING COMPLETE!")
print("=" * 85)
