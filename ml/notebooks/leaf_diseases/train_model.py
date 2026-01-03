"""
Leaf Disease Detection Model Training Script
Trains the model and saves all outputs for the notebook
"""

# Standard Libraries
import os
import json
import shutil
import random
import warnings
from datetime import datetime
from pathlib import Path
import sys

# Data Processing
import numpy as np
import pandas as pd
from PIL import Image

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0

# Metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("="*70)
print("LEAF DISEASE DETECTION MODEL TRAINING")
print("="*70)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Training Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================

# Model Configuration
MODEL_NAME = "leaf_disease_detection_v1"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3

# Training Configuration
PHASE1_EPOCHS = 25  # Frozen base layers
PHASE2_EPOCHS = 35  # Fine-tuning
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5

# Focal Loss Parameters
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Paths - Use current working directory
SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR.parent.parent  # ml directory
DATA_PATH = BASE_PATH / "data" / "raw" / "stage_2_split"
MODEL_SAVE_PATH = BASE_PATH / "models" / MODEL_NAME

# Create output directory
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("CONFIGURATION")
print("="*70)
print(f"Model Name: {MODEL_NAME}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Phase 1 Epochs: {PHASE1_EPOCHS}")
print(f"Phase 2 Epochs: {PHASE2_EPOCHS}")
print(f"\nData Path: {DATA_PATH}")
print(f"Model Save Path: {MODEL_SAVE_PATH}")

# ============================================================
# DATASET ANALYSIS
# ============================================================

print("\n" + "="*70)
print("DATASET ANALYSIS")
print("="*70)

stats = {}
for split in ['train', 'val', 'test']:
    split_path = DATA_PATH / split
    stats[split] = {}

    for class_dir in split_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            n_images = len(list(class_dir.glob('*')))
            stats[split][class_name] = n_images

# Create DataFrame
df = pd.DataFrame(stats).T
df['Total'] = df.sum(axis=1)

print("\nDataset Distribution:")
print(df)

total_images = df['Total'].sum()
print(f"\nTotal Images: {total_images}")
print(f"  Train: {df.loc['train', 'Total']} ({df.loc['train', 'Total']/total_images*100:.1f}%)")
print(f"  Validation: {df.loc['val', 'Total']} ({df.loc['val', 'Total']/total_images*100:.1f}%)")
print(f"  Test: {df.loc['test', 'Total']} ({df.loc['test', 'Total']/total_images*100:.1f}%)")

# Save dataset stats
df.to_csv(MODEL_SAVE_PATH / 'dataset_stats.csv')

# Plot distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Leaf Disease Dataset Distribution', fontsize=14, fontweight='bold')

colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

for idx, split in enumerate(['train', 'val', 'test']):
    ax = axes[idx]
    data = stats[split]

    classes = list(data.keys())
    counts = list(data.values())

    ax.bar(classes, counts, color=colors)
    ax.set_title(f'{split.upper()} Set ({sum(counts)} images)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Images')
    ax.tick_params(axis='x', rotation=45)

    for i, (c, count) in enumerate(zip(classes, counts)):
        ax.text(i, count + max(counts)*0.02, str(count),
               ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved: dataset_distribution.png")

# ============================================================
# DATA GENERATORS
# ============================================================

print("\n" + "="*70)
print("CREATING DATA GENERATORS")
print("="*70)

# Training Data Generator WITH Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Validation & Test Data Generator WITHOUT Augmentation
val_test_datagen = ImageDataGenerator(rescale=1./255)

print("[OK] Training: WITH augmentation (rotation, shift, flip, zoom)")
print("[OK] Validation/Test: WITHOUT augmentation (only rescaling)")

# Create data generators
train_generator = train_datagen.flow_from_directory(
    DATA_PATH / 'train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

validation_generator = val_test_datagen.flow_from_directory(
    DATA_PATH / 'val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    DATA_PATH / 'test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\nClass Indices:")
print(train_generator.class_indices)

INDEX_TO_CLASS = {v: k for k, v in train_generator.class_indices.items()}
class_names_ordered = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda x: x[1])]

# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n" + "="*70)
print("COMPUTING CLASS WEIGHTS")
print("="*70)

labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass Weights (for imbalanced data):")
for idx, weight in class_weight_dict.items():
    class_name = INDEX_TO_CLASS[idx]
    count = np.sum(labels == idx)
    print(f"  {class_name}: {weight:.4f} (n={count})")

# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        focal_weight = tf.keras.backend.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * focal_weight * cross_entropy

        return tf.keras.backend.sum(focal_loss, axis=-1)

    return focal_loss_fn

print(f"\n[OK] Focal Loss configured (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")

# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "="*70)
print("BUILDING MODEL")
print("="*70)

# Load pre-trained EfficientNetB0
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"\n[OK] Model built successfully")
print(f"  Total params: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"  Non-trainable: {non_trainable_params:,}")

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=PHASE1_LR),
    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
    metrics=['accuracy']
)

print(f"\n[OK] Model compiled (lr={PHASE1_LR})")

# ============================================================
# PHASE 1 TRAINING
# ============================================================

print("\n" + "="*70)
print("PHASE 1: TRAINING WITH FROZEN BASE LAYERS")
print("="*70)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH / 'best_model_phase1.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print(f"\nTraining for up to {PHASE1_EPOCHS} epochs...")
print(f"Base model is FROZEN - only training top layers\n")

start_time = datetime.now()

history_phase1 = model.fit(
    train_generator,
    epochs=PHASE1_EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)

phase1_time = (datetime.now() - start_time).total_seconds() / 60
print(f"\n[OK] Phase 1 completed in {phase1_time:.2f} minutes")
print(f"[OK] Best validation accuracy: {max(history_phase1.history['val_accuracy']):.4f}")

# ============================================================
# PHASE 2 TRAINING
# ============================================================

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING WITH UNFROZEN BASE LAYERS")
print("="*70)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"\n[OK] Unfrozen {trainable_layers} layers in base model")

model.compile(
    optimizer=optimizers.Adam(learning_rate=PHASE2_LR),
    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
    metrics=['accuracy']
)

print(f"[OK] Recompiled with lr={PHASE2_LR}")

callbacks_list_phase2 = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-8,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print(f"\nTraining for up to {PHASE2_EPOCHS} epochs...\n")

start_time = datetime.now()

history_phase2 = model.fit(
    train_generator,
    epochs=PHASE2_EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks_list_phase2,
    verbose=1
)

phase2_time = (datetime.now() - start_time).total_seconds() / 60
print(f"\n[OK] Phase 2 completed in {phase2_time:.2f} minutes")
print(f"[OK] Best validation accuracy: {max(history_phase2.history['val_accuracy']):.4f}")

# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Combine histories
acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
loss = history_phase1.history['loss'] + history_phase2.history['loss']
val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

epochs = range(1, len(acc) + 1)
phase1_end = len(history_phase1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History - Leaf Disease Detection Model', fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
ax1.axvline(x=phase1_end, color='g', linestyle='--', label='Phase 1 → Phase 2')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
ax2.axvline(x=phase1_end, color='g', linestyle='--', label='Phase 1 → Phase 2')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'training_history.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: training_history.png")

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "="*70)
print("MODEL EVALUATION ON TEST SET")
print("="*70)

best_model = tf.keras.models.load_model(
    MODEL_SAVE_PATH / 'best_model.keras',
    custom_objects={'focal_loss_fn': focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)}
)

test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)

print(f"\n[OK] Test Loss: {test_loss:.4f}")
print(f"[OK] Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Get predictions
test_generator.reset()
predictions = best_model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)

report = classification_report(y_true, y_pred, target_names=class_names_ordered, digits=4)
print("\n" + report)

# Detailed metrics
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

metrics_df = pd.DataFrame({
    'Class': class_names_ordered,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print("\n" + "="*70)
print("CLASS-WISE METRICS")
print("="*70)
print("\n" + metrics_df.to_string(index=False))

macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print(f"\n{'='*70}")
print("OVERALL METRICS")
print("="*70)
print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1-Score:  {macro_f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrix - Leaf Disease Detection Model', fontsize=14, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_ordered, yticklabels=class_names_ordered, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Counts')

sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names_ordered, yticklabels=class_names_ordered, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Percentages (%)')

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n[OK] Saved: confusion_matrix.png")

# Per-class metrics plot
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(metrics_df))
width = 0.25

bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#3498db')
bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Metrics: Precision, Recall, F1-Score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Class'])
ax.legend(loc='lower right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: per_class_metrics.png")

# ============================================================
# SAVE MODEL INFO
# ============================================================

total_time = phase1_time + phase2_time
total_epochs = len(history_phase1.history['accuracy']) + len(history_phase2.history['accuracy'])

model_info = {
    'model_name': MODEL_NAME,
    'version': '1.0',
    'architecture': 'EfficientNetB0',
    'framework': 'TensorFlow/Keras',
    'tensorflow_version': tf.__version__,
    'image_size': IMG_SIZE,
    'num_classes': NUM_CLASSES,
    'class_names': class_names_ordered,
    'class_indices': train_generator.class_indices,
    'training': {
        'total_epochs': total_epochs,
        'phase1_epochs': len(history_phase1.history['accuracy']),
        'phase2_epochs': len(history_phase2.history['accuracy']),
        'phase1_lr': PHASE1_LR,
        'phase2_lr': PHASE2_LR,
        'batch_size': BATCH_SIZE,
        'focal_gamma': FOCAL_GAMMA,
        'focal_alpha': FOCAL_ALPHA,
        'total_time_minutes': round(total_time, 2),
        'phase1_time_minutes': round(phase1_time, 2),
        'phase2_time_minutes': round(phase2_time, 2),
        'optimizer': 'Adam',
        'loss_function': 'Focal Loss'
    },
    'performance': {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1)
    },
    'per_class_metrics': {
        class_names_ordered[i]: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
        for i in range(len(class_names_ordered))
    },
    'dataset': {
        'train_samples': train_generator.samples,
        'val_samples': validation_generator.samples,
        'test_samples': test_generator.samples,
        'total_samples': train_generator.samples + validation_generator.samples + test_generator.samples
    },
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(MODEL_SAVE_PATH / 'model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n[OK] Saved: model_info.json")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("="*70)

print(f"\n[INFO] Model: {MODEL_NAME}")
print(f"   Architecture: EfficientNetB0 (Transfer Learning)")
print(f"   Classes: {', '.join(class_names_ordered)}")

print(f"\n[TIME] Training Time: {total_time:.2f} minutes ({total_epochs} epochs)")
print(f"   Phase 1: {len(history_phase1.history['accuracy'])} epochs ({phase1_time:.2f} min)")
print(f"   Phase 2: {len(history_phase2.history['accuracy'])} epochs ({phase2_time:.2f} min)")

print(f"\n[RESULTS] Test Results:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Precision: {macro_precision:.4f}")
print(f"   Recall:    {macro_recall:.4f}")
print(f"   F1-Score:  {macro_f1:.4f}")

print(f"\n[METRICS] Per-Class Performance:")
for i, cls in enumerate(class_names_ordered):
    print(f"   {cls:15s} - P: {precision[i]:.4f}, R: {recall[i]:.4f}, F1: {f1[i]:.4f}")

print(f"\n[FILES] Saved Files:")
print(f"   - best_model.keras")
print(f"   - best_model_phase1.keras")
print(f"   - model_info.json")
print(f"   - dataset_distribution.png")
print(f"   - training_history.png")
print(f"   - confusion_matrix.png")
print(f"   - per_class_metrics.png")
print(f"   - dataset_stats.csv")

print("\n" + "="*70)
print("Model ready for deployment!")
print("="*70)

print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
