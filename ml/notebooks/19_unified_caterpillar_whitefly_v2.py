import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
CONFIG = {
    'model_name': 'unified_caterpillar_whitefly_v2',
    'img_size': (224, 224),
    'batch_size': 32,
    'classes': ['caterpillar', 'healthy', 'not_coconut', 'white_fly'],
    'phase1_epochs': 20,
    'phase2_epochs': 30,
    'learning_rate_phase1': 1e-3,
    'learning_rate_phase2': 1e-5,
    'focal_gamma': 2.5,  # Increased for harder examples
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
    'dropout_rate': 0.4,
}

# Paths
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / 'data' / 'raw'
MODEL_SAVE_PATH = BASE_PATH / 'models' / CONFIG['model_name']
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Dataset paths - NO DUPLICATES
CATERPILLAR_PATH = DATA_PATH / 'pest_caterpillar' / 'dataset'
WHITEFLY_PATH = DATA_PATH / 'white_fly'

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print(f"\nModel save path: {MODEL_SAVE_PATH}")

class FocalLossWithSmoothing(keras.losses.Loss):
    """Focal Loss with Label Smoothing for class imbalance."""
    
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, num_classes=4, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
    
    def call(self, y_true, y_pred):
        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        weight = y_true * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
            'num_classes': self.num_classes
        })
        return config

print("Focal Loss with Label Smoothing defined.")
print(f"  Gamma: {CONFIG['focal_gamma']}")
print(f"  Label Smoothing: {CONFIG['label_smoothing']}")

def create_unified_dataset_v2():
    """
    Create unified dataset WITHOUT duplicate healthy/not_coconut data.
    - caterpillar: from pest_caterpillar/dataset
    - white_fly: from white_fly
    - healthy: from pest_caterpillar/dataset ONLY
    - not_coconut: from pest_caterpillar/dataset ONLY
    """
    unified_path = DATA_PATH / 'unified_caterpillar_whitefly_v2'
    
    # Remove old unified dataset if exists
    if unified_path.exists():
        shutil.rmtree(unified_path)
    
    splits = ['train', 'validation', 'test']
    
    # Mapping for folder names
    caterpillar_split_map = {'train': 'train', 'validation': 'validation', 'test': 'test'}
    whitefly_split_map = {'train': 'Training', 'validation': 'validation', 'test': 'test'}
    
    stats = {split: {} for split in splits}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Create directories for all classes
        for cls in CONFIG['classes']:
            (unified_path / split / cls).mkdir(parents=True, exist_ok=True)
        
        # 1. Copy caterpillar images
        cat_split = caterpillar_split_map[split]
        cat_src = CATERPILLAR_PATH / cat_split / 'caterpillar'
        if cat_src.exists():
            count = 0
            for img in cat_src.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img, unified_path / split / 'caterpillar' / img.name)
                    count += 1
            stats[split]['caterpillar'] = count
            print(f"  caterpillar: {count} images")
        
        # 2. Copy white_fly images
        wf_split = whitefly_split_map[split]
        wf_src = WHITEFLY_PATH / wf_split / 'white_fly'
        if wf_src.exists():
            count = 0
            for img in wf_src.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img, unified_path / split / 'white_fly' / img.name)
                    count += 1
            stats[split]['white_fly'] = count
            print(f"  white_fly: {count} images")
        
        # 3. Copy healthy FROM CATERPILLAR DATASET ONLY (no duplicates)
        healthy_src = CATERPILLAR_PATH / cat_split / 'healthy'
        if healthy_src.exists():
            count = 0
            for img in healthy_src.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img, unified_path / split / 'healthy' / img.name)
                    count += 1
            stats[split]['healthy'] = count
            print(f"  healthy: {count} images (from caterpillar dataset only)")
        
        # 4. Copy not_coconut FROM CATERPILLAR DATASET ONLY (no duplicates)
        nc_src = CATERPILLAR_PATH / cat_split / 'not_coconut'
        if nc_src.exists():
            count = 0
            for img in nc_src.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img, unified_path / split / 'not_coconut' / img.name)
                    count += 1
            stats[split]['not_coconut'] = count
            print(f"  not_coconut: {count} images (from caterpillar dataset only)")
    
    return unified_path, stats

# Create the unified dataset
print("="*60)
print("Creating Unified Dataset v2 (No Duplicates)")
print("="*60)
UNIFIED_DATA_PATH, dataset_stats = create_unified_dataset_v2()
print("\n" + "="*60)
print("Dataset Creation Complete!")
print("="*60)

# Create DataFrame for analysis
df_stats = pd.DataFrame(dataset_stats).T
df_stats['Total'] = df_stats.sum(axis=1)
df_stats.loc['Total'] = df_stats.sum()

print("\n" + "="*60)
print("Dataset Distribution (No Duplicates)")
print("="*60)
print(df_stats.to_string())

total_images = int(df_stats.loc['Total', 'Total'])
train_count = int(df_stats.loc['train', 'Total'])
val_count = int(df_stats.loc['validation', 'Total'])
test_count = int(df_stats.loc['test', 'Total'])

print(f"\nTotal Images: {total_images}")
print(f"  Train: {train_count} ({train_count/total_images*100:.1f}%)")
print(f"  Validation: {val_count} ({val_count/total_images*100:.1f}%)")
print(f"  Test: {test_count} ({test_count/total_images*100:.1f}%)")

# Visualize dataset distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Dataset Distribution (v2 - No Duplicates)', fontsize=14, fontweight='bold')

colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
splits_to_plot = ['train', 'validation', 'test']

for idx, split in enumerate(splits_to_plot):
    ax = axes[idx]
    split_data = df_stats.loc[split, CONFIG['classes']]
    bars = ax.bar(CONFIG['classes'], split_data, color=colors)
    ax.set_title(f'{split.capitalize()} Set', fontweight='bold')
    ax.set_ylabel('Number of Images')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, split_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(val)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'dataset_distribution.png'}")

# Class balance pie chart
fig, ax = plt.subplots(figsize=(8, 8))
train_data = df_stats.loc['train', CONFIG['classes']]
explode = (0.02, 0.02, 0.02, 0.02)

wedges, texts, autotexts = ax.pie(
    train_data, 
    labels=CONFIG['classes'],
    colors=colors,
    explode=explode,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75
)
ax.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')

# Add legend with counts
legend_labels = [f'{cls}: {int(train_data[cls])} images' for cls in CONFIG['classes']]
ax.legend(wedges, legend_labels, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'class_distribution_pie.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'class_distribution_pie.png'}")

# Display sample images from each class
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold')

for row, cls in enumerate(CONFIG['classes']):
    class_path = UNIFIED_DATA_PATH / 'train' / cls
    images = list(class_path.glob('*'))[:5]
    
    for col, img_path in enumerate(images):
        ax = axes[row, col]
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(cls, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'sample_images.png'}")

# Strong data augmentation for training (prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='reflect'
)

# No augmentation for validation/test
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    UNIFIED_DATA_PATH / 'train',
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    classes=CONFIG['classes'],
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    UNIFIED_DATA_PATH / 'validation',
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    classes=CONFIG['classes'],
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    UNIFIED_DATA_PATH / 'test',
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    classes=CONFIG['classes'],
    shuffle=False
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Show augmentation examples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')

# Get a sample image
sample_path = list((UNIFIED_DATA_PATH / 'train' / 'caterpillar').glob('*'))[0]
sample_img = plt.imread(sample_path)
sample_img = np.expand_dims(sample_img, 0)

axes[0, 0].imshow(sample_img[0].astype('uint8'))
axes[0, 0].set_title('Original', fontweight='bold')
axes[0, 0].axis('off')

# Generate augmented versions
aug_gen = train_datagen.flow(sample_img, batch_size=1)
for i in range(1, 10):
    row = i // 5
    col = i % 5
    aug_img = next(aug_gen)[0]
    axes[row, col].imshow(aug_img)
    axes[row, col].set_title(f'Augmented {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'augmentation_examples.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'augmentation_examples.png'}")

# Calculate class weights for imbalanced data
y_train = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

print("\nClass Weights (for balanced training):")
print("-" * 40)
for idx, cls in enumerate(CONFIG['classes']):
    count = np.sum(y_train == idx)
    weight = class_weights[idx]
    print(f"  {cls}: {count} samples -> weight: {weight:.4f}")

def build_efficientnet_model(num_classes=4, dropout_rate=0.4):
    """Build EfficientNetB0 model for transfer learning."""
    
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling=None
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(224, 224, 3))
    # EfficientNetB0 expects [0, 255] — rescale back from [0, 1]
    x = layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Dense layers with dropout
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='EfficientNetB0_Unified_v2')
    
    return model, base_model

# Build the model
model, base_model = build_efficientnet_model(
    num_classes=len(CONFIG['classes']),
    dropout_rate=CONFIG['dropout_rate']
)

model.summary()

print("="*60)
print("PHASE 1: Training with Frozen Base Layers")
print("="*60)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate_phase1']),
    loss=FocalLossWithSmoothing(
        gamma=CONFIG['focal_gamma'],
        label_smoothing=CONFIG['label_smoothing'],
        num_classes=len(CONFIG['classes'])
    ),
    metrics=['accuracy']
)

# Callbacks
callbacks_phase1 = [
    ModelCheckpoint(
        str(MODEL_SAVE_PATH / 'best_model_phase1.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print(f"\nTraining for {CONFIG['phase1_epochs']} epochs...")
print(f"Learning rate: {CONFIG['learning_rate_phase1']}")

history_phase1 = model.fit(
    train_generator,
    epochs=CONFIG['phase1_epochs'],
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    class_weight=class_weights,
    verbose=1
)

print(f"\nPhase 1 Best Validation Accuracy: {max(history_phase1.history['val_accuracy']):.4f}")

print("="*60)
print("PHASE 2: Fine-tuning (Unfreezing Top Layers)")
print("="*60)

# Unfreeze top layers of base model
base_model.trainable = True

# Freeze early layers, unfreeze later layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_count = sum([1 for layer in model.layers if layer.trainable])
print(f"Trainable layers: {trainable_count}")

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate_phase2']),
    loss=FocalLossWithSmoothing(
        gamma=CONFIG['focal_gamma'],
        label_smoothing=CONFIG['label_smoothing'],
        num_classes=len(CONFIG['classes'])
    ),
    metrics=['accuracy']
)

# Callbacks for fine-tuning
callbacks_phase2 = [
    ModelCheckpoint(
        str(MODEL_SAVE_PATH / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"\nFine-tuning for {CONFIG['phase2_epochs']} epochs...")
print(f"Learning rate: {CONFIG['learning_rate_phase2']}")

history_phase2 = model.fit(
    train_generator,
    epochs=CONFIG['phase2_epochs'],
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    class_weight=class_weights,
    verbose=1
)

print(f"\nPhase 2 Best Validation Accuracy: {max(history_phase2.history['val_accuracy']):.4f}")

# Combine histories
def combine_histories(h1, h2):
    combined = {}
    for key in h1.history.keys():
        combined[key] = h1.history[key] + h2.history[key]
    return combined

full_history = combine_histories(history_phase1, history_phase2)

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training History - Unified Model v2 (EfficientNetB0)', fontsize=14, fontweight='bold')

epochs_range = range(1, len(full_history['accuracy']) + 1)
phase1_end = len(history_phase1.history['accuracy'])

# Accuracy
ax1 = axes[0, 0]
ax1.plot(epochs_range, full_history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
ax1.plot(epochs_range, full_history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
ax1.axvline(x=phase1_end, color='g', linestyle='--', label='Phase 2 Start')
ax1.set_title('Model Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2 = axes[0, 1]
ax2.plot(epochs_range, full_history['loss'], 'b-', label='Train Loss', linewidth=2)
ax2.plot(epochs_range, full_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax2.axvline(x=phase1_end, color='g', linestyle='--', label='Phase 2 Start')
ax2.set_title('Model Loss', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Learning rate
ax3 = axes[1, 0]
if 'lr' in full_history:
    ax3.plot(epochs_range, full_history['lr'], 'g-', linewidth=2)
elif 'learning_rate' in full_history:
    ax3.plot(epochs_range, full_history['learning_rate'], 'g-', linewidth=2)
ax3.axvline(x=phase1_end, color='r', linestyle='--', label='Phase 2 Start')
ax3.set_title('Learning Rate Schedule', fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Train vs Val gap (overfitting check)
ax4 = axes[1, 1]
gap = [t - v for t, v in zip(full_history['accuracy'], full_history['val_accuracy'])]
ax4.plot(epochs_range, gap, 'purple', linewidth=2)
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=phase1_end, color='g', linestyle='--', label='Phase 2 Start')
ax4.fill_between(epochs_range, gap, 0, alpha=0.3, color='purple')
ax4.set_title('Overfitting Check (Train - Val Accuracy)', fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy Gap')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'training_history.png'}")

print("="*60)
print("Model Evaluation on Test Set")
print("="*60)

# Load best model
best_model = keras.models.load_model(
    MODEL_SAVE_PATH / 'best_model.keras',
    custom_objects={'FocalLossWithSmoothing': FocalLossWithSmoothing}
)

# Evaluate
test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Get predictions
test_generator.reset()
y_pred_probs = best_model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Classification report
print("\n" + "="*60)
print("Classification Report")
print("="*60)
report = classification_report(
    y_true, y_pred,
    target_names=CONFIG['classes'],
    digits=4,
    output_dict=True
)
print(classification_report(y_true, y_pred, target_names=CONFIG['classes'], digits=4))

# Per-class metrics summary
print("\n" + "="*60)
print("Per-Class Metrics Summary")
print("="*60)

metrics_data = []
for cls in CONFIG['classes']:
    metrics_data.append({
        'Class': cls,
        'Precision': report[cls]['precision'],
        'Recall': report[cls]['recall'],
        'F1-Score': report[cls]['f1-score'],
        'Support': int(report[cls]['support'])
    })

metrics_df = pd.DataFrame(metrics_data)
print(metrics_df.to_string(index=False))

# Overall metrics
print("\n" + "="*60)
print("Overall Metrics")
print("="*60)
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1 = report['macro avg']['f1-score']

print(f"  Accuracy:        {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Macro Precision: {macro_precision:.4f}")
print(f"  Macro Recall:    {macro_recall:.4f}")
print(f"  Macro F1-Score:  {macro_f1:.4f}")

# Quality check (Madam's requirements)
print("\n" + "="*60)
print("Quality Check (Madam's Requirements)")
print("="*60)

# Check if P, R, F1 are close for each class
max_diff = 0
for cls in CONFIG['classes']:
    p = report[cls]['precision']
    r = report[cls]['recall']
    f1 = report[cls]['f1-score']
    diff = max(abs(p-r), abs(r-f1), abs(p-f1))
    max_diff = max(max_diff, diff)
    status = "OK" if diff < 0.15 else "WARNING"
    print(f"  {cls}: P={p:.3f}, R={r:.3f}, F1={f1:.3f} -> Max diff: {diff:.3f} [{status}]")

# Check accuracy vs F1
acc_f1_diff = abs(test_accuracy - macro_f1)
print(f"\n  Accuracy vs Macro F1 difference: {acc_f1_diff:.4f}")
print(f"  Accuracy close to F1: {acc_f1_diff < 0.05}")

if max_diff < 0.15:
    print("\n  [PASS] Metrics are well balanced!")
else:
    print(f"\n  [WARNING] Some class imbalance detected (max diff: {max_diff:.3f})")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CONFIG['classes'],
            yticklabels=CONFIG['classes'],
            ax=ax1)
ax1.set_title('Confusion Matrix (Counts)', fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Normalized (percentages)
ax2 = axes[1]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=CONFIG['classes'],
            yticklabels=CONFIG['classes'],
            ax=ax2)
ax2.set_title('Confusion Matrix (Percentages)', fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'confusion_matrix.png'}")

# Per-class metrics bar chart
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(CONFIG['classes']))
width = 0.25

precisions = [report[cls]['precision'] for cls in CONFIG['classes']]
recalls = [report[cls]['recall'] for cls in CONFIG['classes']]
f1_scores = [report[cls]['f1-score'] for cls in CONFIG['classes']]

bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Class', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Per-Class Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CONFIG['classes'])
ax.legend()
ax.set_ylim(0, 1.1)
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'per_class_metrics.png'}")

# Sample predictions
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
fig.suptitle('Sample Predictions', fontsize=14, fontweight='bold')

test_generator.reset()
batch_x, batch_y = next(test_generator)
predictions = best_model.predict(batch_x, verbose=0)

for i in range(16):
    row, col = i // 4, i % 4
    ax = axes[row, col]
    
    img = batch_x[i]
    true_label = CONFIG['classes'][np.argmax(batch_y[i])]
    pred_label = CONFIG['classes'][np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100
    
    ax.imshow(img)
    
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                 color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH / 'sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {MODEL_SAVE_PATH / 'sample_predictions.png'}")

# Save model info
model_info = {
    'model_name': CONFIG['model_name'],
    'version': 'v2',
    'architecture': 'EfficientNetB0',
    'classes': CONFIG['classes'],
    'num_classes': len(CONFIG['classes']),
    'input_size': list(CONFIG['img_size']) + [3],
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'macro_precision': float(macro_precision),
    'macro_recall': float(macro_recall),
    'macro_f1': float(macro_f1),
    'per_class_metrics': {
        cls: {
            'precision': float(report[cls]['precision']),
            'recall': float(report[cls]['recall']),
            'f1-score': float(report[cls]['f1-score']),
            'support': int(report[cls]['support'])
        } for cls in CONFIG['classes']
    },
    'training_config': {
        'batch_size': CONFIG['batch_size'],
        'phase1_epochs': CONFIG['phase1_epochs'],
        'phase2_epochs': CONFIG['phase2_epochs'],
        'focal_gamma': CONFIG['focal_gamma'],
        'label_smoothing': CONFIG['label_smoothing'],
        'dropout_rate': CONFIG['dropout_rate']
    },
    'dataset_info': {
        'total_images': total_images,
        'train_images': train_count,
        'validation_images': val_count,
        'test_images': test_count,
        'note': 'No duplicate healthy/not_coconut data'
    }
}

with open(MODEL_SAVE_PATH / 'model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"Model info saved to: {MODEL_SAVE_PATH / 'model_info.json'}")

print("="*70)
print("                    TRAINING COMPLETE - FINAL SUMMARY")
print("="*70)
print(f"""
Model: {CONFIG['model_name']}
Architecture: EfficientNetB0 (Transfer Learning)
Classes: {', '.join(CONFIG['classes'])}

Test Results:
  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
  Precision: {macro_precision:.4f}
  Recall:    {macro_recall:.4f}
  F1-Score:  {macro_f1:.4f}

Per-Class Performance:
  caterpillar - P: {report['caterpillar']['precision']:.4f}, R: {report['caterpillar']['recall']:.4f}, F1: {report['caterpillar']['f1-score']:.4f}
  healthy     - P: {report['healthy']['precision']:.4f}, R: {report['healthy']['recall']:.4f}, F1: {report['healthy']['f1-score']:.4f}
  not_coconut - P: {report['not_coconut']['precision']:.4f}, R: {report['not_coconut']['recall']:.4f}, F1: {report['not_coconut']['f1-score']:.4f}
  white_fly   - P: {report['white_fly']['precision']:.4f}, R: {report['white_fly']['recall']:.4f}, F1: {report['white_fly']['f1-score']:.4f}

Saved Files:
  - best_model.keras
  - model_info.json
  - training_history.png
  - confusion_matrix.png
  - per_class_metrics.png
  - sample_predictions.png
  - dataset_distribution.png
  - class_distribution_pie.png
  - sample_images.png
  - augmentation_examples.png

Model saved at: {MODEL_SAVE_PATH}
""")
print("="*70)

