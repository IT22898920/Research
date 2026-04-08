"""
Coconut Mite Detection Model v12 - Training Script
Focused on coconut fruit surface patterns

Run: python train_mite_v12.py
"""

import os
import numpy as np
import random
import shutil
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Sklearn imports
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("COCONUT MITE DETECTION MODEL v12")
print("Focused on Coconut Fruit Surface Patterns")
print("=" * 60)

print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set random seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = r'D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research\ml'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'mite - new')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'mite_v12')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'coconut_mite_v12')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

CONFIG = {
    'img_size': 224,
    'batch_size': 32,
    'epochs': 60,
    'learning_rate': 0.0001,
    'patience': 15,
    'min_delta': 0.001,
    'l2_reg': 0.01,
    'dropout': 0.4,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================
# DATA PREPARATION
# ============================================
def count_images(directory):
    counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[class_name] = count
    return counts

def create_split_directories(base_dir):
    for split in ['train', 'validation', 'test']:
        for class_name in ['mite', 'healthy']:
            path = os.path.join(base_dir, split, class_name)
            os.makedirs(path, exist_ok=True)

def split_and_copy_data(raw_dir, processed_dir, train_ratio=0.7, val_ratio=0.15):
    folder_mapping = {'mite': 'mite', 'healty': 'healthy'}

    for raw_name, clean_name in folder_mapping.items():
        class_path = os.path.join(raw_dir, raw_name)
        if not os.path.exists(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        random.shuffle(images)

        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for split, img_list in [('train', train_images),
                                 ('validation', val_images),
                                 ('test', test_images)]:
            dest_dir = os.path.join(processed_dir, split, clean_name)
            for img in img_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        print(f"  {clean_name}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Check if data already split
train_path = os.path.join(PROCESSED_DATA_DIR, 'train')
if os.path.exists(train_path) and len(os.listdir(train_path)) > 0:
    print("Data already split, using existing split...")
else:
    print("Splitting data...")
    create_split_directories(PROCESSED_DATA_DIR)
    split_and_copy_data(RAW_DATA_DIR, PROCESSED_DATA_DIR,
                        CONFIG['train_split'], CONFIG['val_split'])

# Verify split
for split in ['train', 'validation', 'test']:
    split_path = os.path.join(PROCESSED_DATA_DIR, split)
    counts = count_images(split_path)
    total = sum(counts.values())
    print(f"\n{split.upper()}: {counts} (Total: {total})")

# ============================================
# DATA GENERATORS
# ============================================
print("\n" + "=" * 60)
print("CREATING DATA GENERATORS")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=360,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_DIR, 'train'),
    target_size=(CONFIG['img_size'], CONFIG['img_size']),
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_DIR, 'validation'),
    target_size=(CONFIG['img_size'], CONFIG['img_size']),
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(PROCESSED_DATA_DIR, 'test'),
    target_size=(CONFIG['img_size'], CONFIG['img_size']),
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"\nClasses: {class_names}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# ============================================
# CLASS WEIGHTS
# ============================================
train_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights_array))

print("\nClass Weights:")
for idx, class_name in enumerate(class_names):
    print(f"  {class_name}: {class_weights[idx]:.4f}")

# ============================================
# FOCAL LOSS
# ============================================
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        focal = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    return focal_loss_fixed

# ============================================
# MODEL ARCHITECTURE
# ============================================
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])

def create_model(num_classes, l2_reg=0.01, dropout=0.4):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(CONFIG['img_size'], CONFIG['img_size'], 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(CONFIG['img_size'], CONFIG['img_size'], 3))
    x = base_model(inputs, training=False)
    x = squeeze_excite_block(x, ratio=8)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(256, kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

model, base_model = create_model(num_classes, CONFIG['l2_reg'], CONFIG['dropout'])

model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate']),
    loss=focal_loss(CONFIG['focal_gamma'], CONFIG['focal_alpha']),
    metrics=['accuracy']
)

print("Model built: MobileNetV2 + Squeeze-Excite Attention")
print(f"Total parameters: {model.count_params():,}")

# ============================================
# CALLBACKS
# ============================================
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================
# PHASE 1: FROZEN BASE TRAINING
# ============================================
print("\n" + "=" * 60)
print("PHASE 1: Training with frozen base model")
print("=" * 60)

history_phase1 = model.fit(
    train_generator,
    epochs=CONFIG['epochs'],
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_list,
    verbose=1
)

print("\nPhase 1 complete!")

# ============================================
# PHASE 2: FINE-TUNING
# ============================================
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"Unfrozen {trainable_count} layers")

model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate'] / 10),
    loss=focal_loss(CONFIG['focal_gamma'], CONFIG['focal_alpha']),
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_generator,
    epochs=40,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks_list,
    verbose=1
)

print("\nPhase 2 complete!")

# ============================================
# LOAD BEST MODEL & EVALUATE
# ============================================
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

best_model = keras.models.load_model(
    os.path.join(MODEL_DIR, 'best_model.keras'),
    custom_objects={'focal_loss_fixed': focal_loss(CONFIG['focal_gamma'], CONFIG['focal_alpha'])}
)

test_generator.reset()
test_predictions = best_model.predict(test_generator, verbose=1)
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = test_generator.classes

# Calculate metrics
test_accuracy = accuracy_score(test_true_classes, test_pred_classes)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    test_true_classes, test_pred_classes, average='weighted'
)
test_macro_f1 = precision_recall_fscore_support(
    test_true_classes, test_pred_classes, average='macro'
)[2]

if num_classes == 2:
    test_roc_auc = roc_auc_score(test_true_classes, test_predictions[:, 1])
else:
    test_roc_auc = None

print(f"\n{'Metric':<25} {'Value':<10}")
print("-" * 35)
print(f"{'Accuracy':<25} {test_accuracy:.4f}")
print(f"{'Precision (weighted)':<25} {test_precision:.4f}")
print(f"{'Recall (weighted)':<25} {test_recall:.4f}")
print(f"{'F1-Score (weighted)':<25} {test_f1:.4f}")
print(f"{'F1-Score (macro)':<25} {test_macro_f1:.4f}")
if test_roc_auc:
    print(f"{'ROC-AUC':<25} {test_roc_auc:.4f}")

# Class-wise metrics
print("\n" + "=" * 60)
print("CLASS-WISE METRICS")
print("=" * 60)

test_report = classification_report(
    test_true_classes,
    test_pred_classes,
    target_names=class_names,
    digits=4,
    output_dict=True
)

print(classification_report(
    test_true_classes,
    test_pred_classes,
    target_names=class_names,
    digits=4
))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(test_true_classes, test_pred_classes)
print(f"            {class_names[0]:>10} {class_names[1]:>10}")
for i, row in enumerate(cm):
    print(f"{class_names[i]:<12} {row[0]:>10} {row[1]:>10}")

# ============================================
# SAVE MODEL INFO
# ============================================
model_info = {
    'version': 'v12',
    'model_name': 'coconut_mite_v12_fruit_focused',
    'architecture': 'MobileNetV2 + Squeeze-Excite Attention',
    'input_shape': [CONFIG['img_size'], CONFIG['img_size'], 3],
    'num_classes': num_classes,
    'class_names': class_names,
    'class_indices': train_generator.class_indices,
    'training_config': CONFIG,
    'focus': 'Coconut fruit surface patterns - mite damage detection',
    'metrics': {
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1_score': float(test_f1),
        'test_macro_f1': float(test_macro_f1),
        'test_roc_auc': float(test_roc_auc) if test_roc_auc else None,
        'class_wise': {
            c: {
                'precision': float(test_report[c]['precision']),
                'recall': float(test_report[c]['recall']),
                'f1-score': float(test_report[c]['f1-score']),
                'support': int(test_report[c]['support'])
            } for c in class_names
        }
    },
    'dataset': {
        'source': 'mite - new',
        'train_samples': train_generator.samples,
        'validation_samples': val_generator.samples,
        'test_samples': test_generator.samples,
    },
    'trained_at': datetime.now().isoformat(),
}

with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=2)

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel: Coconut Mite v12 (Fruit Surface Focused)")
print(f"Architecture: MobileNetV2 + Squeeze-Excite Attention")
print(f"Classes: {[c.upper() for c in class_names]}")
print(f"\nTEST RESULTS:")
print(f"  Accuracy:  {test_accuracy*100:.2f}%")
print(f"  F1-Score:  {test_f1*100:.2f}%")
print(f"  Macro F1:  {test_macro_f1*100:.2f}%")
if test_roc_auc:
    print(f"  ROC-AUC:   {test_roc_auc*100:.2f}%")
print(f"\nModel saved to: {MODEL_DIR}")
print("=" * 60)
