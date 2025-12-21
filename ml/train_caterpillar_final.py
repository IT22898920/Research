"""
Coconut Caterpillar Detection - Final Training Script
======================================================
- Uses prepared dataset (9,108 images)
- MobileNetV2 Transfer Learning
- Complete metrics for Uthpala Miss requirements
- Beautiful training visualizations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'pest_caterpillar', 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'coconut_caterpillar')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

CLASS_NAMES = ['caterpillar', 'healthy']
os.makedirs(MODEL_DIR, exist_ok=True)


def create_data_generators():
    """Create data generators from prepared dataset."""

    print("\n" + "=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)

    # Training - with augmentation (additional on top of pre-augmented)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation/Test - no augmentation
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'validation'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"\n  Train samples:      {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples:       {test_gen.samples}")
    print(f"  Classes: {train_gen.class_indices}")

    return train_gen, val_gen, test_gen


def build_model():
    """Build MobileNetV2 model."""

    print("\n" + "=" * 60)
    print("  BUILDING MODEL")
    print("=" * 60)

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model
    base_model.trainable = False

    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"  Base: MobileNetV2 (frozen)")
    print(f"  Total params: {model.count_params():,}")

    return model


def train_model(model, train_gen, val_gen):
    """Train the model."""

    print("\n" + "=" * 60)
    print("  TRAINING MODEL")
    print("=" * 60)

    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'caterpillar_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    return model, history.history


def find_optimal_threshold(model, val_gen):
    """Find optimal threshold for balanced P/R/F1."""

    print("\n" + "=" * 60)
    print("  FINDING OPTIMAL THRESHOLD")
    print("=" * 60)

    val_gen.reset()
    y_probs = model.predict(val_gen, verbose=0).flatten()
    y_true = val_gen.classes

    best_threshold = 0.5
    best_score = float('inf')

    for thresh in np.arange(0.20, 0.80, 0.02):
        y_pred = (y_probs >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Calculate gaps
        gap0 = max(precision[0], recall[0], f1[0]) - min(precision[0], recall[0], f1[0])
        gap1 = max(precision[1], recall[1], f1[1]) - min(precision[1], recall[1], f1[1])
        score = gap0 + gap1

        if score < best_score:
            best_score = score
            best_threshold = thresh

    print(f"  Optimal threshold: {best_threshold:.2f}")
    return best_threshold


def evaluate_model(model, test_gen, threshold=0.5):
    """Comprehensive evaluation with Uthpala Miss metrics."""

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    test_gen.reset()
    y_probs = model.predict(test_gen, verbose=1).flatten()
    y_true = test_gen.classes
    y_pred = (y_probs >= threshold).astype(int)

    # Classification report
    print("\n" + "-" * 60)
    print("  CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    accuracy = np.mean(y_pred == y_true)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    print("\n" + "-" * 60)
    print("  UTHPALA MISS REQUIREMENTS CHECK")
    print("-" * 60)

    metrics_list = []
    for i, cls in enumerate(CLASS_NAMES):
        gap = max(precision[i], recall[i], f1[i]) - min(precision[i], recall[i], f1[i])
        status = "PASS" if gap < 0.05 else "OK" if gap < 0.10 else "CHECK"

        print(f"\n  {cls.upper()}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall:    {recall[i]:.4f}")
        print(f"    F1-Score:  {f1[i]:.4f}")
        print(f"    P/R/F1 Gap: {gap:.4f} [{status}]")

        metrics_list.append({
            'class': cls,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'prf_gap': float(gap)
        })

    acc_f1_diff = abs(accuracy - macro_f1)
    class_f1_diff = abs(f1[0] - f1[1])

    print("\n" + "-" * 60)
    print("  OVERALL METRICS")
    print("-" * 60)
    print(f"\n  Accuracy:         {accuracy:.4f}")
    print(f"  Macro Precision:  {macro_precision:.4f}")
    print(f"  Macro Recall:     {macro_recall:.4f}")
    print(f"  Macro F1:         {macro_f1:.4f}")
    print(f"\n  Acc vs F1 Diff:   {acc_f1_diff:.4f} [{'PASS' if acc_f1_diff < 0.03 else 'CHECK'}]")
    print(f"  Class F1 Diff:    {class_f1_diff:.4f} [{'PASS' if class_f1_diff < 0.05 else 'CHECK'}]")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': metrics_list,
        'confusion_matrix': cm.tolist(),
        'threshold': float(threshold),
        'acc_f1_diff': float(acc_f1_diff),
        'class_f1_diff': float(class_f1_diff)
    }


def create_visualizations(history, metrics, save_dir):
    """Create beautiful training visualizations."""

    print("\n" + "=" * 60)
    print("  CREATING VISUALIZATIONS")
    print("=" * 60)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 12))

    # 1. Training & Validation Accuracy
    ax1 = fig.add_subplot(2, 2, 1)
    epochs_range = range(1, len(history['accuracy']) + 1)
    ax1.plot(epochs_range, history['accuracy'], 'b-', linewidth=2, label='Training', marker='o', markersize=4)
    ax1.plot(epochs_range, history['val_accuracy'], 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)

    # 2. Training & Validation Loss
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs_range, history['loss'], 'b-', linewidth=2, label='Training', marker='o', markersize=4)
    ax2.plot(epochs_range, history['val_loss'], 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Per-Class Metrics Bar Chart
    ax3 = fig.add_subplot(2, 2, 3)
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    p_vals = [m['precision'] for m in metrics['per_class']]
    r_vals = [m['recall'] for m in metrics['per_class']]
    f_vals = [m['f1'] for m in metrics['per_class']]

    bars1 = ax3.bar(x - width, p_vals, width, label='Precision', color='#3498db')
    bars2 = ax3.bar(x, r_vals, width, label='Recall', color='#2ecc71')
    bars3 = ax3.bar(x + width, f_vals, width, label='F1-Score', color='#e74c3c')

    ax3.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_ylim([0, 1.15])
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # 4. Confusion Matrix
    ax4 = fig.add_subplot(2, 2, 4)
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=[c.capitalize() for c in CLASS_NAMES],
                yticklabels=[c.capitalize() for c in CLASS_NAMES],
                annot_kws={'size': 14})
    ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Predicted', fontsize=12)
    ax4.set_ylabel('Actual', fontsize=12)

    plt.tight_layout(pad=3.0)

    # Save figure
    chart_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Chart saved: training_results.png")

    # Create summary metrics chart
    create_summary_chart(metrics, save_dir)


def create_summary_chart(metrics, save_dir):
    """Create a summary metrics chart."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Overall metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['macro_precision'],
        metrics['macro_recall'],
        metrics['macro_f1']
    ]

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_title('Overall Model Performance', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax.annotate(f'{val:.2%}',
                   xy=(bar.get_x() + bar.get_width() / 2, val),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_performance.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Chart saved: model_performance.png")


def save_model_info(metrics, history, threshold, save_dir):
    """Save model information."""

    print("\n" + "=" * 60)
    print("  SAVING MODEL INFO")
    print("=" * 60)

    info = {
        'model_name': 'coconut_caterpillar_detector',
        'version': 'final',
        'architecture': 'MobileNetV2',
        'input_size': [IMG_SIZE, IMG_SIZE, 3],
        'classes': CLASS_NAMES,
        'optimal_threshold': threshold,
        'dataset': {
            'train_images': 8925,
            'validation_images': 91,
            'test_images': 92,
            'total_images': 9108,
            'augmentation': '20x for training'
        },
        'performance': metrics,
        'training': {
            'epochs_completed': len(history['accuracy']),
            'final_train_accuracy': float(history['accuracy'][-1]),
            'final_val_accuracy': float(history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(history['val_accuracy']))
        },
        'uthpala_miss_requirements': {
            'prf_balanced_per_class': all(m['prf_gap'] < 0.10 for m in metrics['per_class']),
            'accuracy_equals_f1': metrics['acc_f1_diff'] < 0.05,
            'class_f1_similar': metrics['class_f1_diff'] < 0.10
        },
        'date': datetime.now().isoformat()
    }

    with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  Model info saved!")

    return info


def main():
    print("\n" + "=" * 60)
    print("  COCONUT CATERPILLAR DETECTION")
    print("  Final Training with Full Dataset")
    print("=" * 60)

    start = datetime.now()

    # Load data
    train_gen, val_gen, test_gen = create_data_generators()

    # Build model
    model = build_model()

    # Train
    model, history = train_model(model, train_gen, val_gen)

    # Find optimal threshold
    threshold = find_optimal_threshold(model, val_gen)

    # Evaluate
    metrics = evaluate_model(model, test_gen, threshold)

    # Create visualizations
    create_visualizations(history, metrics, MODEL_DIR)

    # Save model info
    info = save_model_info(metrics, history, threshold, MODEL_DIR)

    elapsed = (datetime.now() - start).total_seconds()

    # Final summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)

    print(f"\n  Time: {elapsed/60:.1f} minutes")
    print(f"  Model: {MODEL_DIR}/caterpillar_model.keras")
    print(f"\n  RESULTS:")
    print(f"    Accuracy:    {metrics['accuracy']:.2%}")
    print(f"    Precision:   {metrics['macro_precision']:.2%}")
    print(f"    Recall:      {metrics['macro_recall']:.2%}")
    print(f"    F1-Score:    {metrics['macro_f1']:.2%}")
    print(f"    Threshold:   {threshold:.2f}")

    print("\n  UTHPALA MISS REQUIREMENTS:")
    reqs = info['uthpala_miss_requirements']
    print(f"    P/R/F1 Balanced:    {'PASS' if reqs['prf_balanced_per_class'] else 'CHECK'}")
    print(f"    Accuracy = F1:      {'PASS' if reqs['accuracy_equals_f1'] else 'CHECK'}")
    print(f"    Class F1 Similar:   {'PASS' if reqs['class_f1_similar'] else 'CHECK'}")


if __name__ == '__main__':
    main()
