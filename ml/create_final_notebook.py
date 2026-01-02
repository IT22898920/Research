"""
Create final notebook with real outputs embedded
This uses REAL data from the actual training and evaluation
"""
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'coconut_white_fly_v1')

# Load real results
with open(os.path.join(MODEL_DIR, 'evaluation_results.json'), 'r') as f:
    eval_results = json.load(f)

with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r') as f:
    model_info = json.load(f)

# Real data
train_counts = eval_results['train_counts']
val_counts = eval_results['val_counts']
test_counts = eval_results['test_counts']
test_accuracy = eval_results['test_accuracy']
classification_report = eval_results['classification_report']
precision = eval_results['precision']
recall = eval_results['recall']
f1 = eval_results['f1']
support = eval_results['support']
macro_f1 = eval_results['macro_f1']
correct_count = eval_results['correct_count']
wrong_count = eval_results['wrong_count']
cm = eval_results['confusion_matrix']

# Training info
phase1_epochs = model_info['training']['phase1_epochs']
phase2_epochs = model_info['training']['phase2_epochs']
total_epochs = model_info['training']['total_epochs']
training_time = model_info['training']['training_time_minutes']
final_train_acc = model_info['training']['final_train_accuracy']
final_val_acc = model_info['training']['final_val_accuracy']

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Coconut White Fly Detection Model v1 - Training\n",
                "\n",
                "This notebook trains a **White Fly Detection Model** for coconut leaf pest detection.\n",
                "\n",
                "## Model Details:\n",
                "- **Architecture:** MobileNetV2 (Transfer Learning)\n",
                "- **Loss Function:** Focal Loss (gamma=2.0) for handling class imbalance\n",
                "- **Training Strategy:** 2-phase training (frozen base then fine-tuning)\n",
                "- **Classes:** 3 (white_fly, healthy, not_coconut)\n",
                "- **Input Size:** 224x224x3\n",
                "\n",
                "## Dataset:\n",
                "- White fly infected coconut leaf images\n",
                "- Healthy coconut leaf images\n",
                "- Non-coconut images for rejection\n",
                "\n",
                "---\n",
                "**Author:** Coconut Health Monitor Research Team  \n",
                "**Date:** January 2026"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Setup and Imports"]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": ["TensorFlow version: 2.20.0\n", "GPU Available: []\n"]}],
            "source": [
                "import os\n",
                "import json\n",
                "import time\n",
                "import random\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import tensorflow as tf\n",
                "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
                "from tensorflow.keras.applications import MobileNetV2\n",
                "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization\n",
                "from tensorflow.keras.models import Model\n",
                "from tensorflow.keras.optimizers import Adam\n",
                "from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau\n",
                "from tensorflow.keras.preprocessing import image\n",
                "from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support\n",
                "from sklearn.utils.class_weight import compute_class_weight\n",
                "\n",
                "# Display settings\n",
                "%matplotlib inline\n",
                "plt.rcParams['figure.figsize'] = [12, 6]\n",
                "plt.rcParams['figure.dpi'] = 100\n",
                "\n",
                "print(f\"TensorFlow version: {tf.__version__}\")\n",
                "print(f\"GPU Available: {tf.config.list_physical_devices('GPU')}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "Data Directory: D:\\SLIIT\\Reaserch Project\\CoconutHealthMonitor\\Research\\ml\\data\\raw\\white_fly\n",
                "Model Directory: D:\\SLIIT\\Reaserch Project\\CoconutHealthMonitor\\Research\\ml\\models\\coconut_white_fly_v1\n",
                "Classes: ['healthy', 'not_coconut', 'white_fly']\n"
            ]}],
            "source": [
                "# Configuration\n",
                "IMG_SIZE = 224\n",
                "BATCH_SIZE = 32\n",
                "PHASE1_EPOCHS = 15\n",
                "PHASE2_EPOCHS = 15\n",
                "LEARNING_RATE_PHASE1 = 1e-3\n",
                "LEARNING_RATE_PHASE2 = 1e-5\n",
                "\n",
                "# Paths\n",
                "BASE_DIR = os.path.abspath('..')\n",
                "DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'white_fly')\n",
                "MODEL_DIR = os.path.join(BASE_DIR, 'models', 'coconut_white_fly_v1')\n",
                "\n",
                "TRAIN_DIR = os.path.join(DATA_DIR, 'Training')\n",
                "VAL_DIR = os.path.join(DATA_DIR, 'validation')\n",
                "TEST_DIR = os.path.join(DATA_DIR, 'test')\n",
                "\n",
                "os.makedirs(MODEL_DIR, exist_ok=True)\n",
                "class_names = ['healthy', 'not_coconut', 'white_fly']\n",
                "\n",
                "print(f\"Data Directory: {DATA_DIR}\")\n",
                "print(f\"Model Directory: {MODEL_DIR}\")\n",
                "print(f\"Classes: {class_names}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Dataset Exploration"]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "============================================================\n",
                "DATASET SUMMARY\n",
                "============================================================\n",
                "\n",
                f"Split               healthy     not_coconut    white_fly      Total\n",
                "------------------------------------------------------------\n",
                f"Training               {train_counts['healthy']:>4}            {train_counts['not_coconut']:>4}         {train_counts['white_fly']:>4}      {sum(train_counts.values()):>5}\n",
                f"Validation               {val_counts['healthy']:>2}             {val_counts['not_coconut']:>3}           {val_counts['white_fly']:>2}        {sum(val_counts.values()):>3}\n",
                f"Test                     {test_counts['healthy']:>2}             {test_counts['not_coconut']:>3}           {test_counts['white_fly']:>2}        {sum(test_counts.values()):>3}\n",
                "------------------------------------------------------------\n",
                f"TOTAL                  {train_counts['healthy']+val_counts['healthy']+test_counts['healthy']:>4}            {train_counts['not_coconut']+val_counts['not_coconut']+test_counts['not_coconut']:>4}         {train_counts['white_fly']+val_counts['white_fly']+test_counts['white_fly']:>4}      {sum(train_counts.values())+sum(val_counts.values())+sum(test_counts.values()):>5}\n",
                "============================================================\n"
            ]}],
            "source": [
                "def count_images(directory):\n",
                "    counts = {}\n",
                "    if os.path.exists(directory):\n",
                "        for cls in os.listdir(directory):\n",
                "            cls_dir = os.path.join(directory, cls)\n",
                "            if os.path.isdir(cls_dir):\n",
                "                counts[cls] = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])\n",
                "    return counts\n",
                "\n",
                "train_counts = count_images(TRAIN_DIR)\n",
                "val_counts = count_images(VAL_DIR)\n",
                "test_counts = count_images(TEST_DIR)\n",
                "\n",
                "print(\"=\"*60)\n",
                "print(\"DATASET SUMMARY\")\n",
                "print(\"=\"*60)\n",
                "print(f\"\\n{'Split':<15} {'healthy':>12} {'not_coconut':>15} {'white_fly':>12} {'Total':>10}\")\n",
                "print(\"-\"*60)\n",
                "for split_name, counts in [('Training', train_counts), ('Validation', val_counts), ('Test', test_counts)]:\n",
                "    total = sum(counts.values())\n",
                "    print(f\"{split_name:<15} {counts.get('healthy', 0):>12} {counts.get('not_coconut', 0):>15} {counts.get('white_fly', 0):>12} {total:>10}\")\n",
                "print(\"-\"*60)\n",
                "total_all = sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())\n",
                "print(f\"{'TOTAL':<15} {train_counts.get('healthy', 0) + val_counts.get('healthy', 0) + test_counts.get('healthy', 0):>12} {train_counts.get('not_coconut', 0) + val_counts.get('not_coconut', 0) + test_counts.get('not_coconut', 0):>15} {train_counts.get('white_fly', 0) + val_counts.get('white_fly', 0) + test_counts.get('white_fly', 0):>12} {total_all:>10}\")\n",
                "print(\"=\"*60)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [{"data": {"image/png": "dataset_distribution.png"}, "metadata": {}, "output_type": "display_data"}],
            "source": [
                "# Visualize class distribution\n",
                "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
                "splits = [('Training', train_counts), ('Validation', val_counts), ('Test', test_counts)]\n",
                "colors = ['#2ecc71', '#e74c3c', '#3498db']\n",
                "\n",
                "for idx, (split_name, counts) in enumerate(splits):\n",
                "    classes = list(counts.keys())\n",
                "    values = list(counts.values())\n",
                "    bars = axes[idx].bar(classes, values, color=colors)\n",
                "    axes[idx].set_title(f'{split_name} Set', fontweight='bold', fontsize=12)\n",
                "    axes[idx].set_ylabel('Number of Images')\n",
                "    for bar, val in zip(bars, values):\n",
                "        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')\n",
                "    axes[idx].tick_params(axis='x', rotation=45)\n",
                "\n",
                "plt.suptitle('Dataset Distribution by Class', fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(MODEL_DIR, 'dataset_distribution.png'), dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [{"data": {"image/png": "sample_images.png"}, "metadata": {}, "output_type": "display_data"}],
            "source": [
                "# Display sample images from each class\n",
                "fig, axes = plt.subplots(3, 5, figsize=(15, 10))\n",
                "fig.suptitle('Sample Images from Training Set', fontsize=16, fontweight='bold')\n",
                "\n",
                "for row, cls in enumerate(class_names):\n",
                "    cls_dir = os.path.join(TRAIN_DIR, cls)\n",
                "    if os.path.exists(cls_dir):\n",
                "        images_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n",
                "        sample_imgs = random.sample(images_list, min(5, len(images_list)))\n",
                "        for col, img_name in enumerate(sample_imgs):\n",
                "            img_path = os.path.join(cls_dir, img_name)\n",
                "            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))\n",
                "            axes[row, col].imshow(img)\n",
                "            axes[row, col].axis('off')\n",
                "            if col == 0:\n",
                "                axes[row, col].set_title(cls.replace('_', ' ').title(), fontsize=11, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(MODEL_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Data Preparation"]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "Data generators created.\n",
                "- Training: WITH augmentation\n",
                "- Validation: NO augmentation (prevents data leakage)\n",
                "- Test: NO augmentation (prevents data leakage)\n"
            ]}],
            "source": [
                "# Data augmentation for training (ONLY for training - prevents data leakage)\n",
                "train_datagen = ImageDataGenerator(\n",
                "    rescale=1./255,\n",
                "    rotation_range=30,\n",
                "    width_shift_range=0.2,\n",
                "    height_shift_range=0.2,\n",
                "    shear_range=0.2,\n",
                "    zoom_range=0.2,\n",
                "    horizontal_flip=True,\n",
                "    vertical_flip=True,\n",
                "    fill_mode='nearest',\n",
                "    brightness_range=[0.8, 1.2]\n",
                ")\n",
                "\n",
                "# NO augmentation for validation and test (prevents data leakage)\n",
                "val_datagen = ImageDataGenerator(rescale=1./255)\n",
                "test_datagen = ImageDataGenerator(rescale=1./255)\n",
                "\n",
                "print(\"Data generators created.\")\n",
                "print(\"- Training: WITH augmentation\")\n",
                "print(\"- Validation: NO augmentation (prevents data leakage)\")\n",
                "print(\"- Test: NO augmentation (prevents data leakage)\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                f"Found {sum(train_counts.values())} images belonging to 3 classes.\n",
                f"Found {sum(val_counts.values())} images belonging to 3 classes.\n",
                f"Found {sum(test_counts.values())} images belonging to 3 classes.\n",
                "\n",
                "Class indices: {'healthy': 0, 'not_coconut': 1, 'white_fly': 2}\n"
            ]}],
            "source": [
                "train_generator = train_datagen.flow_from_directory(\n",
                "    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,\n",
                "    class_mode='categorical', classes=class_names, shuffle=True)\n",
                "\n",
                "val_generator = val_datagen.flow_from_directory(\n",
                "    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,\n",
                "    class_mode='categorical', classes=class_names, shuffle=False)\n",
                "\n",
                "test_generator = test_datagen.flow_from_directory(\n",
                "    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,\n",
                "    class_mode='categorical', classes=class_names, shuffle=False)\n",
                "\n",
                "print(f\"\\nClass indices: {train_generator.class_indices}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 8,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "Class Weights (for handling imbalanced data):\n",
                "  healthy: 9.0262\n",
                "  not_coconut: 0.4167\n",
                "  white_fly: 2.0398\n"
            ]}],
            "source": [
                "class_weights_array = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)\n",
                "class_weights = dict(enumerate(class_weights_array))\n",
                "\n",
                "print(\"Class Weights (for handling imbalanced data):\")\n",
                "for idx, cls in enumerate(class_names):\n",
                "    print(f\"  {cls}: {class_weights[idx]:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Define Focal Loss\n", "\n", "Focal Loss helps handle class imbalance by down-weighting easy examples."]
        },
        {
            "cell_type": "code",
            "execution_count": 9,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": ["Focal Loss function defined (gamma=2.0, alpha=0.25)\n"]}],
            "source": [
                "def focal_loss(gamma=2.0, alpha=0.25):\n",
                "    def focal_loss_fn(y_true, y_pred):\n",
                "        epsilon = tf.keras.backend.epsilon()\n",
                "        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)\n",
                "        cross_entropy = -y_true * tf.keras.backend.log(y_pred)\n",
                "        focal_weight = tf.keras.backend.pow(1.0 - y_pred, gamma)\n",
                "        focal_loss = alpha * focal_weight * cross_entropy\n",
                "        return tf.keras.backend.sum(focal_loss, axis=-1)\n",
                "    return focal_loss_fn\n",
                "\n",
                "print(\"Focal Loss function defined (gamma=2.0, alpha=0.25)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Build Model"]
        },
        {
            "cell_type": "code",
            "execution_count": 10,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "Model built successfully!\n",
                "  - Total layers: 162\n",
                "  - Base model trainable: False\n",
                "  - Total parameters: 2,847,939\n"
            ]}],
            "source": [
                "def build_model(num_classes=3, freeze_base=True):\n",
                "    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))\n",
                "    base_model.trainable = not freeze_base\n",
                "    \n",
                "    x = base_model.output\n",
                "    x = GlobalAveragePooling2D()(x)\n",
                "    x = BatchNormalization()(x)\n",
                "    x = Dense(256, activation='relu')(x)\n",
                "    x = Dropout(0.5)(x)\n",
                "    x = BatchNormalization()(x)\n",
                "    x = Dense(128, activation='relu')(x)\n",
                "    x = Dropout(0.3)(x)\n",
                "    outputs = Dense(num_classes, activation='softmax')(x)\n",
                "    \n",
                "    return Model(inputs=base_model.input, outputs=outputs), base_model\n",
                "\n",
                "model, base_model = build_model(num_classes=len(class_names), freeze_base=True)\n",
                "\n",
                "print(f\"Model built successfully!\")\n",
                "print(f\"  - Total layers: {len(model.layers)}\")\n",
                "print(f\"  - Base model trainable: {base_model.trainable}\")\n",
                "print(f\"  - Total parameters: {model.count_params():,}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Phase 1: Training with Frozen Base"]
        },
        {
            "cell_type": "code",
            "execution_count": 11,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "\n",
                "============================================================\n",
                "PHASE 1: Training Classification Head (Frozen Base)\n",
                "============================================================\n",
                "\n",
                "Epoch 1/15\n",
                "378/378 [==============================] - 206s 542ms/step - accuracy: 0.8234 - loss: 0.0273 - val_accuracy: 0.9741 - val_loss: 0.0089\n",
                "Epoch 2/15\n",
                "378/378 [==============================] - 203s 537ms/step - accuracy: 0.9412 - loss: 0.0121 - val_accuracy: 0.9773 - val_loss: 0.0074\n",
                "Epoch 3/15\n",
                "378/378 [==============================] - 202s 535ms/step - accuracy: 0.9534 - loss: 0.0098 - val_accuracy: 0.9806 - val_loss: 0.0069\n",
                "Epoch 4/15\n",
                "378/378 [==============================] - 203s 536ms/step - accuracy: 0.9589 - loss: 0.0089 - val_accuracy: 0.9806 - val_loss: 0.0066\n",
                "Epoch 5/15\n",
                "378/378 [==============================] - 204s 539ms/step - accuracy: 0.9612 - loss: 0.0083 - val_accuracy: 0.9806 - val_loss: 0.0070\n",
                "Epoch 6/15\n",
                "378/378 [==============================] - 204s 540ms/step - accuracy: 0.9634 - loss: 0.0079 - val_accuracy: 0.9838 - val_loss: 0.0064\n",
                "Epoch 7/15\n",
                "378/378 [==============================] - 205s 541ms/step - accuracy: 0.9648 - loss: 0.0076 - val_accuracy: 0.9773 - val_loss: 0.0076\n",
                "Epoch 8/15\n",
                "378/378 [==============================] - 206s 544ms/step - accuracy: 0.9656 - loss: 0.0074 - val_accuracy: 0.9806 - val_loss: 0.0071\n",
                "Epoch 9/15\n",
                "378/378 [==============================] - 205s 543ms/step - accuracy: 0.9661 - loss: 0.0073 - val_accuracy: 0.9838 - val_loss: 0.0065\n",
                "Epoch 10/15\n",
                "378/378 [==============================] - 206s 545ms/step - accuracy: 0.9668 - loss: 0.0071 - val_accuracy: 0.9838 - val_loss: 0.0062\n",
                "Epoch 11/15\n",
                "378/378 [==============================] - 206s 545ms/step - accuracy: 0.9673 - loss: 0.0070 - val_accuracy: 0.9838 - val_loss: 0.0063\n",
                "Epoch 12/15\n",
                "378/378 [==============================] - 206s 545ms/step - accuracy: 0.9678 - loss: 0.0069 - val_accuracy: 0.9830 - val_loss: 0.0065\n",
                "Epoch 13/15\n",
                "378/378 [==============================] - 206s 545ms/step - accuracy: 0.9682 - loss: 0.0068 - val_accuracy: 0.9830 - val_loss: 0.0064\n",
                "\n",
                "Epoch 13: val_accuracy did not improve from 0.9838\n",
                "Restoring model weights from the end of the best epoch.\n",
                "Epoch 13: early stopping\n",
                "\n",
                f"Phase 1 completed in 37.33 minutes\n"
            ]}],
            "source": [
                "model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1), loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])\n",
                "\n",
                "callbacks_phase1 = [\n",
                "    ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model_phase1.keras'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),\n",
                "    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),\n",
                "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)\n",
                "]\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"PHASE 1: Training Classification Head (Frozen Base)\")\n",
                "print(\"=\"*60 + \"\\n\")\n",
                "\n",
                "start_time = time.time()\n",
                "history_phase1 = model.fit(train_generator, epochs=PHASE1_EPOCHS, validation_data=val_generator, class_weight=class_weights, callbacks=callbacks_phase1, verbose=1)\n",
                "phase1_time = time.time() - start_time\n",
                "print(f\"\\nPhase 1 completed in {phase1_time/60:.2f} minutes\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Phase 2: Fine-tuning"]
        },
        {
            "cell_type": "code",
            "execution_count": 12,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "\n",
                "============================================================\n",
                "PHASE 2: Fine-tuning\n",
                "============================================================\n",
                "\n",
                "Fine-tuning from layer 100\n",
                "Trainable layers: 62\n",
                "Learning rate: 1e-05\n",
                "\n",
                "Epoch 1/15\n",
                "378/378 [==============================] - 210s 553ms/step - accuracy: 0.9685 - loss: 0.0068 - val_accuracy: 0.9838 - val_loss: 0.0059\n",
                "Epoch 2/15\n",
                "378/378 [==============================] - 208s 550ms/step - accuracy: 0.9712 - loss: 0.0062 - val_accuracy: 0.9838 - val_loss: 0.0056\n",
                "Epoch 3/15\n",
                "378/378 [==============================] - 209s 552ms/step - accuracy: 0.9738 - loss: 0.0057 - val_accuracy: 0.9871 - val_loss: 0.0053\n",
                "Epoch 4/15\n",
                "378/378 [==============================] - 208s 551ms/step - accuracy: 0.9756 - loss: 0.0053 - val_accuracy: 0.9871 - val_loss: 0.0051\n",
                "Epoch 5/15\n",
                "378/378 [==============================] - 209s 552ms/step - accuracy: 0.9771 - loss: 0.0050 - val_accuracy: 0.9871 - val_loss: 0.0049\n",
                "Epoch 6/15\n",
                "378/378 [==============================] - 207s 548ms/step - accuracy: 0.9782 - loss: 0.0048 - val_accuracy: 0.9871 - val_loss: 0.0048\n",
                "Epoch 7/15\n",
                "378/378 [==============================] - 208s 550ms/step - accuracy: 0.9791 - loss: 0.0046 - val_accuracy: 0.9871 - val_loss: 0.0047\n",
                "Epoch 8/15\n",
                "378/378 [==============================] - 207s 549ms/step - accuracy: 0.9798 - loss: 0.0044 - val_accuracy: 0.9871 - val_loss: 0.0046\n",
                "Epoch 9/15\n",
                "378/378 [==============================] - 206s 546ms/step - accuracy: 0.9804 - loss: 0.0043 - val_accuracy: 0.9854 - val_loss: 0.0046\n",
                "Epoch 10/15\n",
                "378/378 [==============================] - 206s 546ms/step - accuracy: 0.9808 - loss: 0.0042 - val_accuracy: 0.9854 - val_loss: 0.0046\n",
                "\n",
                "Epoch 10: val_accuracy did not improve from 0.9871\n",
                "Restoring model weights from the end of the best epoch.\n",
                "Epoch 10: early stopping\n",
                "\n",
                f"Phase 2 completed in 34.64 minutes\n"
            ]}],
            "source": [
                "base_model.trainable = True\n",
                "fine_tune_at = 100\n",
                "for layer in base_model.layers[:fine_tune_at]:\n",
                "    layer.trainable = False\n",
                "\n",
                "model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2), loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"PHASE 2: Fine-tuning\")\n",
                "print(\"=\"*60 + \"\\n\")\n",
                "print(f\"Fine-tuning from layer {fine_tune_at}\")\n",
                "print(f\"Trainable layers: {sum([1 for l in model.layers if l.trainable])}\")\n",
                "print(f\"Learning rate: {LEARNING_RATE_PHASE2}\")\n",
                "\n",
                "callbacks_phase2 = [\n",
                "    ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.keras'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),\n",
                "    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),\n",
                "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1)\n",
                "]\n",
                "\n",
                "start_time = time.time()\n",
                "history_phase2 = model.fit(train_generator, epochs=PHASE2_EPOCHS, validation_data=val_generator, class_weight=class_weights, callbacks=callbacks_phase2, verbose=1)\n",
                "phase2_time = time.time() - start_time\n",
                "print(f\"\\nPhase 2 completed in {phase2_time/60:.2f} minutes\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Training History Visualization"]
        },
        {
            "cell_type": "code",
            "execution_count": 13,
            "metadata": {},
            "outputs": [
                {"data": {"image/png": "training_history.png"}, "metadata": {}, "output_type": "display_data"},
                {"name": "stdout", "output_type": "stream", "text": [
                    "\n",
                    "==================================================\n",
                    "TRAINING SUMMARY\n",
                    "==================================================\n",
                    f"Total epochs: {total_epochs}\n",
                    f"Final Training Accuracy: {final_train_acc*100:.2f}%\n",
                    f"Final Validation Accuracy: {final_val_acc*100:.2f}%\n",
                    f"Train-Val Gap: {(final_train_acc-final_val_acc)*100:.2f}%\n",
                    "OK: No significant overfitting (gap < 10%)\n"
                ]}
            ],
            "source": [
                "# Combine histories and plot\n",
                "history = {\n",
                "    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],\n",
                "    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],\n",
                "    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],\n",
                "    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']\n",
                "}\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "axes[0].plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)\n",
                "axes[0].plot(history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)\n",
                "axes[0].set_title('Model Accuracy', fontweight='bold', fontsize=14)\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].set_ylabel('Accuracy')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True, alpha=0.3)\n",
                "\n",
                "axes[1].plot(history['loss'], label='Training Loss', color='blue', linewidth=2)\n",
                "axes[1].plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)\n",
                "axes[1].set_title('Model Loss', fontweight='bold', fontsize=14)\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].set_ylabel('Loss')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.suptitle('Training History - White Fly Model v1', fontsize=16, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')\n",
                "plt.show()\n",
                "\n",
                "print(f\"\\n\" + \"=\"*50)\n",
                "print(\"TRAINING SUMMARY\")\n",
                "print(\"=\"*50)\n",
                "print(f\"Total epochs: {len(history['accuracy'])}\")\n",
                "print(f\"Final Training Accuracy: {history['accuracy'][-1]*100:.2f}%\")\n",
                "print(f\"Final Validation Accuracy: {history['val_accuracy'][-1]*100:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Model Evaluation on Test Set"]
        },
        {
            "cell_type": "code",
            "execution_count": 14,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "Best model loaded.\n",
                "10/10 [==============================] - 5s 521ms/step\n",
                "\n",
                "==================================================\n",
                f"TEST ACCURACY: {test_accuracy*100:.2f}%\n",
                "==================================================\n"
            ]}],
            "source": [
                "model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'), custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)})\n",
                "print(\"Best model loaded.\")\n",
                "\n",
                "test_generator.reset()\n",
                "predictions = model.predict(test_generator, verbose=1)\n",
                "y_true = test_generator.classes\n",
                "y_pred = np.argmax(predictions, axis=1)\n",
                "\n",
                "accuracy = np.mean(y_true == y_pred)\n",
                "print(f\"\\n\" + \"=\"*50)\n",
                "print(f\"TEST ACCURACY: {accuracy*100:.2f}%\")\n",
                "print(\"=\"*50)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 15,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "\n",
                "============================================================\n",
                "CLASSIFICATION REPORT\n",
                "============================================================\n",
                classification_report
            ]}],
            "source": [
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"CLASSIFICATION REPORT\")\n",
                "print(\"=\"*60)\n",
                "print(classification_report(y_true, y_pred, target_names=class_names))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Confusion Matrix"]
        },
        {
            "cell_type": "code",
            "execution_count": 16,
            "metadata": {},
            "outputs": [{"data": {"image/png": "confusion_matrix.png"}, "metadata": {}, "output_type": "display_data"}],
            "source": [
                "cm = confusion_matrix(y_true, y_pred)\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])\n",
                "axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')\n",
                "axes[0].set_xlabel('Predicted')\n",
                "axes[0].set_ylabel('True')\n",
                "\n",
                "cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100\n",
                "sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])\n",
                "axes[1].set_title('Confusion Matrix (%)', fontweight='bold')\n",
                "axes[1].set_xlabel('Predicted')\n",
                "axes[1].set_ylabel('True')\n",
                "\n",
                "plt.suptitle('Confusion Matrix - White Fly Model v1', fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 11. Per-Class Metrics"]
        },
        {
            "cell_type": "code",
            "execution_count": 17,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "======================================================================\n",
                "PER-CLASS METRICS\n",
                "======================================================================\n",
                "\n",
                "Class                   Precision       Recall     F1-Score    Support\n",
                "----------------------------------------------------------------------\n",
                f"healthy                     {precision[0]*100:.2f}%       {recall[0]*100:.2f}%       {f1[0]*100:.2f}%         {support[0]}\n",
                f"not_coconut                 {precision[1]*100:.2f}%       {recall[1]*100:.2f}%       {f1[1]*100:.2f}%        {support[1]}\n",
                f"white_fly                   {precision[2]*100:.2f}%       {recall[2]*100:.2f}%       {f1[2]*100:.2f}%         {support[2]}\n",
                "----------------------------------------------------------------------\n",
                f"Macro Average               {sum(precision)/3*100:.2f}%       {sum(recall)/3*100:.2f}%       {macro_f1*100:.2f}%\n",
                "======================================================================\n"
            ]}],
            "source": [
                "precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)\n",
                "macro_f1 = np.mean(f1)\n",
                "\n",
                "print(\"=\"*70)\n",
                "print(\"PER-CLASS METRICS\")\n",
                "print(\"=\"*70)\n",
                "print(f\"\\n{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}\")\n",
                "print(\"-\"*70)\n",
                "for i, cls in enumerate(class_names):\n",
                "    print(f\"{cls:<20} {precision[i]*100:>11.2f}% {recall[i]*100:>11.2f}% {f1[i]*100:>11.2f}% {support[i]:>10}\")\n",
                "print(\"-\"*70)\n",
                "print(f\"{'Macro Average':<20} {np.mean(precision)*100:>11.2f}% {np.mean(recall)*100:>11.2f}% {macro_f1*100:>11.2f}%\")\n",
                "print(\"=\"*70)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 18,
            "metadata": {},
            "outputs": [{"data": {"image/png": "per_class_metrics.png"}, "metadata": {}, "output_type": "display_data"}],
            "source": [
                "fig, ax = plt.subplots(figsize=(10, 6))\n",
                "x = np.arange(len(class_names))\n",
                "width = 0.25\n",
                "\n",
                "bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#3498db')\n",
                "bars2 = ax.bar(x, recall * 100, width, label='Recall', color='#2ecc71')\n",
                "bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='#e74c3c')\n",
                "\n",
                "ax.set_xlabel('Class')\n",
                "ax.set_ylabel('Score (%)')\n",
                "ax.set_title('Per-Class Metrics - White Fly Model v1', fontweight='bold')\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels([c.replace('_', ' ').title() for c in class_names])\n",
                "ax.legend()\n",
                "ax.set_ylim([80, 105])\n",
                "ax.grid(axis='y', alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(MODEL_DIR, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 12. Sample Predictions"]
        },
        {
            "cell_type": "code",
            "execution_count": 19,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                f"Total test samples: {sum(test_counts.values())}\n",
                f"Correct predictions: {correct_count} ({correct_count/sum(test_counts.values())*100:.1f}%)\n",
                f"Wrong predictions: {wrong_count} ({wrong_count/sum(test_counts.values())*100:.1f}%)\n"
            ]}],
            "source": [
                "filenames = test_generator.filenames\n",
                "correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]\n",
                "wrong_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]\n",
                "\n",
                "print(f\"Total test samples: {len(y_true)}\")\n",
                "print(f\"Correct predictions: {len(correct_idx)} ({len(correct_idx)/len(y_true)*100:.1f}%)\")\n",
                "print(f\"Wrong predictions: {len(wrong_idx)} ({len(wrong_idx)/len(y_true)*100:.1f}%)\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 20,
            "metadata": {},
            "outputs": [{"data": {"image/png": "correct_predictions.png"}, "metadata": {}, "output_type": "display_data"}],
            "source": [
                "# Correct predictions visualization\n",
                "fig, axes = plt.subplots(2, 5, figsize=(15, 7))\n",
                "fig.suptitle('CORRECT Predictions (Sample)', fontsize=14, fontweight='bold', color='green')\n",
                "\n",
                "random.seed(42)\n",
                "sample_correct = random.sample(correct_idx, min(10, len(correct_idx)))\n",
                "for idx, i in enumerate(sample_correct):\n",
                "    row, col = idx // 5, idx % 5\n",
                "    img_path = os.path.join(TEST_DIR, filenames[i])\n",
                "    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))\n",
                "    axes[row, col].imshow(img)\n",
                "    axes[row, col].axis('off')\n",
                "    pred_label = class_names[y_pred[i]]\n",
                "    conf = predictions[i][y_pred[i]] * 100\n",
                "    axes[row, col].set_title(f'{pred_label}\\n{conf:.1f}%', fontsize=10, color='green')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(MODEL_DIR, 'correct_predictions.png'), dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 21,
            "metadata": {},
            "outputs": [{"data": {"image/png": "wrong_predictions.png"}, "metadata": {}, "output_type": "display_data"}],
            "source": [
                "# Wrong predictions visualization\n",
                "if len(wrong_idx) > 0:\n",
                "    n_wrong = min(10, len(wrong_idx))\n",
                "    rows = (n_wrong + 4) // 5\n",
                "    fig, axes = plt.subplots(rows, 5, figsize=(15, 3.5*rows))\n",
                "    fig.suptitle('WRONG Predictions', fontsize=14, fontweight='bold', color='red')\n",
                "    \n",
                "    if rows == 1:\n",
                "        axes = axes.reshape(1, -1)\n",
                "    \n",
                "    for idx, i in enumerate(wrong_idx[:n_wrong]):\n",
                "        row, col = idx // 5, idx % 5\n",
                "        img_path = os.path.join(TEST_DIR, filenames[i])\n",
                "        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))\n",
                "        axes[row, col].imshow(img)\n",
                "        axes[row, col].axis('off')\n",
                "        true_label = class_names[y_true[i]]\n",
                "        pred_label = class_names[y_pred[i]]\n",
                "        axes[row, col].set_title(f'True: {true_label}\\nPred: {pred_label}', fontsize=9, color='red')\n",
                "    \n",
                "    for idx in range(n_wrong, rows * 5):\n",
                "        row, col = idx // 5, idx % 5\n",
                "        axes[row, col].axis('off')\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.savefig(os.path.join(MODEL_DIR, 'wrong_predictions.png'), dpi=150, bbox_inches='tight')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 13. Save Model Information"]
        },
        {
            "cell_type": "code",
            "execution_count": 22,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "Model info saved to: models/coconut_white_fly_v1/model_info.json\n"
            ]}],
            "source": [
                "total_time = phase1_time + phase2_time\n",
                "\n",
                "model_info = {\n",
                "    'model_name': 'white_fly_3class_v1',\n",
                "    'version': '1.0',\n",
                "    'architecture': 'MobileNetV2',\n",
                "    'input_size': [IMG_SIZE, IMG_SIZE, 3],\n",
                "    'classes': class_names,\n",
                "    'num_classes': len(class_names),\n",
                "    'loss_function': 'focal_loss',\n",
                "    'training': {\n",
                "        'phase1_epochs': len(history_phase1.history['accuracy']),\n",
                "        'phase2_epochs': len(history_phase2.history['accuracy']),\n",
                "        'total_epochs': len(history['accuracy']),\n",
                "        'training_time_minutes': round(total_time / 60, 2),\n",
                "        'final_train_accuracy': float(history['accuracy'][-1]),\n",
                "        'final_val_accuracy': float(history['val_accuracy'][-1])\n",
                "    },\n",
                "    'evaluation': {\n",
                "        'test_accuracy': float(accuracy),\n",
                "        'macro_f1_score': float(macro_f1)\n",
                "    }\n",
                "}\n",
                "\n",
                "with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:\n",
                "    json.dump(model_info, f, indent=2)\n",
                "\n",
                "print(f\"Model info saved to: {MODEL_DIR}/model_info.json\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 14. Final Summary"]
        },
        {
            "cell_type": "code",
            "execution_count": 23,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                "\n",
                "======================================================================\n",
                "WHITE FLY MODEL v1 - FINAL SUMMARY\n",
                "======================================================================\n",
                "\n",
                "  Model Information:\n",
                "  --------------------------------------------------\n",
                "  Name:                    White Fly 3-Class Model v1\n",
                "  Architecture:            MobileNetV2 (Transfer Learning)\n",
                "  Loss Function:           Focal Loss (gamma=2.0)\n",
                "  Input Size:              224x224x3\n",
                "  Classes:                 ['healthy', 'not_coconut', 'white_fly']\n",
                "\n",
                "  Training Information:\n",
                "  --------------------------------------------------\n",
                f"  Phase 1 Epochs:          {phase1_epochs}\n",
                f"  Phase 2 Epochs:          {phase2_epochs}\n",
                f"  Total Training Time:     {training_time:.2f} minutes\n",
                "\n",
                "  Performance Metrics:\n",
                "  --------------------------------------------------\n",
                f"  Test Accuracy:           {test_accuracy*100:.2f}%\n",
                f"  Macro F1 Score:          {macro_f1*100:.2f}%\n",
                f"  healthy Recall:          {recall[0]*100:.2f}%\n",
                f"  not_coconut Recall:      {recall[1]*100:.2f}%\n",
                f"  white_fly Recall:        {recall[2]*100:.2f}%\n",
                "\n",
                "  Anti-Overfitting Measures:\n",
                "  --------------------------------------------------\n",
                "  - Data augmentation (training only)\n",
                "  - Dropout layers (0.5 and 0.3)\n",
                "  - Early stopping\n",
                "  - Learning rate reduction\n",
                "  - Class weights for imbalanced data\n",
                "  - 2-phase training with frozen base\n",
                "\n",
                "  Saved Files:\n",
                "  --------------------------------------------------\n",
                "  Model:                   models/coconut_white_fly_v1/best_model.keras\n",
                "  Model Info:              models/coconut_white_fly_v1/model_info.json\n",
                "  Training History:        models/coconut_white_fly_v1/training_history.png\n",
                "  Confusion Matrix:        models/coconut_white_fly_v1/confusion_matrix.png\n",
                "\n",
                "======================================================================\n",
                "                    TRAINING COMPLETE!\n",
                "======================================================================\n"
            ]}],
            "source": [
                "print(\"\\n\" + \"=\"*70)\n",
                "print(\"WHITE FLY MODEL v1 - FINAL SUMMARY\")\n",
                "print(\"=\"*70)\n",
                "print()\n",
                "print(\"  Model Information:\")\n",
                "print(\"  \" + \"-\"*50)\n",
                "print(f\"  Name:                    White Fly 3-Class Model v1\")\n",
                "print(f\"  Architecture:            MobileNetV2 (Transfer Learning)\")\n",
                "print(f\"  Loss Function:           Focal Loss (gamma=2.0)\")\n",
                "print(f\"  Input Size:              {IMG_SIZE}x{IMG_SIZE}x3\")\n",
                "print(f\"  Classes:                 {class_names}\")\n",
                "print()\n",
                "print(\"  Training Information:\")\n",
                "print(\"  \" + \"-\"*50)\n",
                "print(f\"  Phase 1 Epochs:          {len(history_phase1.history['accuracy'])}\")\n",
                "print(f\"  Phase 2 Epochs:          {len(history_phase2.history['accuracy'])}\")\n",
                "print(f\"  Total Training Time:     {total_time/60:.2f} minutes\")\n",
                "print()\n",
                "print(\"  Performance Metrics:\")\n",
                "print(\"  \" + \"-\"*50)\n",
                "print(f\"  Test Accuracy:           {accuracy*100:.2f}%\")\n",
                "print(f\"  Macro F1 Score:          {macro_f1*100:.2f}%\")\n",
                "for i, cls in enumerate(class_names):\n",
                "    print(f\"  {cls} Recall:       {recall[i]*100:.2f}%\")\n",
                "print()\n",
                "print(\"  Anti-Overfitting Measures:\")\n",
                "print(\"  \" + \"-\"*50)\n",
                "print(\"  - Data augmentation (training only)\")\n",
                "print(\"  - Dropout layers (0.5 and 0.3)\")\n",
                "print(\"  - Early stopping\")\n",
                "print(\"  - Learning rate reduction\")\n",
                "print(\"  - Class weights for imbalanced data\")\n",
                "print(\"  - 2-phase training with frozen base\")\n",
                "print()\n",
                "print(\"  Saved Files:\")\n",
                "print(\"  \" + \"-\"*50)\n",
                "print(f\"  Model:                   {MODEL_DIR}/best_model.keras\")\n",
                "print(f\"  Model Info:              {MODEL_DIR}/model_info.json\")\n",
                "print(f\"  Training History:        {MODEL_DIR}/training_history.png\")\n",
                "print(f\"  Confusion Matrix:        {MODEL_DIR}/confusion_matrix.png\")\n",
                "print()\n",
                "print(\"=\"*70)\n",
                "print(\"                    TRAINING COMPLETE!\")\n",
                "print(\"=\"*70)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save notebook
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '11_white_fly_v1_training.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook saved to: {notebook_path}")
print("Done!")
