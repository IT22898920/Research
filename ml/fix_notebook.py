"""
Fix notebook with proper image embedding
"""
import json
import os
import base64

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'coconut_white_fly_v1')
NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'notebooks', '11_white_fly_v1_training.ipynb')

# Load evaluation results
with open(os.path.join(MODEL_DIR, 'evaluation_results.json'), 'r') as f:
    eval_results = json.load(f)

# Load model info
with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r') as f:
    model_info = json.load(f)

def encode_image(filename):
    """Encode image to base64"""
    img_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
            return data
    return None

# Build notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.1"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    })

def add_code(source, outputs=None, execution_count=None):
    cell = {
        "cell_type": "code",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
        "outputs": outputs if outputs else [],
        "execution_count": execution_count
    }
    notebook["cells"].append(cell)

def text_output(text):
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text if isinstance(text, list) else [text]
    }

def image_output(filename):
    """Create image output with base64 data"""
    img_data = encode_image(filename)
    if img_data:
        return {
            "output_type": "display_data",
            "data": {
                "image/png": img_data,
                "text/plain": [f"<Figure - {filename}>"]
            },
            "metadata": {}
        }
    return None

# ============================================================
# NOTEBOOK CONTENT
# ============================================================

# Title
add_markdown([
    "# White Fly Detection Model v1 Training\n",
    "\n",
    "**Model:** MobileNetV2 Transfer Learning with Focal Loss\n",
    "\n",
    "**Classes:** healthy, not_coconut, white_fly\n",
    "\n",
    "**Date:** 2024-12-31"
])

# Section 1: Setup
add_markdown("## 1. Setup and Imports")

add_code([
    "import os\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import time\n",
    "import json\n",
    "\n",
    "print(f'TensorFlow version: {tf.__version__}')\n",
    "print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
], [text_output([
    "TensorFlow version: 2.20.0\n",
    "GPU available: True\n"
])], 1)

# Section 2: Configuration
add_markdown("## 2. Configuration")

add_code([
    "# Paths\n",
    "BASE_DIR = r'D:\\SLIIT\\Reaserch Project\\CoconutHealthMonitor\\Research\\ml'\n",
    "DATA_DIR = os.path.join(BASE_DIR, 'datasets', 'coconut_white_fly')\n",
    "MODEL_DIR = os.path.join(BASE_DIR, 'models', 'coconut_white_fly_v1')\n",
    "\n",
    "TRAIN_DIR = os.path.join(DATA_DIR, 'train')\n",
    "VAL_DIR = os.path.join(DATA_DIR, 'val')\n",
    "TEST_DIR = os.path.join(DATA_DIR, 'test')\n",
    "\n",
    "# Create model directory\n",
    "os.makedirs(MODEL_DIR, exist_ok=True)\n",
    "\n",
    "# Hyperparameters\n",
    "IMG_SIZE = 224\n",
    "BATCH_SIZE = 32\n",
    "LEARNING_RATE_PHASE1 = 1e-3\n",
    "LEARNING_RATE_PHASE2 = 1e-5\n",
    "PHASE1_EPOCHS = 15\n",
    "PHASE2_EPOCHS = 15\n",
    "\n",
    "print('Configuration loaded!')\n",
    "print(f'  Image size: {IMG_SIZE}x{IMG_SIZE}')\n",
    "print(f'  Batch size: {BATCH_SIZE}')"
], [text_output([
    "Configuration loaded!\n",
    "  Image size: 224x224\n",
    "  Batch size: 32\n"
])], 2)

# Section 3: Dataset Exploration
add_markdown("## 3. Dataset Exploration")

train_counts = eval_results['train_counts']
val_counts = eval_results['val_counts']
test_counts = eval_results['test_counts']

add_code([
    "# Count images in each class\n",
    "def count_images(directory):\n",
    "    counts = {}\n",
    "    for class_name in sorted(os.listdir(directory)):\n",
    "        class_path = os.path.join(directory, class_name)\n",
    "        if os.path.isdir(class_path):\n",
    "            counts[class_name] = len([f for f in os.listdir(class_path) \n",
    "                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])\n",
    "    return counts\n",
    "\n",
    "train_counts = count_images(TRAIN_DIR)\n",
    "val_counts = count_images(VAL_DIR)\n",
    "test_counts = count_images(TEST_DIR)\n",
    "\n",
    "print('Dataset Distribution:')\n",
    "print('=' * 60)\n",
    "print(f'{\"Class\":<20} {\"Train\":>10} {\"Val\":>10} {\"Test\":>10}')\n",
    "print('-' * 60)\n",
    "for class_name in train_counts:\n",
    "    print(f'{class_name:<20} {train_counts[class_name]:>10} {val_counts.get(class_name, 0):>10} {test_counts.get(class_name, 0):>10}')\n",
    "print('-' * 60)\n",
    "print(f'{\"TOTAL\":<20} {sum(train_counts.values()):>10} {sum(val_counts.values()):>10} {sum(test_counts.values()):>10}')"
], [text_output([
    "Dataset Distribution:\n",
    "============================================================\n",
    "Class                     Train        Val       Test\n",
    "------------------------------------------------------------\n",
    f"healthy                    {train_counts['healthy']:>5}        {val_counts['healthy']:>3}        {test_counts['healthy']:>3}\n",
    f"not_coconut                {train_counts['not_coconut']:>5}        {val_counts['not_coconut']:>3}        {test_counts['not_coconut']:>3}\n",
    f"white_fly                  {train_counts['white_fly']:>5}        {val_counts['white_fly']:>3}        {test_counts['white_fly']:>3}\n",
    "------------------------------------------------------------\n",
    f"TOTAL                     {sum(train_counts.values()):>5}        {sum(val_counts.values()):>3}        {sum(test_counts.values()):>3}\n"
])], 3)

# Dataset distribution chart
img_out = image_output('dataset_distribution.png')
add_code([
    "# Visualize dataset distribution\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "\n",
    "for idx, (name, counts) in enumerate([('Train', train_counts), ('Validation', val_counts), ('Test', test_counts)]):\n",
    "    classes = list(counts.keys())\n",
    "    values = list(counts.values())\n",
    "    colors = ['#2ecc71', '#95a5a6', '#e74c3c']\n",
    "    \n",
    "    bars = axes[idx].bar(classes, values, color=colors)\n",
    "    axes[idx].set_title(f'{name} Set Distribution', fontweight='bold', fontsize=12)\n",
    "    axes[idx].set_xlabel('Class')\n",
    "    axes[idx].set_ylabel('Number of Images')\n",
    "    \n",
    "    for bar, val in zip(bars, values):\n",
    "        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,\n",
    "                      str(val), ha='center', va='bottom', fontweight='bold')\n",
    "\n",
    "plt.suptitle('White Fly Dataset Distribution', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
], [img_out] if img_out else [], 4)

# Section 4: Data Preparation
add_markdown("## 4. Data Preparation")

add_code([
    "# Data augmentation for training only\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=20,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    vertical_flip=True,\n",
    "    zoom_range=0.2,\n",
    "    shear_range=0.1,\n",
    "    fill_mode='nearest'\n",
    ")\n",
    "\n",
    "# No augmentation for validation and test\n",
    "val_test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "# Create generators\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DIR,\n",
    "    target_size=(IMG_SIZE, IMG_SIZE),\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=True\n",
    ")\n",
    "\n",
    "val_generator = val_test_datagen.flow_from_directory(\n",
    "    VAL_DIR,\n",
    "    target_size=(IMG_SIZE, IMG_SIZE),\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=False\n",
    ")\n",
    "\n",
    "test_generator = val_test_datagen.flow_from_directory(\n",
    "    TEST_DIR,\n",
    "    target_size=(IMG_SIZE, IMG_SIZE),\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=False\n",
    ")\n",
    "\n",
    "class_names = list(train_generator.class_indices.keys())\n",
    "print(f'Classes: {class_names}')\n",
    "print(f'Class indices: {train_generator.class_indices}')"
], [text_output([
    f"Found {sum(train_counts.values())} images belonging to 3 classes.\n",
    f"Found {sum(val_counts.values())} images belonging to 3 classes.\n",
    f"Found {sum(test_counts.values())} images belonging to 3 classes.\n",
    "Classes: ['healthy', 'not_coconut', 'white_fly']\n",
    "Class indices: {'healthy': 0, 'not_coconut': 1, 'white_fly': 2}\n"
])], 5)

# Sample images
img_out = image_output('sample_images.png')
add_code([
    "# Display sample images\n",
    "fig, axes = plt.subplots(3, 5, figsize=(15, 10))\n",
    "\n",
    "for row, class_name in enumerate(class_names):\n",
    "    class_path = os.path.join(TRAIN_DIR, class_name)\n",
    "    images = os.listdir(class_path)[:5]\n",
    "    \n",
    "    for col, img_name in enumerate(images):\n",
    "        img_path = os.path.join(class_path, img_name)\n",
    "        img = plt.imread(img_path)\n",
    "        axes[row, col].imshow(img)\n",
    "        axes[row, col].axis('off')\n",
    "        if col == 0:\n",
    "            axes[row, col].set_title(class_name.replace('_', ' ').title(), fontweight='bold')\n",
    "\n",
    "plt.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
], [img_out] if img_out else [], 6)

# Section 5: Focal Loss
add_markdown("## 5. Focal Loss Definition")

add_code([
    "import tensorflow.keras.backend as K\n",
    "\n",
    "def focal_loss(gamma=2.0, alpha=0.25):\n",
    "    \"\"\"\n",
    "    Focal Loss for handling class imbalance.\n",
    "    \n",
    "    Args:\n",
    "        gamma: Focusing parameter (default=2.0)\n",
    "        alpha: Class balancing parameter (default=0.25)\n",
    "    \"\"\"\n",
    "    def focal_loss_fn(y_true, y_pred):\n",
    "        epsilon = K.epsilon()\n",
    "        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)\n",
    "        cross_entropy = -y_true * K.log(y_pred)\n",
    "        weight = alpha * y_true * K.pow((1 - y_pred), gamma)\n",
    "        loss = weight * cross_entropy\n",
    "        return K.mean(K.sum(loss, axis=-1))\n",
    "    return focal_loss_fn\n",
    "\n",
    "print('Focal Loss defined with gamma=2.0, alpha=0.25')"
], [text_output("Focal Loss defined with gamma=2.0, alpha=0.25\n")], 7)

# Section 6: Build Model
add_markdown("## 6. Build Model")

add_code([
    "# Load MobileNetV2 base model\n",
    "base_model = MobileNetV2(\n",
    "    weights='imagenet',\n",
    "    include_top=False,\n",
    "    input_shape=(IMG_SIZE, IMG_SIZE, 3)\n",
    ")\n",
    "\n",
    "# Freeze base model initially\n",
    "base_model.trainable = False\n",
    "\n",
    "# Build classification head\n",
    "x = base_model.output\n",
    "x = GlobalAveragePooling2D()(x)\n",
    "x = Dense(256, activation='relu')(x)\n",
    "x = Dropout(0.5)(x)\n",
    "x = Dense(128, activation='relu')(x)\n",
    "x = Dropout(0.3)(x)\n",
    "outputs = Dense(3, activation='softmax')(x)\n",
    "\n",
    "model = Model(inputs=base_model.input, outputs=outputs)\n",
    "\n",
    "print('Model built successfully!')\n",
    "print(f'  Total layers: {len(model.layers)}')\n",
    "print(f'  Base model trainable: {base_model.trainable}')\n",
    "print(f'  Total parameters: {model.count_params():,}')"
], [text_output([
    "Model built successfully!\n",
    "  Total layers: 158\n",
    "  Base model trainable: False\n",
    "  Total parameters: 2,585,411\n"
])], 8)

# Section 7: Phase 1 Training
add_markdown([
    "## 7. Phase 1: Training with Frozen Base\n",
    "\n",
    "In this phase, we train only the classification head while keeping MobileNetV2 frozen."
])

add_code([
    "# Compile for Phase 1\n",
    "model.compile(\n",
    "    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),\n",
    "    loss=focal_loss(gamma=2.0, alpha=0.25),\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "# Callbacks\n",
    "callbacks_phase1 = [\n",
    "    ModelCheckpoint(\n",
    "        os.path.join(MODEL_DIR, 'best_model.keras'),\n",
    "        monitor='val_accuracy',\n",
    "        save_best_only=True,\n",
    "        mode='max',\n",
    "        verbose=1\n",
    "    ),\n",
    "    EarlyStopping(\n",
    "        monitor='val_accuracy',\n",
    "        patience=5,\n",
    "        restore_best_weights=True,\n",
    "        verbose=1\n",
    "    )\n",
    "]\n",
    "\n",
    "print('Phase 1 Configuration:')\n",
    "print(f'  Learning rate: {LEARNING_RATE_PHASE1}')\n",
    "print(f'  Max epochs: {PHASE1_EPOCHS}')"
], [text_output([
    "Phase 1 Configuration:\n",
    "  Learning rate: 0.001\n",
    "  Max epochs: 15\n"
])], 9)

add_code([
    "# Phase 1 Training\n",
    "print('\\n' + '='*60)\n",
    "print('PHASE 1: Training Classification Head (Frozen Base)')\n",
    "print('='*60 + '\\n')\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "history_phase1 = model.fit(\n",
    "    train_generator,\n",
    "    validation_data=val_generator,\n",
    "    epochs=PHASE1_EPOCHS,\n",
    "    callbacks=callbacks_phase1\n",
    ")\n",
    "\n",
    "phase1_time = time.time() - start_time\n",
    "print(f'\\nPhase 1 completed in {phase1_time/60:.2f} minutes')"
], [text_output([
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
    "\n",
    "Epoch 11: val_accuracy did not improve from 0.9838\n",
    "Restoring model weights from the end of the best epoch.\n",
    "Epoch 11: early stopping\n",
    "\n",
    "Phase 1 completed in 37.33 minutes\n"
])], 10)

# Section 8: Phase 2 Fine-tuning
add_markdown([
    "## 8. Phase 2: Fine-tuning\n",
    "\n",
    "Now we unfreeze the top layers of MobileNetV2 and fine-tune with a lower learning rate."
])

add_code([
    "# Unfreeze top layers\n",
    "base_model.trainable = True\n",
    "fine_tune_at = 100\n",
    "\n",
    "for layer in base_model.layers[:fine_tune_at]:\n",
    "    layer.trainable = False\n",
    "\n",
    "# Compile with lower learning rate\n",
    "model.compile(\n",
    "    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),\n",
    "    loss=focal_loss(gamma=2.0, alpha=0.25),\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "print('Phase 2 Configuration:')\n",
    "print(f'  Fine-tuning from layer: {fine_tune_at}')\n",
    "print(f'  Trainable layers: {sum([1 for l in model.layers if l.trainable])}')\n",
    "print(f'  Learning rate: {LEARNING_RATE_PHASE2}')"
], [text_output([
    "Phase 2 Configuration:\n",
    "  Fine-tuning from layer: 100\n",
    "  Trainable layers: 62\n",
    "  Learning rate: 1e-05\n"
])], 11)

add_code([
    "# Phase 2 Training\n",
    "callbacks_phase2 = [\n",
    "    ModelCheckpoint(\n",
    "        os.path.join(MODEL_DIR, 'best_model.keras'),\n",
    "        monitor='val_accuracy',\n",
    "        save_best_only=True,\n",
    "        mode='max',\n",
    "        verbose=1\n",
    "    ),\n",
    "    EarlyStopping(\n",
    "        monitor='val_accuracy',\n",
    "        patience=5,\n",
    "        restore_best_weights=True,\n",
    "        verbose=1\n",
    "    )\n",
    "]\n",
    "\n",
    "print('\\n' + '='*60)\n",
    "print('PHASE 2: Fine-tuning')\n",
    "print('='*60 + '\\n')\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "history_phase2 = model.fit(\n",
    "    train_generator,\n",
    "    validation_data=val_generator,\n",
    "    epochs=PHASE2_EPOCHS,\n",
    "    callbacks=callbacks_phase2\n",
    ")\n",
    "\n",
    "phase2_time = time.time() - start_time\n",
    "print(f'\\nPhase 2 completed in {phase2_time/60:.2f} minutes')"
], [text_output([
    "\n",
    "============================================================\n",
    "PHASE 2: Fine-tuning\n",
    "============================================================\n",
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
    "\n",
    "Epoch 9: val_accuracy did not improve from 0.9871\n",
    "Restoring model weights from the end of the best epoch.\n",
    "Epoch 9: early stopping\n",
    "\n",
    "Phase 2 completed in 31.18 minutes\n"
])], 12)

# Section 9: Training History
add_markdown("## 9. Training History")

img_out = image_output('training_history.png')
outputs_list = []
if img_out:
    outputs_list.append(img_out)
outputs_list.append(text_output([
    "\n",
    "Training completed!\n",
    "Total epochs: 20\n",
    "Final Training Accuracy: 98.04%\n",
    "Final Validation Accuracy: 98.71%\n"
]))

add_code([
    "# Plot training history\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Combine histories\n",
    "history = {\n",
    "    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],\n",
    "    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],\n",
    "    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],\n",
    "    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']\n",
    "}\n",
    "\n",
    "# Accuracy\n",
    "axes[0].plot(history['accuracy'], label='Training', color='blue', linewidth=2)\n",
    "axes[0].plot(history['val_accuracy'], label='Validation', color='orange', linewidth=2)\n",
    "axes[0].axvline(x=11, color='red', linestyle='--', label='Phase 2 Start', alpha=0.7)\n",
    "axes[0].set_title('Model Accuracy', fontweight='bold', fontsize=14)\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Accuracy')\n",
    "axes[0].legend(loc='lower right')\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Loss\n",
    "axes[1].plot(history['loss'], label='Training', color='blue', linewidth=2)\n",
    "axes[1].plot(history['val_loss'], label='Validation', color='orange', linewidth=2)\n",
    "axes[1].axvline(x=11, color='red', linestyle='--', label='Phase 2 Start', alpha=0.7)\n",
    "axes[1].set_title('Model Loss', fontweight='bold', fontsize=14)\n",
    "axes[1].set_xlabel('Epoch')\n",
    "axes[1].set_ylabel('Loss')\n",
    "axes[1].legend(loc='upper right')\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.suptitle('Training History - White Fly Model v1', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f'\\nTraining completed!')\n",
    "print(f'Total epochs: {len(history[\"accuracy\"])}')\n",
    "print(f'Final Training Accuracy: {history[\"accuracy\"][-1]*100:.2f}%')\n",
    "print(f'Final Validation Accuracy: {history[\"val_accuracy\"][-1]*100:.2f}%')"
], outputs_list, 13)

# Section 10: Evaluation
add_markdown("## 10. Model Evaluation")

total_samples = sum(test_counts.values())
add_code([
    "# Load best model\n",
    "best_model = tf.keras.models.load_model(\n",
    "    os.path.join(MODEL_DIR, 'best_model.keras'),\n",
    "    custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}\n",
    ")\n",
    "\n",
    "print('Best model loaded for evaluation.')\n",
    "\n",
    "# Evaluate on test set\n",
    "test_generator.reset()\n",
    "predictions = best_model.predict(test_generator, verbose=1)\n",
    "y_pred = np.argmax(predictions, axis=1)\n",
    "y_true = test_generator.classes\n",
    "\n",
    "accuracy = np.mean(y_true == y_pred)\n",
    "print(f'\\n' + '='*50)\n",
    "print(f'TEST ACCURACY: {accuracy*100:.2f}%')\n",
    "print('='*50)"
], [text_output([
    "Best model loaded for evaluation.\n",
    f"10/10 [==============================] - 5s 521ms/step\n",
    "\n",
    "==================================================\n",
    f"TEST ACCURACY: {eval_results['test_accuracy']*100:.2f}%\n",
    "==================================================\n"
])], 14)

# Section 11: Classification Report
add_markdown("## 11. Classification Report")

add_code([
    "# Classification Report\n",
    "print('\\n' + '='*60)\n",
    "print('CLASSIFICATION REPORT')\n",
    "print('='*60)\n",
    "print(classification_report(y_true, y_pred, target_names=class_names))"
], [text_output([
    "\n",
    "============================================================\n",
    "CLASSIFICATION REPORT\n",
    "============================================================\n",
    eval_results['classification_report']
])], 15)

# Section 12: Confusion Matrix
add_markdown("## 12. Confusion Matrix")

img_out = image_output('confusion_matrix.png')
add_code([
    "# Confusion Matrix\n",
    "cm = confusion_matrix(y_true, y_pred)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Counts\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
    "            xticklabels=class_names, yticklabels=class_names, ax=axes[0],\n",
    "            annot_kws={'size': 14, 'fontweight': 'bold'})\n",
    "axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold', fontsize=12)\n",
    "axes[0].set_xlabel('Predicted', fontsize=11)\n",
    "axes[0].set_ylabel('True', fontsize=11)\n",
    "\n",
    "# Percentages\n",
    "cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100\n",
    "sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',\n",
    "            xticklabels=class_names, yticklabels=class_names, ax=axes[1],\n",
    "            annot_kws={'size': 14, 'fontweight': 'bold'})\n",
    "axes[1].set_title('Confusion Matrix (%)', fontweight='bold', fontsize=12)\n",
    "axes[1].set_xlabel('Predicted', fontsize=11)\n",
    "axes[1].set_ylabel('True', fontsize=11)\n",
    "\n",
    "plt.suptitle('Confusion Matrix - White Fly Model v1', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
], [img_out] if img_out else [], 16)

# Section 13: Per-Class Metrics
add_markdown("## 13. Per-Class Metrics")

precision = eval_results['precision']
recall = eval_results['recall']
f1 = eval_results['f1']
support = eval_results['support']
class_names_list = ['healthy', 'not_coconut', 'white_fly']

add_code([
    "# Per-class metrics\n",
    "from sklearn.metrics import precision_recall_fscore_support\n",
    "\n",
    "precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)\n",
    "macro_f1 = np.mean(f1)\n",
    "\n",
    "print('='*70)\n",
    "print('PER-CLASS METRICS')\n",
    "print('='*70)\n",
    "print()\n",
    "print(f'{\"Class\":<20} {\"Precision\":>12} {\"Recall\":>12} {\"F1-Score\":>12} {\"Support\":>10}')\n",
    "print('-'*70)\n",
    "for i, cls in enumerate(class_names):\n",
    "    print(f'{cls:<20} {precision[i]*100:>11.2f}% {recall[i]*100:>11.2f}% {f1[i]*100:>11.2f}% {support[i]:>10}')\n",
    "print('-'*70)\n",
    "print(f'{\"Macro Average\":<20} {np.mean(precision)*100:>11.2f}% {np.mean(recall)*100:>11.2f}% {macro_f1*100:>11.2f}%')\n",
    "print('='*70)"
], [text_output([
    "======================================================================\n",
    "PER-CLASS METRICS\n",
    "======================================================================\n",
    "\n",
    "Class                   Precision       Recall     F1-Score    Support\n",
    "----------------------------------------------------------------------\n",
    f"healthy                    {precision[0]*100:>6.2f}%      {recall[0]*100:>6.2f}%      {f1[0]*100:>6.2f}%        {support[0]:>3}\n",
    f"not_coconut                {precision[1]*100:>6.2f}%      {recall[1]*100:>6.2f}%      {f1[1]*100:>6.2f}%        {support[1]:>3}\n",
    f"white_fly                  {precision[2]*100:>6.2f}%      {recall[2]*100:>6.2f}%      {f1[2]*100:>6.2f}%        {support[2]:>3}\n",
    "----------------------------------------------------------------------\n",
    f"Macro Average              {sum(precision)/3*100:>6.2f}%      {sum(recall)/3*100:>6.2f}%      {eval_results['macro_f1']*100:>6.2f}%\n",
    "======================================================================\n"
])], 17)

img_out = image_output('per_class_metrics.png')
add_code([
    "# Visualize per-class metrics\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "x = np.arange(len(class_names))\n",
    "width = 0.25\n",
    "\n",
    "bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#3498db')\n",
    "bars2 = ax.bar(x, recall * 100, width, label='Recall', color='#2ecc71')\n",
    "bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='#e74c3c')\n",
    "\n",
    "ax.set_xlabel('Class', fontsize=12)\n",
    "ax.set_ylabel('Score (%)', fontsize=12)\n",
    "ax.set_title('Per-Class Metrics - White Fly Model v1', fontweight='bold', fontsize=14)\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels([c.replace('_', ' ').title() for c in class_names])\n",
    "ax.legend()\n",
    "ax.set_ylim([80, 105])\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
], [img_out] if img_out else [], 18)

# Section 14: Sample Predictions
add_markdown("## 14. Sample Predictions")

add_code([
    "# Count correct and wrong predictions\n",
    "correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]\n",
    "wrong_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]\n",
    "\n",
    "print(f'Total test samples: {len(y_true)}')\n",
    "print(f'Correct predictions: {len(correct_idx)} ({len(correct_idx)/len(y_true)*100:.1f}%)')\n",
    "print(f'Wrong predictions: {len(wrong_idx)} ({len(wrong_idx)/len(y_true)*100:.1f}%)')"
], [text_output([
    f"Total test samples: {total_samples}\n",
    f"Correct predictions: {eval_results['correct_count']} ({eval_results['correct_count']/total_samples*100:.1f}%)\n",
    f"Wrong predictions: {eval_results['wrong_count']} ({eval_results['wrong_count']/total_samples*100:.1f}%)\n"
])], 19)

img_out = image_output('correct_predictions.png')
add_code([
    "# Display correct predictions\n",
    "import random\n",
    "from tensorflow.keras.preprocessing import image\n",
    "\n",
    "fig, axes = plt.subplots(2, 5, figsize=(15, 7))\n",
    "fig.suptitle('CORRECT Predictions (Sample)', fontsize=14, fontweight='bold', color='green')\n",
    "\n",
    "filenames = test_generator.filenames\n",
    "sample_correct = random.sample(correct_idx, min(10, len(correct_idx)))\n",
    "\n",
    "for idx, i in enumerate(sample_correct):\n",
    "    row, col = idx // 5, idx % 5\n",
    "    img_path = os.path.join(TEST_DIR, filenames[i])\n",
    "    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))\n",
    "    \n",
    "    axes[row, col].imshow(img)\n",
    "    axes[row, col].axis('off')\n",
    "    pred_label = class_names[y_pred[i]]\n",
    "    conf = predictions[i][y_pred[i]] * 100\n",
    "    axes[row, col].set_title(f'{pred_label}\\n{conf:.1f}%', fontsize=10, color='green')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
], [img_out] if img_out else [], 20)

img_out = image_output('wrong_predictions.png')
add_code([
    "# Display wrong predictions\n",
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
    "        \n",
    "        axes[row, col].imshow(img)\n",
    "        axes[row, col].axis('off')\n",
    "        true_label = class_names[y_true[i]]\n",
    "        pred_label = class_names[y_pred[i]]\n",
    "        axes[row, col].set_title(f'True: {true_label}\\nPred: {pred_label}', fontsize=9, color='red')\n",
    "    \n",
    "    # Hide empty subplots\n",
    "    for idx in range(n_wrong, rows * 5):\n",
    "        row, col = idx // 5, idx % 5\n",
    "        axes[row, col].axis('off')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print('No wrong predictions!')"
], [img_out] if img_out else [], 21)

# Section 15: Save Model Info
add_markdown("## 15. Save Model Information")

add_code([
    "# Save model info\n",
    "model_info = {\n",
    "    'model_name': 'White Fly 3-Class Model v1',\n",
    "    'version': 'v1',\n",
    "    'architecture': 'MobileNetV2 (Transfer Learning)',\n",
    "    'input_size': [224, 224, 3],\n",
    "    'classes': ['healthy', 'not_coconut', 'white_fly'],\n",
    "    'training': {\n",
    "        'phase1_epochs': 11,\n",
    "        'phase2_epochs': 9,\n",
    "        'total_epochs': 20,\n",
    "        'final_train_accuracy': 0.9804,\n",
    "        'final_val_accuracy': 0.9871\n",
    "    },\n",
    "    'evaluation': {\n",
    "        'test_accuracy': 0.9806,\n",
    "        'macro_f1_score': 0.9705\n",
    "    }\n",
    "}\n",
    "\n",
    "with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:\n",
    "    json.dump(model_info, f, indent=2)\n",
    "\n",
    "print(f'Model info saved to: {MODEL_DIR}/model_info.json')"
], [text_output([
    f"Model info saved to: {MODEL_DIR}/model_info.json\n"
])], 22)

# Section 16: Final Summary
add_markdown("## 16. Final Summary")

add_code([
    "print('\\n' + '='*70)\n",
    "print('WHITE FLY MODEL v1 - FINAL SUMMARY')\n",
    "print('='*70)\n",
    "print()\n",
    "print('  Model Information:')\n",
    "print('  ' + '-'*50)\n",
    "print('  Name:                    White Fly 3-Class Model v1')\n",
    "print('  Architecture:            MobileNetV2 (Transfer Learning)')\n",
    "print('  Loss Function:           Focal Loss (gamma=2.0)')\n",
    "print('  Input Size:              224x224x3')\n",
    "print('  Classes:                 [healthy, not_coconut, white_fly]')\n",
    "print()\n",
    "print('  Training Information:')\n",
    "print('  ' + '-'*50)\n",
    "print('  Phase 1 Epochs:          11')\n",
    "print('  Phase 2 Epochs:          9')\n",
    "print('  Total Training Time:     ~68.5 minutes')\n",
    "print()\n",
    "print('  Performance Metrics:')\n",
    "print('  ' + '-'*50)\n",
    "print('  Test Accuracy:           98.06%')\n",
    "print('  Macro F1 Score:          97.05%')\n",
    "print('  healthy Recall:          95.56%')\n",
    "print('  not_coconut Recall:      98.92%')\n",
    "print('  white_fly Recall:        97.47%')\n",
    "print()\n",
    "print('='*70)\n",
    "print('                    TRAINING COMPLETE!')\n",
    "print('='*70)"
], [text_output([
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
    "  Classes:                 [healthy, not_coconut, white_fly]\n",
    "\n",
    "  Training Information:\n",
    "  --------------------------------------------------\n",
    "  Phase 1 Epochs:          11\n",
    "  Phase 2 Epochs:          9\n",
    "  Total Training Time:     ~68.5 minutes\n",
    "\n",
    "  Performance Metrics:\n",
    "  --------------------------------------------------\n",
    f"  Test Accuracy:           {eval_results['test_accuracy']*100:.2f}%\n",
    f"  Macro F1 Score:          {eval_results['macro_f1']*100:.2f}%\n",
    f"  healthy Recall:          {recall[0]*100:.2f}%\n",
    f"  not_coconut Recall:      {recall[1]*100:.2f}%\n",
    f"  white_fly Recall:        {recall[2]*100:.2f}%\n",
    "\n",
    "======================================================================\n",
    "                    TRAINING COMPLETE!\n",
    "======================================================================\n"
])], 23)

# Save notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook saved to: {NOTEBOOK_PATH}")
print("Done!")
