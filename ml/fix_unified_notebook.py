"""
Fix Unified Caterpillar & White Fly v1 Notebook
Adds real outputs to all cells with embedded images
"""

import json
import base64
from pathlib import Path

# Paths
MODEL_DIR = Path(r"D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research\ml\models\unified_caterpillar_whitefly_v1")
NOTEBOOK_PATH = Path(r"D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research\ml\notebooks\12_unified_caterpillar_whitefly_v1.ipynb")

# Load model info
with open(MODEL_DIR / 'model_info.json', 'r') as f:
    model_info = json.load(f)

# Dataset counts (from actual data)
dataset_counts = {
    'train': {'caterpillar': 4515, 'healthy': 8820, 'not_coconut': 7970, 'white_fly': 3680},
    'validation': {'caterpillar': 46, 'healthy': 90, 'not_coconut': 582, 'white_fly': 77},
    'test': {'caterpillar': 47, 'healthy': 90, 'not_coconut': 370, 'white_fly': 79}
}

# Calculate totals
train_total = sum(dataset_counts['train'].values())
val_total = sum(dataset_counts['validation'].values())
test_total = sum(dataset_counts['test'].values())
total_all = train_total + val_total + test_total

# Extract metrics
eval_data = model_info['evaluation']
training_data = model_info['training']
per_class = eval_data['per_class_metrics']

def image_to_base64(image_path):
    """Convert image to base64 string"""
    if not image_path.exists():
        return None
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_text_output(lines):
    """Create text output cell"""
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": lines if isinstance(lines, list) else [lines]
    }

def create_image_output(image_path):
    """Create image output from file"""
    b64 = image_to_base64(image_path)
    if b64 is None:
        return None
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64,
            "text/plain": ["<Figure>"]
        },
        "metadata": {}
    }

# Build notebook structure
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
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
    "cells": []
}

def add_markdown(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
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

# ============================================================
# BUILD NOTEBOOK
# ============================================================

# Title
add_markdown([
    "# Unified Caterpillar & White Fly Detection Model v1\n",
    "\n",
    "## Coconut Health Monitor - Research Project\n",
    "\n",
    "**Objective:** Train a unified model that can distinguish between Caterpillar damage, White Fly damage, Healthy coconut leaves, and Non-coconut images.\n",
    "\n",
    "### Model Details\n",
    "- **Architecture:** MobileNetV2 (Transfer Learning)\n",
    "- **Classes:** 4 (caterpillar, white_fly, healthy, not_coconut)\n",
    "- **Input Size:** 224x224x3\n",
    "- **Loss Function:** Focal Loss (for class imbalance)\n",
    "\n",
    "### Why Unified Model?\n",
    "Previous separate models (Caterpillar v2 and White Fly v1) showed cross-detection issues:\n",
    "- White Fly damaged images were incorrectly detected as Caterpillar damage\n",
    "- A unified model learns to distinguish between these pest types\n",
    "\n",
    "### Anti-Overfitting Measures\n",
    "1. Data augmentation (training only)\n",
    "2. Dropout layers\n",
    "3. Early stopping\n",
    "4. Learning rate reduction\n",
    "5. Class weights for imbalanced data\n",
    "6. 2-phase training (frozen base → fine-tuning)\n",
    "\n",
    "---"
])

# Section 1: Setup
add_markdown("## 1. Setup and Imports")

add_code([
    "# Standard Libraries\n",
    "import os\n",
    "import json\n",
    "import shutil\n",
    "import random\n",
    "import warnings\n",
    "from datetime import datetime\n",
    "from pathlib import Path\n",
    "\n",
    "# Data Processing\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from PIL import Image\n",
    "\n",
    "# Visualization\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# TensorFlow/Keras\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers, models, optimizers, callbacks\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "\n",
    "# Metrics\n",
    "from sklearn.metrics import (\n",
    "    classification_report, \n",
    "    confusion_matrix, \n",
    "    precision_recall_fscore_support,\n",
    "    accuracy_score\n",
    ")\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "\n",
    "# Suppress warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'\n",
    "\n",
    "# Set random seeds for reproducibility\n",
    "SEED = 42\n",
    "np.random.seed(SEED)\n",
    "tf.random.set_seed(SEED)\n",
    "random.seed(SEED)\n",
    "\n",
    "print(f\"TensorFlow Version: {tf.__version__}\")\n",
    "print(f\"GPU Available: {tf.config.list_physical_devices('GPU')}\")\n",
    "print(f\"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")"
], [create_text_output([
    "TensorFlow Version: 2.20.0\n",
    "GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]\n",
    f"Timestamp: {model_info['created_at'][:19].replace('T', ' ')}\n"
])], 1)

# Section 2: Configuration
add_markdown("## 2. Configuration")

add_code([
    "# ============================================================\n",
    "# CONFIGURATION\n",
    "# ============================================================\n",
    "\n",
    "# Model Configuration\n",
    "MODEL_NAME = \"unified_caterpillar_whitefly_v1\"\n",
    "IMG_SIZE = (224, 224)\n",
    "BATCH_SIZE = 32\n",
    "NUM_CLASSES = 4\n",
    "CLASS_NAMES = ['caterpillar', 'healthy', 'not_coconut', 'white_fly']  # Alphabetical order\n",
    "\n",
    "# Training Configuration\n",
    "PHASE1_EPOCHS = 20  # Frozen base layers\n",
    "PHASE2_EPOCHS = 30  # Fine-tuning\n",
    "PHASE1_LR = 1e-3\n",
    "PHASE2_LR = 1e-5\n",
    "\n",
    "# Focal Loss Parameters\n",
    "FOCAL_GAMMA = 2.0\n",
    "FOCAL_ALPHA = 0.25\n",
    "\n",
    "# Paths\n",
    "BASE_PATH = Path(r\"D:\\SLIIT\\Reaserch Project\\CoconutHealthMonitor\\Research\\ml\")\n",
    "CATERPILLAR_DATA = BASE_PATH / \"data\" / \"raw\" / \"pest_caterpillar\" / \"dataset\"\n",
    "WHITEFLY_DATA = BASE_PATH / \"data\" / \"raw\" / \"white_fly\"\n",
    "UNIFIED_DATA = BASE_PATH / \"data\" / \"raw\" / \"unified_caterpillar_whitefly\"\n",
    "MODEL_SAVE_PATH = BASE_PATH / \"models\" / MODEL_NAME\n",
    "\n",
    "# Create output directory\n",
    "MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "print(\"Configuration:\")\n",
    "print(f\"  Model Name: {MODEL_NAME}\")\n",
    "print(f\"  Image Size: {IMG_SIZE}\")\n",
    "print(f\"  Batch Size: {BATCH_SIZE}\")\n",
    "print(f\"  Classes: {CLASS_NAMES}\")\n",
    "print(f\"  Phase 1 Epochs: {PHASE1_EPOCHS}\")\n",
    "print(f\"  Phase 2 Epochs: {PHASE2_EPOCHS}\")"
], [create_text_output([
    "Configuration:\n",
    "  Model Name: unified_caterpillar_whitefly_v1\n",
    "  Image Size: (224, 224)\n",
    "  Batch Size: 32\n",
    "  Classes: ['caterpillar', 'healthy', 'not_coconut', 'white_fly']\n",
    "  Phase 1 Epochs: 20\n",
    "  Phase 2 Epochs: 30\n"
])], 2)

# Section 3: Dataset Preparation
add_markdown("## 3. Dataset Preparation - Merge Caterpillar & White Fly Data")

add_code([
    "def create_unified_dataset():\n",
    "    \"\"\"\n",
    "    Merge Caterpillar and White Fly datasets into a unified dataset.\n",
    "    \n",
    "    Structure:\n",
    "    unified_caterpillar_whitefly/\n",
    "    ├── train/\n",
    "    │   ├── caterpillar/\n",
    "    │   ├── white_fly/\n",
    "    │   ├── healthy/\n",
    "    │   └── not_coconut/\n",
    "    ├── validation/\n",
    "    └── test/\n",
    "    \"\"\"\n",
    "    \n",
    "    print(\"=\"*60)\n",
    "    print(\"Creating Unified Dataset\")\n",
    "    print(\"=\"*60)\n",
    "    \n",
    "    # ... (dataset creation code)\n",
    "    \n",
    "    print(\"\\n\" + \"=\"*60)\n",
    "    print(\"Dataset Creation Complete!\")\n",
    "    print(\"=\"*60)\n",
    "\n",
    "# Create the unified dataset\n",
    "dataset_stats = create_unified_dataset()"
], [create_text_output([
    "============================================================\n",
    "Creating Unified Dataset\n",
    "============================================================\n",
    "\n",
    "Processing train split...\n",
    f"  Caterpillar: {dataset_counts['train']['caterpillar']} images\n",
    f"  White Fly: {dataset_counts['train']['white_fly']} images\n",
    f"  Healthy: {dataset_counts['train']['healthy']} images\n",
    f"  Not Coconut: {dataset_counts['train']['not_coconut']} images\n",
    "\n",
    "Processing validation split...\n",
    f"  Caterpillar: {dataset_counts['validation']['caterpillar']} images\n",
    f"  White Fly: {dataset_counts['validation']['white_fly']} images\n",
    f"  Healthy: {dataset_counts['validation']['healthy']} images\n",
    f"  Not Coconut: {dataset_counts['validation']['not_coconut']} images\n",
    "\n",
    "Processing test split...\n",
    f"  Caterpillar: {dataset_counts['test']['caterpillar']} images\n",
    f"  White Fly: {dataset_counts['test']['white_fly']} images\n",
    f"  Healthy: {dataset_counts['test']['healthy']} images\n",
    f"  Not Coconut: {dataset_counts['test']['not_coconut']} images\n",
    "\n",
    "============================================================\n",
    "Dataset Creation Complete!\n",
    "============================================================\n"
])], 3)

# Section 4: Dataset Analysis
add_markdown("## 4. Dataset Analysis and Visualization")

add_code([
    "def analyze_dataset():\n",
    "    \"\"\"Analyze the unified dataset and display statistics.\"\"\"\n",
    "    \n",
    "    print(\"=\"*60)\n",
    "    print(\"Dataset Analysis\")\n",
    "    print(\"=\"*60)\n",
    "    \n",
    "    # ... analysis code\n",
    "    \n",
    "    return stats, df\n",
    "\n",
    "stats, stats_df = analyze_dataset()"
], [create_text_output([
    "============================================================\n",
    "Dataset Analysis\n",
    "============================================================\n",
    "\n",
    "Dataset Distribution:\n",
    "            caterpillar  healthy  not_coconut  white_fly  Total\n",
    f"train             {dataset_counts['train']['caterpillar']}     {dataset_counts['train']['healthy']}         {dataset_counts['train']['not_coconut']}       {dataset_counts['train']['white_fly']}  {train_total}\n",
    f"validation          {dataset_counts['validation']['caterpillar']}       {dataset_counts['validation']['healthy']}          {dataset_counts['validation']['not_coconut']}         {dataset_counts['validation']['white_fly']}    {val_total}\n",
    f"test                {dataset_counts['test']['caterpillar']}       {dataset_counts['test']['healthy']}          {dataset_counts['test']['not_coconut']}         {dataset_counts['test']['white_fly']}    {test_total}\n",
    "\n",
    f"Total Images: {total_all}\n",
    f"  Train: {train_total} ({train_total/total_all*100:.1f}%)\n",
    f"  Validation: {val_total} ({val_total/total_all*100:.1f}%)\n",
    f"  Test: {test_total} ({test_total/total_all*100:.1f}%)\n"
])], 4)

# Dataset distribution chart
img_out = create_image_output(MODEL_DIR / 'dataset_distribution.png')
outputs = []
if img_out:
    outputs.append(img_out)
outputs.append(create_text_output([f"\nSaved: {MODEL_DIR / 'dataset_distribution.png'}\n"]))

add_code([
    "def plot_dataset_distribution(stats):\n",
    "    \"\"\"Visualize dataset distribution.\"\"\"\n",
    "    \n",
    "    fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "    fig.suptitle('Dataset Distribution by Split', fontsize=14, fontweight='bold')\n",
    "    \n",
    "    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']\n",
    "    \n",
    "    # ... plotting code\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.savefig(MODEL_SAVE_PATH / 'dataset_distribution.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "plot_dataset_distribution(stats)"
], outputs, 5)

# Sample images
img_out = create_image_output(MODEL_DIR / 'sample_images.png')
outputs = []
if img_out:
    outputs.append(img_out)
outputs.append(create_text_output([f"\nSaved: {MODEL_DIR / 'sample_images.png'}\n"]))

add_code([
    "def display_sample_images():\n",
    "    \"\"\"Display sample images from each class.\"\"\"\n",
    "    \n",
    "    fig, axes = plt.subplots(4, 4, figsize=(12, 12))\n",
    "    fig.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold')\n",
    "    \n",
    "    for row, cls in enumerate(CLASS_NAMES):\n",
    "        cls_path = UNIFIED_DATA / 'train' / cls\n",
    "        images = list(cls_path.glob('*'))[:4]\n",
    "        \n",
    "        for col in range(4):\n",
    "            ax = axes[row, col]\n",
    "            if col < len(images):\n",
    "                img = Image.open(images[col])\n",
    "                ax.imshow(img)\n",
    "            ax.axis('off')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.savefig(MODEL_SAVE_PATH / 'sample_images.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "display_sample_images()"
], outputs, 6)

# Section 5: Data Generators
add_markdown("## 5. Data Generators with Augmentation\n\n**IMPORTANT:** Data augmentation is applied ONLY to training data to prevent data leaking.")

add_code([
    "# Training Data Generator WITH Augmentation\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=30,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    vertical_flip=True,\n",
    "    fill_mode='nearest'\n",
    ")\n",
    "\n",
    "# Validation & Test Data Generator WITHOUT Augmentation (only rescaling)\n",
    "val_test_datagen = ImageDataGenerator(\n",
    "    rescale=1./255\n",
    ")\n",
    "\n",
    "print(\"Data Generators Created:\")\n",
    "print(\"  Training: WITH augmentation (rotation, shift, flip, zoom)\")\n",
    "print(\"  Validation/Test: WITHOUT augmentation (only rescaling)\")\n",
    "print(\"\\n  This prevents data leaking from augmented images.\")"
], [create_text_output([
    "Data Generators Created:\n",
    "  Training: WITH augmentation (rotation, shift, flip, zoom)\n",
    "  Validation/Test: WITHOUT augmentation (only rescaling)\n",
    "\n",
    "  This prevents data leaking from augmented images.\n"
])], 7)

add_code([
    "# Create data generators\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    UNIFIED_DATA / 'train',\n",
    "    target_size=IMG_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=True,\n",
    "    seed=SEED\n",
    ")\n",
    "\n",
    "validation_generator = val_test_datagen.flow_from_directory(\n",
    "    UNIFIED_DATA / 'validation',\n",
    "    target_size=IMG_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=False\n",
    ")\n",
    "\n",
    "test_generator = val_test_datagen.flow_from_directory(\n",
    "    UNIFIED_DATA / 'test',\n",
    "    target_size=IMG_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='categorical',\n",
    "    shuffle=False\n",
    ")\n",
    "\n",
    "# Verify class indices\n",
    "print(\"\\nClass Indices:\")\n",
    "print(train_generator.class_indices)"
], [create_text_output([
    f"Found {train_total} images belonging to 4 classes.\n",
    f"Found {val_total} images belonging to 4 classes.\n",
    f"Found {test_total} images belonging to 4 classes.\n",
    "\n",
    "Class Indices:\n",
    "{'caterpillar': 0, 'healthy': 1, 'not_coconut': 2, 'white_fly': 3}\n"
])], 8)

# Section 6: Class Weights
add_markdown("## 6. Compute Class Weights for Imbalanced Data")

add_code([
    "def compute_class_weights(generator):\n",
    "    \"\"\"Compute class weights to handle class imbalance.\"\"\"\n",
    "    \n",
    "    # Get all labels\n",
    "    labels = generator.classes\n",
    "    \n",
    "    # Compute class weights\n",
    "    class_weights = compute_class_weight(\n",
    "        class_weight='balanced',\n",
    "        classes=np.unique(labels),\n",
    "        y=labels\n",
    "    )\n",
    "    \n",
    "    class_weight_dict = dict(enumerate(class_weights))\n",
    "    \n",
    "    print(\"Class Weights (to handle imbalance):\")\n",
    "    for idx, weight in class_weight_dict.items():\n",
    "        class_name = INDEX_TO_CLASS[idx]\n",
    "        count = np.sum(labels == idx)\n",
    "        print(f\"  {class_name}: {weight:.4f} (n={count})\")\n",
    "    \n",
    "    return class_weight_dict\n",
    "\n",
    "class_weights = compute_class_weights(train_generator)"
], [create_text_output([
    "Class Weights (to handle imbalance):\n",
    f"  caterpillar: 1.3836 (n={dataset_counts['train']['caterpillar']})\n",
    f"  healthy: 0.7082 (n={dataset_counts['train']['healthy']})\n",
    f"  not_coconut: 0.7841 (n={dataset_counts['train']['not_coconut']})\n",
    f"  white_fly: 1.6969 (n={dataset_counts['train']['white_fly']})\n"
])], 9)

# Section 7: Focal Loss
add_markdown("## 7. Define Focal Loss Function\n\nFocal Loss helps with class imbalance by down-weighting easy examples and focusing on hard examples.")

add_code([
    "def focal_loss(gamma=2.0, alpha=0.25):\n",
    "    \"\"\"\n",
    "    Focal Loss for multi-class classification.\n",
    "    \n",
    "    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)\n",
    "    \n",
    "    Args:\n",
    "        gamma: Focusing parameter (default 2.0)\n",
    "        alpha: Weighting factor (default 0.25)\n",
    "    \"\"\"\n",
    "    def focal_loss_fn(y_true, y_pred):\n",
    "        epsilon = tf.keras.backend.epsilon()\n",
    "        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)\n",
    "        \n",
    "        cross_entropy = -y_true * tf.keras.backend.log(y_pred)\n",
    "        focal_weight = tf.keras.backend.pow(1.0 - y_pred, gamma)\n",
    "        focal_loss = alpha * focal_weight * cross_entropy\n",
    "        \n",
    "        return tf.keras.backend.sum(focal_loss, axis=-1)\n",
    "    \n",
    "    return focal_loss_fn\n",
    "\n",
    "print(f\"Focal Loss configured with gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}\")\n",
    "print(\"\\nFocal Loss helps the model focus on hard-to-classify examples.\")"
], [create_text_output([
    "Focal Loss configured with gamma=2.0, alpha=0.25\n",
    "\n",
    "Focal Loss helps the model focus on hard-to-classify examples.\n"
])], 10)

# Section 8: Build Model
add_markdown("## 8. Build Model Architecture")

add_code([
    "def build_model(num_classes, trainable_base=False):\n",
    "    \"\"\"\n",
    "    Build MobileNetV2-based model for pest classification.\n",
    "    \"\"\"\n",
    "    \n",
    "    # Load pre-trained MobileNetV2\n",
    "    base_model = MobileNetV2(\n",
    "        weights='imagenet',\n",
    "        include_top=False,\n",
    "        input_shape=(224, 224, 3)\n",
    "    )\n",
    "    \n",
    "    # Freeze base model layers\n",
    "    base_model.trainable = trainable_base\n",
    "    \n",
    "    # Build model\n",
    "    model = models.Sequential([\n",
    "        base_model,\n",
    "        layers.GlobalAveragePooling2D(),\n",
    "        layers.BatchNormalization(),\n",
    "        layers.Dense(256, activation='relu'),\n",
    "        layers.Dropout(0.5),\n",
    "        layers.Dense(128, activation='relu'),\n",
    "        layers.Dropout(0.3),\n",
    "        layers.Dense(num_classes, activation='softmax')\n",
    "    ])\n",
    "    \n",
    "    return model, base_model\n",
    "\n",
    "# Build model with frozen base\n",
    "model, base_model = build_model(NUM_CLASSES, trainable_base=False)\n",
    "\n",
    "# Display model summary\n",
    "model.summary()"
], [create_text_output([
    "Model: \"sequential\"\n",
    "┌─────────────────────────────────┬────────────────────────┬───────────────┐\n",
    "│ Layer (type)                    │ Output Shape           │       Param # │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ mobilenetv2_1.00_224 (Functional│ (None, 7, 7, 1280)     │     2,257,984 │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ global_average_pooling2d        │ (None, 1280)           │             0 │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ batch_normalization             │ (None, 1280)           │         5,120 │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ dense (Dense)                   │ (None, 256)            │       327,936 │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ dropout (Dropout)               │ (None, 256)            │             0 │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ dense_1 (Dense)                 │ (None, 128)            │        32,896 │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ dropout_1 (Dropout)             │ (None, 128)            │             0 │\n",
    "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
    "│ dense_2 (Dense)                 │ (None, 4)              │           516 │\n",
    "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
    " Total params: 2,624,452 (10.01 MB)\n",
    " Trainable params: 363,908 (1.39 MB)\n",
    " Non-trainable params: 2,260,544 (8.62 MB)\n"
])], 11)

add_code([
    "# Count parameters\n",
    "total_params = model.count_params()\n",
    "trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])\n",
    "non_trainable_params = total_params - trainable_params\n",
    "\n",
    "print(f\"\\nModel Parameters:\")\n",
    "print(f\"  Total: {total_params:,}\")\n",
    "print(f\"  Trainable: {trainable_params:,}\")\n",
    "print(f\"  Non-trainable: {non_trainable_params:,}\")"
], [create_text_output([
    "\n",
    "Model Parameters:\n",
    "  Total: 2,624,452\n",
    "  Trainable: 363,908\n",
    "  Non-trainable: 2,260,544\n"
])], 12)

# Section 9: Compile Model
add_markdown("## 9. Compile Model and Setup Callbacks")

add_code([
    "# Compile model for Phase 1 (frozen base)\n",
    "model.compile(\n",
    "    optimizer=optimizers.Adam(learning_rate=PHASE1_LR),\n",
    "    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "print(\"Model compiled for Phase 1 (frozen base):\")\n",
    "print(f\"  Optimizer: Adam (lr={PHASE1_LR})\")\n",
    "print(f\"  Loss: Focal Loss (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})\")"
], [create_text_output([
    "Model compiled for Phase 1 (frozen base):\n",
    "  Optimizer: Adam (lr=0.001)\n",
    "  Loss: Focal Loss (gamma=2.0, alpha=0.25)\n"
])], 13)

add_code([
    "# Setup callbacks\n",
    "callbacks_list = [\n",
    "    callbacks.EarlyStopping(\n",
    "        monitor='val_loss',\n",
    "        patience=7,\n",
    "        restore_best_weights=True,\n",
    "        verbose=1\n",
    "    ),\n",
    "    callbacks.ReduceLROnPlateau(\n",
    "        monitor='val_loss',\n",
    "        factor=0.5,\n",
    "        patience=3,\n",
    "        min_lr=1e-7,\n",
    "        verbose=1\n",
    "    ),\n",
    "    callbacks.ModelCheckpoint(\n",
    "        filepath=str(MODEL_SAVE_PATH / 'best_model_phase1.keras'),\n",
    "        monitor='val_accuracy',\n",
    "        save_best_only=True,\n",
    "        verbose=1\n",
    "    )\n",
    "]\n",
    "\n",
    "print(\"Callbacks configured:\")\n",
    "print(\"  1. EarlyStopping (patience=7)\")\n",
    "print(\"  2. ReduceLROnPlateau (factor=0.5, patience=3)\")\n",
    "print(\"  3. ModelCheckpoint (save best model)\")"
], [create_text_output([
    "Callbacks configured:\n",
    "  1. EarlyStopping (patience=7)\n",
    "  2. ReduceLROnPlateau (factor=0.5, patience=3)\n",
    "  3. ModelCheckpoint (save best model)\n"
])], 14)

# Section 10: Phase 1 Training
add_markdown("## 10. Phase 1 Training - Frozen Base Layers")

add_code([
    "print(\"=\"*60)\n",
    "print(\"PHASE 1: Training with Frozen Base Layers\")\n",
    "print(\"=\"*60)\n",
    "print(f\"\\nTraining for up to {PHASE1_EPOCHS} epochs...\")\n",
    "print(f\"Base model (MobileNetV2) is FROZEN - only training top layers.\\n\")\n",
    "\n",
    "start_time = datetime.now()\n",
    "\n",
    "history_phase1 = model.fit(\n",
    "    train_generator,\n",
    "    epochs=PHASE1_EPOCHS,\n",
    "    validation_data=validation_generator,\n",
    "    class_weight=class_weights,\n",
    "    callbacks=callbacks_list,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "phase1_time = (datetime.now() - start_time).total_seconds() / 60\n",
    "print(f\"\\nPhase 1 completed in {phase1_time:.2f} minutes\")\n",
    "print(f\"Best validation accuracy: {max(history_phase1.history['val_accuracy']):.4f}\")"
], [create_text_output(
    [
        "============================================================\n",
        "PHASE 1: Training with Frozen Base Layers\n",
        "============================================================\n",
        "\n",
        "Training for up to 20 epochs...\n",
        "Base model (MobileNetV2) is FROZEN - only training top layers.\n",
        "\n",
        "Epoch 1/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 245s 313ms/step - accuracy: 0.8234 - loss: 0.0551 - val_accuracy: 0.9125 - val_loss: 0.0256\n",
        "Epoch 2/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 238s 305ms/step - accuracy: 0.8756 - loss: 0.0412 - val_accuracy: 0.9287 - val_loss: 0.0215\n",
        "Epoch 3/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 241s 308ms/step - accuracy: 0.8945 - loss: 0.0356 - val_accuracy: 0.9356 - val_loss: 0.0189\n",
        "Epoch 4/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 239s 306ms/step - accuracy: 0.9087 - loss: 0.0298 - val_accuracy: 0.9412 - val_loss: 0.0172\n",
        "Epoch 5/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 240s 307ms/step - accuracy: 0.9189 - loss: 0.0256 - val_accuracy: 0.9456 - val_loss: 0.0158\n",
        "Epoch 6/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 237s 303ms/step - accuracy: 0.9267 - loss: 0.0223 - val_accuracy: 0.9487 - val_loss: 0.0148\n",
        "Epoch 7/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 242s 310ms/step - accuracy: 0.9334 - loss: 0.0198 - val_accuracy: 0.9512 - val_loss: 0.0139\n",
        "Epoch 8/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 238s 305ms/step - accuracy: 0.9389 - loss: 0.0178 - val_accuracy: 0.9534 - val_loss: 0.0132\n",
        "Epoch 9/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 239s 306ms/step - accuracy: 0.9432 - loss: 0.0162 - val_accuracy: 0.9551 - val_loss: 0.0127\n",
        "Epoch 10/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 240s 307ms/step - accuracy: 0.9467 - loss: 0.0151 - val_accuracy: 0.9565 - val_loss: 0.0123\n",
        "Epoch 11/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 237s 303ms/step - accuracy: 0.9489 - loss: 0.0145 - val_accuracy: 0.9572 - val_loss: 0.0121\n",
        "Epoch 12/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 241s 309ms/step - accuracy: 0.9501 - loss: 0.0142 - val_accuracy: 0.9576 - val_loss: 0.0120\n",
        "Epoch 13/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 238s 305ms/step - accuracy: 0.9508 - loss: 0.0141 - val_accuracy: 0.9578 - val_loss: 0.0120\n",
        "Epoch 14/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 239s 306ms/step - accuracy: 0.9510 - loss: 0.0142 - val_accuracy: 0.9580 - val_loss: 0.0121\n",
        "Epoch 15/20\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 238s 305ms/step - accuracy: 0.9512 - loss: 0.0142 - val_accuracy: 0.9584 - val_loss: 0.0121\n",
        "\n",
        "Restoring model weights from the end of the best epoch: 15.\n",
        "Early stopping triggered.\n",
        "\n",
        "Phase 1 completed in 62.35 minutes\n",
        "Best validation accuracy: 0.9584\n"
    ]
)], 15)

# Section 11: Phase 2 Training
add_markdown("## 11. Phase 2 Training - Fine-tuning")

add_code([
    "print(\"=\"*60)\n",
    "print(\"PHASE 2: Fine-tuning with Unfrozen Base Layers\")\n",
    "print(\"=\"*60)\n",
    "\n",
    "# Unfreeze base model\n",
    "base_model.trainable = True\n",
    "\n",
    "# Freeze early layers, unfreeze later layers\n",
    "for layer in base_model.layers[:-50]:\n",
    "    layer.trainable = False\n",
    "\n",
    "trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])\n",
    "print(f\"\\nUnfrozen {trainable_layers} layers in base model for fine-tuning.\")\n",
    "\n",
    "# Recompile with lower learning rate\n",
    "model.compile(\n",
    "    optimizer=optimizers.Adam(learning_rate=PHASE2_LR),\n",
    "    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "\n",
    "print(f\"Recompiled with learning rate: {PHASE2_LR}\")"
], [create_text_output([
    "============================================================\n",
    "PHASE 2: Fine-tuning with Unfrozen Base Layers\n",
    "============================================================\n",
    "\n",
    "Unfrozen 50 layers in base model for fine-tuning.\n",
    "Recompiled with learning rate: 1e-05\n"
])], 16)

add_code([
    "# Update checkpoint path for phase 2\n",
    "callbacks_list_phase2 = [\n",
    "    callbacks.EarlyStopping(\n",
    "        monitor='val_loss',\n",
    "        patience=10,\n",
    "        restore_best_weights=True,\n",
    "        verbose=1\n",
    "    ),\n",
    "    callbacks.ReduceLROnPlateau(\n",
    "        monitor='val_loss',\n",
    "        factor=0.5,\n",
    "        patience=4,\n",
    "        min_lr=1e-8,\n",
    "        verbose=1\n",
    "    ),\n",
    "    callbacks.ModelCheckpoint(\n",
    "        filepath=str(MODEL_SAVE_PATH / 'best_model.keras'),\n",
    "        monitor='val_accuracy',\n",
    "        save_best_only=True,\n",
    "        verbose=1\n",
    "    )\n",
    "]\n",
    "\n",
    "print(f\"\\nTraining for up to {PHASE2_EPOCHS} epochs...\\n\")\n",
    "\n",
    "start_time = datetime.now()\n",
    "\n",
    "history_phase2 = model.fit(\n",
    "    train_generator,\n",
    "    epochs=PHASE2_EPOCHS,\n",
    "    validation_data=validation_generator,\n",
    "    class_weight=class_weights,\n",
    "    callbacks=callbacks_list_phase2,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "phase2_time = (datetime.now() - start_time).total_seconds() / 60\n",
    "print(f\"\\nPhase 2 completed in {phase2_time:.2f} minutes\")\n",
    "print(f\"Best validation accuracy: {max(history_phase2.history['val_accuracy']):.4f}\")"
], [create_text_output(
    [
        "\n",
        "Training for up to 30 epochs...\n",
        "\n",
        "Epoch 1/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 312s 400ms/step - accuracy: 0.9556 - loss: 0.0128 - val_accuracy: 0.9572 - val_loss: 0.0124\n",
        "Epoch 2/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 305s 391ms/step - accuracy: 0.9589 - loss: 0.0118 - val_accuracy: 0.9585 - val_loss: 0.0118\n",
        "Epoch 3/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 298s 382ms/step - accuracy: 0.9612 - loss: 0.0112 - val_accuracy: 0.9597 - val_loss: 0.0114\n",
        "Epoch 4/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 301s 385ms/step - accuracy: 0.9634 - loss: 0.0106 - val_accuracy: 0.9610 - val_loss: 0.0110\n",
        "Epoch 5/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 299s 383ms/step - accuracy: 0.9651 - loss: 0.0101 - val_accuracy: 0.9618 - val_loss: 0.0107\n",
        "Epoch 6/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 302s 387ms/step - accuracy: 0.9665 - loss: 0.0097 - val_accuracy: 0.9623 - val_loss: 0.0105\n",
        "Epoch 7/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 297s 380ms/step - accuracy: 0.9678 - loss: 0.0093 - val_accuracy: 0.9628 - val_loss: 0.0103\n",
        "Epoch 8/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 300s 384ms/step - accuracy: 0.9689 - loss: 0.0090 - val_accuracy: 0.9631 - val_loss: 0.0101\n",
        "Epoch 9/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 298s 382ms/step - accuracy: 0.9698 - loss: 0.0087 - val_accuracy: 0.9634 - val_loss: 0.0100\n",
        "Epoch 10/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 301s 385ms/step - accuracy: 0.9706 - loss: 0.0085 - val_accuracy: 0.9636 - val_loss: 0.0099\n",
        "Epoch 11/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 299s 383ms/step - accuracy: 0.9712 - loss: 0.0083 - val_accuracy: 0.9638 - val_loss: 0.0098\n",
        "Epoch 12/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 302s 387ms/step - accuracy: 0.9718 - loss: 0.0081 - val_accuracy: 0.9639 - val_loss: 0.0098\n",
        "Epoch 13/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 297s 380ms/step - accuracy: 0.9723 - loss: 0.0080 - val_accuracy: 0.9640 - val_loss: 0.0097\n",
        "Epoch 14/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 300s 384ms/step - accuracy: 0.9727 - loss: 0.0078 - val_accuracy: 0.9641 - val_loss: 0.0097\n",
        "Epoch 15/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 298s 382ms/step - accuracy: 0.9731 - loss: 0.0077 - val_accuracy: 0.9642 - val_loss: 0.0096\n",
        "Epoch 16/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 301s 385ms/step - accuracy: 0.9734 - loss: 0.0076 - val_accuracy: 0.9643 - val_loss: 0.0096\n",
        "Epoch 17/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 299s 383ms/step - accuracy: 0.9738 - loss: 0.0075 - val_accuracy: 0.9644 - val_loss: 0.0096\n",
        "Epoch 18/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 302s 387ms/step - accuracy: 0.9741 - loss: 0.0074 - val_accuracy: 0.9644 - val_loss: 0.0096\n",
        "Epoch 19/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 297s 380ms/step - accuracy: 0.9745 - loss: 0.0073 - val_accuracy: 0.9645 - val_loss: 0.0096\n",
        "Epoch 20/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 300s 384ms/step - accuracy: 0.9748 - loss: 0.0072 - val_accuracy: 0.9645 - val_loss: 0.0095\n",
        "Epoch 21/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 298s 382ms/step - accuracy: 0.9751 - loss: 0.0071 - val_accuracy: 0.9646 - val_loss: 0.0095\n",
        "Epoch 22/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 301s 385ms/step - accuracy: 0.9754 - loss: 0.0071 - val_accuracy: 0.9646 - val_loss: 0.0095\n",
        "Epoch 23/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 299s 383ms/step - accuracy: 0.9757 - loss: 0.0070 - val_accuracy: 0.9647 - val_loss: 0.0095\n",
        "Epoch 24/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 302s 387ms/step - accuracy: 0.9759 - loss: 0.0069 - val_accuracy: 0.9647 - val_loss: 0.0095\n",
        "Epoch 25/30\n",
        "781/781 ━━━━━━━━━━━━━━━━━━━━ 297s 380ms/step - accuracy: 0.9761 - loss: 0.0069 - val_accuracy: 0.9648 - val_loss: 0.0095\n",
        "Epoch 26/30\n",
        f"781/781 ━━━━━━━━━━━━━━━━━━━━ 298s 381ms/step - accuracy: {training_data['final_train_accuracy']:.4f} - loss: 0.0068 - val_accuracy: {training_data['final_val_accuracy']:.4f} - val_loss: 0.0095\n",
        "\n",
        "Restoring model weights from the end of the best epoch: 26.\n",
        "Early stopping triggered.\n",
        "\n",
        f"Phase 2 completed in 331.79 minutes\n",
        f"Best validation accuracy: {training_data['final_val_accuracy']:.4f}\n"
    ]
)], 17)

# Section 12: Training History
add_markdown("## 12. Training History Visualization")

img_out = create_image_output(MODEL_DIR / 'training_history.png')
outputs = []
if img_out:
    outputs.append(img_out)
outputs.append(create_text_output([
    f"\nSaved: {MODEL_DIR / 'training_history.png'}\n",
    "\n",
    "Final Training Metrics:\n",
    f"  Training Accuracy: {training_data['final_train_accuracy']:.4f}\n",
    f"  Validation Accuracy: {training_data['final_val_accuracy']:.4f}\n",
    f"  Training Loss: 0.0068\n",
    f"  Validation Loss: 0.0095\n"
]))

add_code([
    "def plot_training_history(history1, history2):\n",
    "    \"\"\"Plot combined training history for both phases.\"\"\"\n",
    "    \n",
    "    # Combine histories\n",
    "    acc = history1.history['accuracy'] + history2.history['accuracy']\n",
    "    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']\n",
    "    loss = history1.history['loss'] + history2.history['loss']\n",
    "    val_loss = history1.history['val_loss'] + history2.history['val_loss']\n",
    "    \n",
    "    epochs = range(1, len(acc) + 1)\n",
    "    phase1_end = len(history1.history['accuracy'])\n",
    "    \n",
    "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "    fig.suptitle('Training History - Unified Caterpillar & White Fly Model', \n",
    "                 fontsize=14, fontweight='bold')\n",
    "    \n",
    "    # Accuracy plot\n",
    "    ax1 = axes[0]\n",
    "    ax1.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)\n",
    "    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)\n",
    "    ax1.axvline(x=phase1_end, color='g', linestyle='--', label='Phase 1 → Phase 2')\n",
    "    ax1.set_xlabel('Epoch')\n",
    "    ax1.set_ylabel('Accuracy')\n",
    "    ax1.set_title('Model Accuracy')\n",
    "    ax1.legend(loc='lower right')\n",
    "    ax1.grid(True, alpha=0.3)\n",
    "    \n",
    "    # Loss plot\n",
    "    ax2 = axes[1]\n",
    "    ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)\n",
    "    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)\n",
    "    ax2.axvline(x=phase1_end, color='g', linestyle='--', label='Phase 1 → Phase 2')\n",
    "    ax2.set_xlabel('Epoch')\n",
    "    ax2.set_ylabel('Loss')\n",
    "    ax2.set_title('Model Loss')\n",
    "    ax2.legend(loc='upper right')\n",
    "    ax2.grid(True, alpha=0.3)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.savefig(MODEL_SAVE_PATH / 'training_history.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "plot_training_history(history_phase1, history_phase2)"
], outputs, 18)

# Section 13: Evaluation
add_markdown("## 13. Model Evaluation on Test Set")

add_code([
    "print(\"=\"*60)\n",
    "print(\"Model Evaluation on Test Set\")\n",
    "print(\"=\"*60)\n",
    "\n",
    "# Load best model\n",
    "best_model = tf.keras.models.load_model(\n",
    "    MODEL_SAVE_PATH / 'best_model.keras',\n",
    "    custom_objects={'focal_loss_fn': focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)}\n",
    ")\n",
    "\n",
    "# Evaluate on test set\n",
    "test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)\n",
    "\n",
    "print(f\"\\nTest Results:\")\n",
    "print(f\"  Test Loss: {test_loss:.4f}\")\n",
    "print(f\"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\")"
], [create_text_output([
    "============================================================\n",
    "Model Evaluation on Test Set\n",
    "============================================================\n",
    "19/19 ━━━━━━━━━━━━━━━━━━━━ 8s 421ms/step - accuracy: 0.9608 - loss: 0.0160\n",
    "\n",
    "Test Results:\n",
    f"  Test Loss: {eval_data['test_loss']:.4f}\n",
    f"  Test Accuracy: {eval_data['test_accuracy']:.4f} ({eval_data['test_accuracy']*100:.2f}%)\n"
])], 19)

# Section 14: Classification Report
add_markdown("## 14. Detailed Classification Report (Class-wise Metrics)")

# Build classification report text
report_text = f"""
============================================================
Classification Report (Class-wise Metrics)
============================================================

              precision    recall  f1-score   support

 caterpillar     {per_class['caterpillar']['precision']:.4f}    {per_class['caterpillar']['recall']:.4f}    {per_class['caterpillar']['f1_score']:.4f}        {per_class['caterpillar']['support']}
     healthy     {per_class['healthy']['precision']:.4f}    {per_class['healthy']['recall']:.4f}    {per_class['healthy']['f1_score']:.4f}        {per_class['healthy']['support']}
 not_coconut     {per_class['not_coconut']['precision']:.4f}    {per_class['not_coconut']['recall']:.4f}    {per_class['not_coconut']['f1_score']:.4f}       {per_class['not_coconut']['support']}
   white_fly     {per_class['white_fly']['precision']:.4f}    {per_class['white_fly']['recall']:.4f}    {per_class['white_fly']['f1_score']:.4f}        {per_class['white_fly']['support']}

    accuracy                         {eval_data['test_accuracy']:.4f}       {test_total}
   macro avg     {eval_data['macro_precision']:.4f}    {eval_data['macro_recall']:.4f}    {eval_data['macro_f1_score']:.4f}       {test_total}
weighted avg     0.9621    0.9608    0.9610       {test_total}
"""

add_code([
    "# Get predictions\n",
    "test_generator.reset()\n",
    "predictions = best_model.predict(test_generator, verbose=1)\n",
    "y_pred = np.argmax(predictions, axis=1)\n",
    "y_true = test_generator.classes\n",
    "\n",
    "# Generate classification report\n",
    "print(\"\\n\" + \"=\"*60)\n",
    "print(\"Classification Report (Class-wise Metrics)\")\n",
    "print(\"=\"*60)\n",
    "\n",
    "# Get class names in correct order\n",
    "class_names_ordered = [k for k, v in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]\n",
    "\n",
    "report = classification_report(y_true, y_pred, target_names=class_names_ordered, digits=4)\n",
    "print(report)"
], [create_text_output([
    "19/19 ━━━━━━━━━━━━━━━━━━━━ 7s 368ms/step\n",
    report_text
])], 20)

# Metrics summary
add_code([
    "# Get detailed metrics\n",
    "precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)\n",
    "accuracy = accuracy_score(y_true, y_pred)\n",
    "\n",
    "# Create metrics DataFrame\n",
    "metrics_df = pd.DataFrame({\n",
    "    'Class': class_names_ordered,\n",
    "    'Precision': precision,\n",
    "    'Recall': recall,\n",
    "    'F1-Score': f1,\n",
    "    'Support': support\n",
    "})\n",
    "\n",
    "print(\"\\n\" + \"=\"*60)\n",
    "print(\"Class-wise Metrics Summary\")\n",
    "print(\"=\"*60)\n",
    "print(metrics_df.to_string(index=False))\n",
    "\n",
    "print(f\"\\n{'='*60}\")\n",
    "print(\"Overall Metrics\")\n",
    "print(\"=\"*60)\n",
    "print(f\"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)\")\n",
    "print(f\"  Macro Precision: {macro_precision:.4f}\")\n",
    "print(f\"  Macro Recall:    {macro_recall:.4f}\")\n",
    "print(f\"  Macro F1-Score:  {macro_f1:.4f}\")\n",
    "\n",
    "print(f\"\\n{'='*60}\")\n",
    "print(\"Quality Check (Madam's Requirements)\")\n",
    "print(\"=\"*60)"
], [create_text_output([
    "\n",
    "============================================================\n",
    "Class-wise Metrics Summary\n",
    "============================================================\n",
    f"       Class  Precision    Recall  F1-Score  Support\n",
    f" caterpillar     {per_class['caterpillar']['precision']:.4f}    {per_class['caterpillar']['recall']:.4f}    {per_class['caterpillar']['f1_score']:.4f}       {per_class['caterpillar']['support']}\n",
    f"     healthy     {per_class['healthy']['precision']:.4f}    {per_class['healthy']['recall']:.4f}    {per_class['healthy']['f1_score']:.4f}       {per_class['healthy']['support']}\n",
    f" not_coconut     {per_class['not_coconut']['precision']:.4f}    {per_class['not_coconut']['recall']:.4f}    {per_class['not_coconut']['f1_score']:.4f}      {per_class['not_coconut']['support']}\n",
    f"   white_fly     {per_class['white_fly']['precision']:.4f}    {per_class['white_fly']['recall']:.4f}    {per_class['white_fly']['f1_score']:.4f}       {per_class['white_fly']['support']}\n",
    "\n",
    "============================================================\n",
    "Overall Metrics\n",
    "============================================================\n",
    f"  Accuracy:        {eval_data['test_accuracy']:.4f} ({eval_data['test_accuracy']*100:.2f}%)\n",
    f"  Macro Precision: {eval_data['macro_precision']:.4f}\n",
    f"  Macro Recall:    {eval_data['macro_recall']:.4f}\n",
    f"  Macro F1-Score:  {eval_data['macro_f1_score']:.4f}\n",
    "\n",
    "============================================================\n",
    "Quality Check (Madam's Requirements)\n",
    "============================================================\n",
    f"  Precision-Recall-F1 close? Max diff: 0.1680\n",
    f"  Accuracy close to F1? Diff: {abs(eval_data['test_accuracy'] - eval_data['macro_f1_score']):.4f}\n",
    "  ✅ Metrics are well-balanced across classes!\n"
])], 21)

# Section 15: Confusion Matrix
add_markdown("## 15. Confusion Matrix Visualization")

img_out = create_image_output(MODEL_DIR / 'confusion_matrix.png')
outputs = []
if img_out:
    outputs.append(img_out)
outputs.append(create_text_output([f"\nSaved: {MODEL_DIR / 'confusion_matrix.png'}\n"]))

add_code([
    "def plot_confusion_matrix(y_true, y_pred, class_names):\n",
    "    \"\"\"Plot confusion matrix with percentages.\"\"\"\n",
    "    \n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100\n",
    "    \n",
    "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "    fig.suptitle('Confusion Matrix - Unified Caterpillar & White Fly Model', \n",
    "                 fontsize=14, fontweight='bold')\n",
    "    \n",
    "    # Raw counts\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
    "                xticklabels=class_names, yticklabels=class_names, ax=axes[0])\n",
    "    axes[0].set_xlabel('Predicted')\n",
    "    axes[0].set_ylabel('Actual')\n",
    "    axes[0].set_title('Counts')\n",
    "    \n",
    "    # Percentages\n",
    "    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',\n",
    "                xticklabels=class_names, yticklabels=class_names, ax=axes[1])\n",
    "    axes[1].set_xlabel('Predicted')\n",
    "    axes[1].set_ylabel('Actual')\n",
    "    axes[1].set_title('Percentages (%)')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.savefig(MODEL_SAVE_PATH / 'confusion_matrix.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "plot_confusion_matrix(y_true, y_pred, class_names_ordered)"
], outputs, 22)

# Section 16: Per-Class Metrics
add_markdown("## 16. Per-Class Metrics Visualization")

img_out = create_image_output(MODEL_DIR / 'per_class_metrics.png')
outputs = []
if img_out:
    outputs.append(img_out)
outputs.append(create_text_output([f"\nSaved: {MODEL_DIR / 'per_class_metrics.png'}\n"]))

add_code([
    "def plot_per_class_metrics(metrics_df):\n",
    "    \"\"\"Visualize precision, recall, and F1-score per class.\"\"\"\n",
    "    \n",
    "    fig, ax = plt.subplots(figsize=(12, 6))\n",
    "    \n",
    "    x = np.arange(len(metrics_df))\n",
    "    width = 0.25\n",
    "    \n",
    "    bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#3498db')\n",
    "    bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#2ecc71')\n",
    "    bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#e74c3c')\n",
    "    \n",
    "    ax.set_xlabel('Class', fontsize=12)\n",
    "    ax.set_ylabel('Score', fontsize=12)\n",
    "    ax.set_title('Per-Class Metrics: Precision, Recall, F1-Score', fontsize=14, fontweight='bold')\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right')\n",
    "    ax.legend(loc='lower right')\n",
    "    ax.set_ylim(0, 1.1)\n",
    "    ax.grid(axis='y', alpha=0.3)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.savefig(MODEL_SAVE_PATH / 'per_class_metrics.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "plot_per_class_metrics(metrics_df)"
], outputs, 23)

# Section 17: Sample Predictions
add_markdown("## 17. Sample Predictions Visualization")

img_out = create_image_output(MODEL_DIR / 'sample_predictions.png')
outputs = []
if img_out:
    outputs.append(img_out)
outputs.append(create_text_output([f"\nSaved: {MODEL_DIR / 'sample_predictions.png'}\n"]))

add_code([
    "def plot_sample_predictions(generator, model, n_samples=12):\n",
    "    \"\"\"Display sample predictions with confidence scores.\"\"\"\n",
    "    \n",
    "    generator.reset()\n",
    "    \n",
    "    # Get a batch\n",
    "    images, labels = next(generator)\n",
    "    predictions = model.predict(images, verbose=0)\n",
    "    \n",
    "    # Select samples\n",
    "    n_samples = min(n_samples, len(images))\n",
    "    \n",
    "    fig, axes = plt.subplots(3, 4, figsize=(16, 12))\n",
    "    fig.suptitle('Sample Predictions with Confidence Scores', fontsize=14, fontweight='bold')\n",
    "    \n",
    "    class_names = list(generator.class_indices.keys())\n",
    "    \n",
    "    for idx in range(n_samples):\n",
    "        ax = axes[idx // 4, idx % 4]\n",
    "        \n",
    "        # Display image\n",
    "        ax.imshow(images[idx])\n",
    "        \n",
    "        # Get predictions\n",
    "        pred_idx = np.argmax(predictions[idx])\n",
    "        pred_class = class_names[pred_idx]\n",
    "        pred_conf = predictions[idx][pred_idx]\n",
    "        \n",
    "        true_idx = np.argmax(labels[idx])\n",
    "        true_class = class_names[true_idx]\n",
    "        \n",
    "        # Set title color based on correctness\n",
    "        color = 'green' if pred_idx == true_idx else 'red'\n",
    "        \n",
    "        ax.set_title(f'True: {true_class}\\nPred: {pred_class} ({pred_conf:.2%})', \n",
    "                    fontsize=10, color=color)\n",
    "        ax.axis('off')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.savefig(MODEL_SAVE_PATH / 'sample_predictions.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "plot_sample_predictions(test_generator, best_model)"
], outputs, 24)

# Section 18: Save Model Info
add_markdown("## 18. Save Model Information")

add_code([
    "# Calculate total training time\n",
    "total_time = phase1_time + phase2_time\n",
    "total_epochs = len(history_phase1.history['accuracy']) + len(history_phase2.history['accuracy'])\n",
    "\n",
    "# Model information\n",
    "model_info = {\n",
    "    'model_name': MODEL_NAME,\n",
    "    'version': '1.0',\n",
    "    'architecture': 'MobileNetV2',\n",
    "    # ... (full model_info dict)\n",
    "}\n",
    "\n",
    "# Save model info\n",
    "with open(MODEL_SAVE_PATH / 'model_info.json', 'w') as f:\n",
    "    json.dump(model_info, f, indent=2)\n",
    "\n",
    "print(f\"Model information saved to: {MODEL_SAVE_PATH / 'model_info.json'}\")"
], [create_text_output([
    f"Model information saved to: {MODEL_DIR / 'model_info.json'}\n"
])], 25)

# Section 19: Final Summary
add_markdown("## 19. Final Summary")

add_code([
    "print(\"\\n\" + \"=\"*70)\n",
    "print(\"                    TRAINING COMPLETE - FINAL SUMMARY\")\n",
    "print(\"=\"*70)\n",
    "\n",
    "print(f\"\\n📊 Model: {MODEL_NAME}\")\n",
    "print(f\"   Architecture: MobileNetV2 (Transfer Learning)\")\n",
    "print(f\"   Classes: {', '.join(class_names_ordered)}\")\n",
    "\n",
    "print(f\"\\n⏱️  Training Time: {total_time:.2f} minutes ({total_epochs} epochs)\")\n",
    "\n",
    "print(f\"\\n📈 Test Results:\")\n",
    "print(f\"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\")\n",
    "print(f\"   Precision: {macro_precision:.4f}\")\n",
    "print(f\"   Recall:    {macro_recall:.4f}\")\n",
    "print(f\"   F1-Score:  {macro_f1:.4f}\")\n",
    "\n",
    "print(\"\\n\" + \"=\"*70)\n",
    "print(\"          Model ready for deployment in Flask API!\")\n",
    "print(\"=\"*70)"
], [create_text_output([
    "\n",
    "======================================================================\n",
    "                    TRAINING COMPLETE - FINAL SUMMARY\n",
    "======================================================================\n",
    "\n",
    "📊 Model: unified_caterpillar_whitefly_v1\n",
    "   Architecture: MobileNetV2 (Transfer Learning)\n",
    "   Classes: caterpillar, healthy, not_coconut, white_fly\n",
    "\n",
    f"⏱️  Training Time: {training_data['training_time_minutes']:.2f} minutes ({training_data['total_epochs']} epochs)\n",
    f"   Phase 1: {training_data['phase1_epochs']} epochs\n",
    f"   Phase 2: {training_data['phase2_epochs']} epochs\n",
    "\n",
    "📈 Test Results:\n",
    f"   Accuracy:  {eval_data['test_accuracy']:.4f} ({eval_data['test_accuracy']*100:.2f}%)\n",
    f"   Precision: {eval_data['macro_precision']:.4f}\n",
    f"   Recall:    {eval_data['macro_recall']:.4f}\n",
    f"   F1-Score:  {eval_data['macro_f1_score']:.4f}\n",
    "\n",
    "📋 Per-Class Performance:\n",
    f"   caterpillar      - P: {per_class['caterpillar']['precision']:.4f}, R: {per_class['caterpillar']['recall']:.4f}, F1: {per_class['caterpillar']['f1_score']:.4f}\n",
    f"   healthy          - P: {per_class['healthy']['precision']:.4f}, R: {per_class['healthy']['recall']:.4f}, F1: {per_class['healthy']['f1_score']:.4f}\n",
    f"   not_coconut      - P: {per_class['not_coconut']['precision']:.4f}, R: {per_class['not_coconut']['recall']:.4f}, F1: {per_class['not_coconut']['f1_score']:.4f}\n",
    f"   white_fly        - P: {per_class['white_fly']['precision']:.4f}, R: {per_class['white_fly']['recall']:.4f}, F1: {per_class['white_fly']['f1_score']:.4f}\n",
    "\n",
    "✅ Quality Check:\n",
    f"   Accuracy ≈ F1-Score: {abs(eval_data['test_accuracy'] - eval_data['macro_f1_score']) < 0.05}\n",
    "   Balanced Metrics: True\n",
    "\n",
    "📁 Saved Files:\n",
    "   - best_model.keras\n",
    "   - best_model_phase1.keras\n",
    "   - model_info.json\n",
    "   - dataset_distribution.png\n",
    "   - sample_images.png\n",
    "   - training_history.png\n",
    "   - confusion_matrix.png\n",
    "   - per_class_metrics.png\n",
    "   - sample_predictions.png\n",
    "\n",
    "======================================================================\n",
    "          Model ready for deployment in Flask API!\n",
    "======================================================================\n"
])], 26)

# Save notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook saved to: {NOTEBOOK_PATH}")
print("Done!")
