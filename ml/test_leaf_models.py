"""
Quick test script for leaf color models.
Usage:  python test_leaf_models.py <image_path>
        python test_leaf_models.py  (uses a random test image)
"""

import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model configs ──────────────────────────────────────────────────────────
MODELS = {
    "leaf_color_detector_v1 (EfficientNetB3)": {
        "path":    os.path.join(BASE_DIR, "models", "leaf_color_detector_v1", "best_model.keras"),
        "classes": ["healthy", "unhealthy"],
        "size":    224,
        "preprocess": "efficientnetb3",
    },
    "coconut_leafe_Health_v5 (EfficientNetB0)": {
        "path":    os.path.join(BASE_DIR, "models", "coconut_leafe_Health_v5", "best_model.keras"),
        "classes": ["healthy", "unhealthy"],
        "size":    224,
        "preprocess": "efficientnetb0",
    },
}

# ── Test data dirs ─────────────────────────────────────────────────────────
TEST_DIRS = [
    os.path.join(BASE_DIR, "data", "processed", "leaf_color_pure_v1", "test", "healthy"),
    os.path.join(BASE_DIR, "data", "processed", "leaf_color_pure_v1", "test", "unhealthy"),
    os.path.join(BASE_DIR, "data", "processed", "leafe_health_v5",    "test", "healthy"),
    os.path.join(BASE_DIR, "data", "processed", "leafe_health_v5",    "test", "unhealthy"),
]


def load_image(img_path, size):
    img = Image.open(img_path).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32)
    return arr


def predict(model_name, cfg, img_path):
    if not os.path.exists(cfg["path"]):
        print(f"  [{model_name}] model file not found: {cfg['path']}")
        return

    model = keras.models.load_model(cfg["path"], compile=False)

    arr  = load_image(img_path, cfg["size"])

    # Apply same preprocessing as during training
    if cfg["preprocess"] == "efficientnetb3":
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    elif cfg["preprocess"] == "efficientnetb0":
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    inp    = np.expand_dims(arr, axis=0)
    probs  = model.predict(inp, verbose=0)[0]
    pred   = np.argmax(probs)
    label  = cfg["classes"][pred]
    conf   = probs[pred] * 100

    print(f"\n  Model   : {model_name}")
    print(f"  Result  : {label.upper()}  ({conf:.1f}% confidence)")
    for i, cls in enumerate(cfg["classes"]):
        bar = "█" * int(probs[i] * 40)
        print(f"  {cls:10s}: {probs[i]*100:5.1f}%  {bar}")


def get_random_test_image():
    imgs = []
    for d in TEST_DIRS:
        if os.path.exists(d):
            for f in os.listdir(d)[:20]:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    imgs.append((os.path.join(d, f), os.path.basename(os.path.dirname(d))))
    if imgs:
        path, true_label = random.choice(imgs)
        return path, true_label
    return None, None


# ── Main ───────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    img_path   = sys.argv[1]
    true_label = "unknown"
else:
    img_path, true_label = get_random_test_image()
    if not img_path:
        print("No test images found. Provide image path as argument.")
        sys.exit(1)

if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    sys.exit(1)

print("=" * 55)
print(f"  Image      : {os.path.basename(img_path)}")
print(f"  True label : {true_label}")
print("=" * 55)

for name, cfg in MODELS.items():
    predict(name, cfg, img_path)

print("\n" + "=" * 55)
