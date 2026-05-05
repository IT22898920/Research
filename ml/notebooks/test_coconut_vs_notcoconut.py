"""
Quick prediction utility for the coconut_vs_notcoconut_v1 model.
Usage:
    python test_coconut_vs_notcoconut.py <image_path>
    python test_coconut_vs_notcoconut.py <folder>           # all images in folder
"""
import os, sys, json, glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, models

BASE    = r"D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research"
MODEL_D = os.path.join(BASE, "ml", "models", "coconut_vs_notcoconut_v1")
MODEL   = os.path.join(MODEL_D, "best_model.keras")

with open(os.path.join(MODEL_D, "model_info.json")) as f:
    info = json.load(f)
classes = info["classes"]
IMG     = info["input_size"][0]

print(f"Loading {MODEL} ...")
model = models.load_model(MODEL)
print("OK\n")

def predict_one(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG, IMG))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    pre = applications.mobilenet_v2.preprocess_input(arr)
    p   = float(model.predict(pre[None, ...], verbose=0)[0][0])
    cls = classes[int(p >= 0.5)]
    conf = p if p >= 0.5 else 1 - p
    return cls, conf, p

if len(sys.argv) < 2:
    print("usage: python test_coconut_vs_notcoconut.py <image_or_folder>")
    sys.exit(1)

target = sys.argv[1]
paths = []
if os.path.isdir(target):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(target, ext)))
else:
    paths = [target]

print(f"{'file':<60} {'pred':<14} {'conf':>7}  raw")
print("-" * 95)
for p in paths:
    cls, conf, raw = predict_one(p)
    print(f"{os.path.basename(p):<60} {cls:<14} {conf*100:>6.2f}%  {raw:.4f}")
