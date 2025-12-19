"""
Convert best_model.keras to TFLite for mobile deployment
"""

import tensorflow as tf
import os
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'coconut_mite')

print("="*60)
print("  CONVERTING BEST MODEL TO TFLITE")
print("="*60)

# Load best model
print("\n1. Loading best_model.keras...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))
print("   ✅ Model loaded!")

# Convert to TFLite
print("\n2. Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizations for mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Quantize for smaller size (comment out if accuracy drops)
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
print("   ✅ Conversion complete!")

# Save TFLite model
tflite_path = os.path.join(MODEL_DIR, 'coconut_mite_model.tflite')
print(f"\n3. Saving to {tflite_path}...")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# Get file size
size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"   ✅ Saved! Size: {size_mb:.2f} MB")

# Update model info
print("\n4. Updating model_info.json...")
info_path = os.path.join(MODEL_DIR, 'model_info.json')
with open(info_path, 'r') as f:
    model_info = json.load(f)

model_info['tflite_size_mb'] = round(size_mb, 2)
model_info['model_file'] = 'best_model.keras'

with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
print("   ✅ Updated!")

print("\n" + "="*60)
print("  ✅ TFLITE MODEL READY FOR MOBILE!")
print("="*60)
print(f"\n  File: {tflite_path}")
print(f"  Size: {size_mb:.2f} MB")
print("\n  Next: Copy this file to React Native app")
