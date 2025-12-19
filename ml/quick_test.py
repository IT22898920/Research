"""
Quick test script - Check if coconut has mite disease
Directly uses the model (no API server needed!)

Usage: python quick_test.py <image_path>

Example:
  python quick_test.py "path/to/coconut_image.jpg"
  python quick_test.py data/raw/pest/coconut_mite/aug_001_1.jpg

No arguments = tests sample images automatically
"""

import sys
import os
import json
import numpy as np
from PIL import Image

# Add parent path
sys.path.insert(0, os.path.dirname(__file__))

def load_model():
    """Load the trained model"""
    import tensorflow as tf

    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'coconut_mite')
    # Use best_model.keras - this has the best validation accuracy!
    model_path = os.path.join(model_dir, 'best_model.keras')
    info_path = os.path.join(model_dir, 'model_info.json')

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    with open(info_path, 'r') as f:
        model_info = json.load(f)

    print(f"Model loaded: {model_info['model_name']}")
    print(f"Test Accuracy: {model_info['test_accuracy']:.2%}")

    return model, model_info

def predict_image(image_path, model, model_info):
    """Predict if coconut has mite disease"""

    # Load and preprocess image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_size = model_info['input_shape'][0]  # 224
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32)
    mean = np.array(model_info['normalization']['mean'])
    std = np.array(model_info['normalization']['std'])
    img_array = (img_array / 255.0 - mean) / std
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0]

    predicted_idx = np.argmax(probs)
    predicted_class = model_info['classes'][predicted_idx]
    confidence = float(probs[predicted_idx])

    return {
        'class': predicted_class,
        'confidence': confidence,
        'is_infected': predicted_class == 'coconut_mite',
        'probabilities': {
            model_info['classes'][i]: float(probs[i])
            for i in range(len(model_info['classes']))
        }
    }

def main():
    if len(sys.argv) < 2:
        # If no argument, test with sample images
        print("No image path provided. Testing with sample images...\n")

        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'pest')

        test_images = []

        # Get one mite image
        mite_dir = os.path.join(data_dir, 'coconut_mite')
        if os.path.exists(mite_dir):
            mite_files = [f for f in os.listdir(mite_dir) if f.endswith('.jpg')]
            if mite_files:
                test_images.append(('MITE SAMPLE', os.path.join(mite_dir, mite_files[0])))

        # Get one healthy image
        healthy_dir = os.path.join(data_dir, 'healthy')
        if os.path.exists(healthy_dir):
            healthy_files = [f for f in os.listdir(healthy_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
            if healthy_files:
                test_images.append(('HEALTHY SAMPLE', os.path.join(healthy_dir, healthy_files[0])))

        if not test_images:
            print("No test images found!")
            print("Usage: python quick_test.py <image_path>")
            return

        image_paths = test_images
    else:
        image_paths = [('USER IMAGE', sys.argv[1])]

    # Load model once
    model, model_info = load_model()

    print("\n" + "=" * 50)
    print("  COCONUT MITE DETECTION RESULTS")
    print("=" * 50)

    for label, image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"\n[{label}] File not found: {image_path}")
            continue

        print(f"\n[{label}]")
        print(f"  File: {os.path.basename(image_path)}")

        result = predict_image(image_path, model, model_info)

        if result['is_infected']:
            status = "INFECTED - Coconut Mite Detected!"
        else:
            status = "HEALTHY - No Mite Detected"

        print(f"  Status: {status}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"    - {cls}: {prob:.2%}")

    print("\n" + "=" * 50)

if __name__ == '__main__':
    main()
