"""
Test the BEST model (saved during training with best val_accuracy)
"""

import os
import json
import random
import numpy as np
from PIL import Image

def load_model():
    import tensorflow as tf
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'coconut_mite')

    # Use BEST model instead of final model
    model_path = os.path.join(model_dir, 'best_model.keras')
    print(f"Loading: {model_path}")

    model = tf.keras.models.load_model(model_path)

    with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
        model_info = json.load(f)
    return model, model_info

def predict_image(image_path, model, model_info):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    mean = np.array(model_info['normalization']['mean'])
    std = np.array(model_info['normalization']['std'])
    img_array = (img_array / 255.0 - mean) / std
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0]
    predicted_idx = np.argmax(probs)

    return {
        'class': model_info['classes'][predicted_idx],
        'confidence': float(probs[predicted_idx]),
        'mite_prob': float(probs[0]),
        'healthy_prob': float(probs[1])
    }

def main():
    print("="*60)
    print("  TESTING BEST MODEL (best_model.keras)")
    print("="*60)

    model, model_info = load_model()
    print(f"Model: {model_info['model_name']}\n")

    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'pest')
    mite_dir = os.path.join(data_dir, 'coconut_mite')
    healthy_dir = os.path.join(data_dir, 'healthy')

    mite_files = [f for f in os.listdir(mite_dir) if f.endswith('.jpg')]
    healthy_files = [f for f in os.listdir(healthy_dir) if f.endswith(('.jpg', '.JPG'))]

    # Test 5 random from each class
    print("Testing 5 MITE INFECTED images:")
    print("-"*60)
    mite_correct = 0
    for f in random.sample(mite_files, min(5, len(mite_files))):
        path = os.path.join(mite_dir, f)
        result = predict_image(path, model, model_info)
        status = "CORRECT" if result['class'] == 'coconut_mite' else "WRONG"
        if result['class'] == 'coconut_mite':
            mite_correct += 1
        print(f"  {f[:25]:25} -> {result['class']:12} ({result['confidence']:.1%}) [{status}]")

    print(f"\n  Mite Accuracy: {mite_correct}/5 = {mite_correct/5:.0%}")

    print("\n" + "Testing 5 HEALTHY images:")
    print("-"*60)
    healthy_correct = 0
    for f in random.sample(healthy_files, min(5, len(healthy_files))):
        path = os.path.join(healthy_dir, f)
        result = predict_image(path, model, model_info)
        status = "CORRECT" if result['class'] == 'healthy' else "WRONG"
        if result['class'] == 'healthy':
            healthy_correct += 1
        print(f"  {f[:25]:25} -> {result['class']:12} ({result['confidence']:.1%}) [{status}]")

    print(f"\n  Healthy Accuracy: {healthy_correct}/5 = {healthy_correct/5:.0%}")

    print("\n" + "="*60)
    total = mite_correct + healthy_correct
    print(f"OVERALL: {total}/10 = {total/10:.0%}")
    print("="*60)

if __name__ == '__main__':
    main()
