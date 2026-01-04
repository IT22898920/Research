"""
Leaf Health Prediction Script
Use the trained model to predict if a leaf is healthy or unhealthy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import argparse

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss function (needed to load the model)"""
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        focal_weight = tf.keras.backend.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * focal_weight * cross_entropy
        return tf.keras.backend.sum(focal_loss, axis=-1)
    return focal_loss_fn

def load_model(model_path='models/leaf_health_v1/best_model.keras'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(
        model_path,
        custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
    )
    print("Model loaded successfully!")
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    # Load image
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize(target_size)

    # Convert to array and normalize
    img_array = np.array(img) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict(model, image_path):
    """Make prediction on a single image"""
    # Preprocess image
    img_array = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    # Get class names
    class_names = ['healthy', 'unhealthy']

    # Get predicted class and confidence
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100

    # Get probabilities for both classes
    healthy_prob = prediction[0][0] * 100
    unhealthy_prob = prediction[0][1] * 100

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'healthy_probability': healthy_prob,
        'unhealthy_probability': unhealthy_prob
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict leaf health')
    parser.add_argument('image_path', type=str, help='Path to leaf image')
    parser.add_argument('--model', type=str,
                       default='models/leaf_health_v1/best_model.keras',
                       help='Path to model file')

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Make prediction
    print(f"\nAnalyzing image: {args.image_path}")
    print("-" * 60)

    result = predict(model, args.image_path)

    # Display results
    print(f"\nPrediction: {result['predicted_class'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"\nDetailed Probabilities:")
    print(f"  Healthy:   {result['healthy_probability']:.2f}%")
    print(f"  Unhealthy: {result['unhealthy_probability']:.2f}%")
    print("-" * 60)

    # Health recommendation
    if result['predicted_class'] == 'healthy':
        print("\n✓ Leaf appears healthy!")
    else:
        print("\n⚠ Leaf shows signs of yellowing/unhealthy condition")
        print("  Consider investigating possible causes:")
        print("  - Nutrient deficiency")
        print("  - Water stress")
        print("  - Disease or pest damage")

if __name__ == "__main__":
    main()
