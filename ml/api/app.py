"""
Coconut Health Monitor - Flask API for Pest Detection
======================================================
This API serves the trained coconut mite detection model.
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import tensorflow as tf
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native app

# Configuration - Use best_model.keras (best validation accuracy!)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'coconut_mite', 'best_model.keras')
MODEL_INFO_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'coconut_mite', 'model_info.json')

# Global variables
model = None
model_info = None

def load_model():
    """Load the trained model and model info"""
    global model, model_info

    print("Loading coconut mite detection model...")

    # Load model info
    with open(MODEL_INFO_PATH, 'r') as f:
        model_info = json.load(f)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Model loaded successfully!")
    print(f"  - Model: {model_info['model_name']}")
    print(f"  - Input shape: {model_info['input_shape']}")
    print(f"  - Classes: {model_info['classes']}")
    print(f"  - Test accuracy: {model_info['test_accuracy']:.2%}")

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    # Open image
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to model input size
    img_size = model_info['input_shape'][0]
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize using ImageNet stats
    mean = np.array(model_info['normalization']['mean'])
    std = np.array(model_info['normalization']['std'])
    img_array = (img_array / 255.0 - mean) / std

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Coconut Health Monitor - Pest Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'POST - Predict pest infection from image',
            '/model-info': 'GET - Model information'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'model_name': model_info['model_name'],
        'input_shape': model_info['input_shape'],
        'classes': model_info['classes'],
        'test_accuracy': model_info['test_accuracy'],
        'training_date': model_info['training_date']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict pest infection from uploaded image"""

    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Use form field "image"'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Preprocess image
        processed_image = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)

        # Get class probabilities
        probs = predictions[0]
        predicted_class_idx = np.argmax(probs)
        predicted_class = model_info['classes'][predicted_class_idx]
        confidence = float(probs[predicted_class_idx])

        # Prepare response
        result = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': predicted_class == 'coconut_mite',
                'label': 'Coconut Mite Infected' if predicted_class == 'coconut_mite' else 'Healthy'
            },
            'probabilities': {
                model_info['classes'][i]: float(probs[i])
                for i in range(len(model_info['classes']))
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict pest infection for multiple images"""

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'images' not in request.files:
        return jsonify({'error': 'No images provided. Use form field "images"'}), 400

    files = request.files.getlist('images')

    if len(files) == 0:
        return jsonify({'error': 'No images selected'}), 400

    results = []

    for file in files:
        try:
            image_bytes = file.read()
            processed_image = preprocess_image(image_bytes)
            predictions = model.predict(processed_image, verbose=0)

            probs = predictions[0]
            predicted_class_idx = np.argmax(probs)
            predicted_class = model_info['classes'][predicted_class_idx]
            confidence = float(probs[predicted_class_idx])

            results.append({
                'filename': file.filename,
                'success': True,
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'is_infected': predicted_class == 'coconut_mite',
                    'label': 'Coconut Mite Infected' if predicted_class == 'coconut_mite' else 'Healthy'
                },
                'probabilities': {
                    model_info['classes'][i]: float(probs[i])
                    for i in range(len(model_info['classes']))
                }
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'count': len(results),
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()

    # Run the Flask app
    print("\nStarting Coconut Health Monitor API...")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
