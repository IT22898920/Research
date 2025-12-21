"""
Coconut Health Monitor - Flask API for Pest Detection
======================================================
This API serves trained models for:
- Coconut Mite Detection
- Coconut Caterpillar Detection
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

# Configuration paths
BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')

# Mite model paths
MITE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite', 'best_model.keras')
MITE_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite', 'model_info.json')

# Caterpillar model paths (using final model with optimal threshold)
CATERPILLAR_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_caterpillar', 'caterpillar_model.keras')
CATERPILLAR_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_caterpillar', 'model_info.json')

# Optimal threshold for caterpillar model (balanced P/R/F1)
CATERPILLAR_THRESHOLD = 0.20

# Global variables for models
models = {}
model_infos = {}

def load_models():
    """Load all trained models"""
    global models, model_infos

    print("=" * 60)
    print("  Loading Coconut Health Monitor Models...")
    print("=" * 60)

    # Load Mite Model
    try:
        print("\n[1] Loading Coconut Mite model...")
        with open(MITE_MODEL_INFO_PATH, 'r') as f:
            model_infos['mite'] = json.load(f)
        models['mite'] = tf.keras.models.load_model(MITE_MODEL_PATH)
        print(f"    Model: {model_infos['mite']['model_name']}")
        print(f"    Accuracy: {model_infos['mite'].get('test_accuracy', 'N/A')}")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading mite model: {e}")
        models['mite'] = None
        model_infos['mite'] = None

    # Load Caterpillar Model (final with optimal threshold)
    try:
        print("\n[2] Loading Coconut Caterpillar model...")
        with open(CATERPILLAR_MODEL_INFO_PATH, 'r') as f:
            model_infos['caterpillar'] = json.load(f)
        models['caterpillar'] = tf.keras.models.load_model(CATERPILLAR_MODEL_PATH)
        print(f"    Model: {model_infos['caterpillar']['model_name']}")
        print(f"    Accuracy: {model_infos['caterpillar']['performance']['accuracy']:.2%}")
        print(f"    Threshold: {CATERPILLAR_THRESHOLD} (optimized for balanced P/R/F1)")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading caterpillar model: {e}")
        models['caterpillar'] = None
        model_infos['caterpillar'] = None

    print("\n" + "=" * 60)
    loaded_count = sum(1 for m in models.values() if m is not None)
    print(f"  Models loaded: {loaded_count}/2")
    print("=" * 60)

def preprocess_image_mite(image_bytes):
    """Preprocess image for mite model (ImageNet normalization)"""
    info = model_infos['mite']
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_size = info['input_shape'][0]
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    # ImageNet normalization
    mean = np.array(info['normalization']['mean'])
    std = np.array(info['normalization']['std'])
    img_array = (img_array / 255.0 - mean) / std

    return np.expand_dims(img_array, axis=0)

def preprocess_image_caterpillar(image_bytes):
    """Preprocess image for caterpillar model (simple 0-1 scaling)"""
    info = model_infos['caterpillar']
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_size = info['input_size'][0]  # Note: 'input_size' not 'input_shape'
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    # Simple 0-1 normalization (same as training)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Coconut Health Monitor - Pest Detection API',
        'version': '2.0.0',
        'status': 'running',
        'models': {
            'mite': 'loaded' if models.get('mite') is not None else 'not loaded',
            'caterpillar': 'loaded' if models.get('caterpillar') is not None else 'not loaded'
        },
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/models': 'List all available models',
            '/predict/mite': 'POST - Detect coconut mite infection',
            '/predict/caterpillar': 'POST - Detect caterpillar damage',
            '/predict/all': 'POST - Run all pest detection models'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'mite': models.get('mite') is not None,
            'caterpillar': models.get('caterpillar') is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models with their info"""
    result = {}

    if model_infos.get('mite'):
        info = model_infos['mite']
        result['mite'] = {
            'name': info['model_name'],
            'classes': info['classes'],
            'accuracy': info.get('test_accuracy'),
            'loaded': models.get('mite') is not None
        }

    if model_infos.get('caterpillar'):
        info = model_infos['caterpillar']
        result['caterpillar'] = {
            'name': info['model_name'],
            'classes': info['classes'],
            'accuracy': info['performance']['accuracy'],
            'threshold': CATERPILLAR_THRESHOLD,
            'loaded': models.get('caterpillar') is not None
        }

    return jsonify(result)

@app.route('/predict/mite', methods=['POST'])
def predict_mite():
    """Detect coconut mite infection"""

    if models.get('mite') is None:
        return jsonify({'error': 'Mite model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_mite(image_bytes)
        predictions = models['mite'].predict(processed_image, verbose=0)

        probs = predictions[0]
        predicted_class_idx = np.argmax(probs)
        info = model_infos['mite']
        predicted_class = info['classes'][predicted_class_idx]
        confidence = float(probs[predicted_class_idx])

        return jsonify({
            'success': True,
            'pest_type': 'mite',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': predicted_class == 'coconut_mite',
                'label': 'Coconut Mite Infected' if predicted_class == 'coconut_mite' else 'Healthy'
            },
            'probabilities': {
                info['classes'][i]: float(probs[i])
                for i in range(len(info['classes']))
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/caterpillar', methods=['POST'])
def predict_caterpillar():
    """Detect caterpillar damage"""

    if models.get('caterpillar') is None:
        return jsonify({'error': 'Caterpillar model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_caterpillar(image_bytes)
        prediction = models['caterpillar'].predict(processed_image, verbose=0)[0][0]

        # Binary classification using OPTIMAL threshold (0.20) for balanced P/R/F1
        # < threshold = caterpillar, >= threshold = healthy
        info = model_infos['caterpillar']
        is_caterpillar = bool(prediction < CATERPILLAR_THRESHOLD)
        confidence = float(1 - prediction if is_caterpillar else prediction)

        predicted_class = 'caterpillar' if is_caterpillar else 'healthy'

        return jsonify({
            'success': True,
            'pest_type': 'caterpillar',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_caterpillar,
                'label': 'Caterpillar Damage Detected' if is_caterpillar else 'Healthy',
                'raw_score': float(prediction),
                'threshold_used': CATERPILLAR_THRESHOLD
            },
            'probabilities': {
                'caterpillar': float(1 - prediction),
                'healthy': float(prediction)
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """Run all available pest detection models on the image"""

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    image_bytes = file.read()
    results = {}
    detected_pests = []

    # Run Mite Detection
    if models.get('mite') is not None:
        try:
            processed = preprocess_image_mite(image_bytes)
            preds = models['mite'].predict(processed, verbose=0)[0]
            idx = np.argmax(preds)
            info = model_infos['mite']
            predicted_class = info['classes'][idx]

            results['mite'] = {
                'class': predicted_class,
                'confidence': float(preds[idx]),
                'is_infected': predicted_class == 'coconut_mite'
            }
            if predicted_class == 'coconut_mite':
                detected_pests.append('Coconut Mite')
        except Exception as e:
            results['mite'] = {'error': str(e)}

    # Run Caterpillar Detection
    if models.get('caterpillar') is not None:
        try:
            # Reset file pointer for second read
            file.seek(0)
            processed = preprocess_image_caterpillar(image_bytes)
            pred = models['caterpillar'].predict(processed, verbose=0)[0][0]

            # Use optimal threshold (0.20) for balanced P/R/F1
            is_caterpillar = bool(pred < CATERPILLAR_THRESHOLD)
            results['caterpillar'] = {
                'class': 'caterpillar' if is_caterpillar else 'healthy',
                'confidence': float(1 - pred if is_caterpillar else pred),
                'is_infected': is_caterpillar,
                'threshold_used': CATERPILLAR_THRESHOLD
            }
            if is_caterpillar:
                detected_pests.append('Caterpillar')
        except Exception as e:
            results['caterpillar'] = {'error': str(e)}

    return jsonify({
        'success': True,
        'results': results,
        'summary': {
            'pests_detected': detected_pests,
            'is_healthy': len(detected_pests) == 0,
            'label': ', '.join(detected_pests) if detected_pests else 'Healthy'
        },
        'timestamp': datetime.now().isoformat()
    })

# Legacy endpoint for backward compatibility
@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy endpoint - redirects to mite detection"""
    return predict_mite()

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load all models on startup
    load_models()

    # Run the Flask app
    print("\nStarting Coconut Health Monitor API v2.0...")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
