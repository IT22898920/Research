"""
Coconut Health Monitor - Flask API for Pest Detection
======================================================
This API serves trained models for:
- Coconut Mite Detection (v10 - separate 3-class model)
- Unified Caterpillar & White Fly Detection (v1 - combined 4-class model)
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

# Mite model paths (v10 - 3-class with Focal Loss)
MITE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v10', 'best_model.keras')
MITE_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v10', 'model_info.json')

# Mite v10 optimal threshold (from threshold tuning)
MITE_THRESHOLD = 0.10
MITE_BOOST_FACTOR = 0.5 / MITE_THRESHOLD  # 5x boost for mite class

# Mite v10 class indices
MITE_CLASSES = ['coconut_mite', 'healthy', 'not_coconut']

# Minimum confidence threshold for valid predictions
MIN_CONFIDENCE_THRESHOLD = 0.50

# Unified Caterpillar & White Fly model paths (v1 - 4-class with Focal Loss)
UNIFIED_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'unified_caterpillar_whitefly_v1', 'best_model.keras')
UNIFIED_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'unified_caterpillar_whitefly_v1', 'model_info.json')

# Unified model class indices (alphabetical order from ImageDataGenerator)
UNIFIED_CLASSES = ['caterpillar', 'healthy', 'not_coconut', 'white_fly']

# Global variables for models
models = {}
model_infos = {}

def focal_loss(gamma=2.0, alpha=0.25):
    """Custom focal loss for loading models"""
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        focal_weight = tf.keras.backend.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * focal_weight * cross_entropy
        return tf.keras.backend.sum(focal_loss, axis=-1)
    return focal_loss_fn

def load_models():
    """Load all trained models"""
    global models, model_infos

    print("=" * 60)
    print("  Loading Coconut Health Monitor Models...")
    print("=" * 60)

    # Load Mite Model (v10 - 3-class with Focal Loss)
    try:
        print("\n[1] Loading Coconut Mite model (v10 - 3-class)...")

        models['mite'] = tf.keras.models.load_model(
            MITE_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        try:
            with open(MITE_MODEL_INFO_PATH, 'r') as f:
                model_infos['mite'] = json.load(f)
        except:
            model_infos['mite'] = {
                'version': 'v10_mite_focused',
                'classes': MITE_CLASSES,
                'performance': {'test_accuracy': 0.9144, 'mite_recall': 0.79}
            }

        print(f"    Version: v10 (3-class, Focal Loss)")
        print(f"    Classes: {MITE_CLASSES}")
        print(f"    Accuracy: 91.44%")
        print(f"    Mite Recall: 79%")
        print(f"    Threshold: {MITE_THRESHOLD} (boost factor: {MITE_BOOST_FACTOR}x)")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading mite model: {e}")
        models['mite'] = None
        model_infos['mite'] = None

    # Load Unified Caterpillar & White Fly Model (v1 - 4-class with Focal Loss)
    try:
        print("\n[2] Loading Unified Caterpillar & White Fly model (v1 - 4-class)...")

        models['unified'] = tf.keras.models.load_model(
            UNIFIED_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        try:
            with open(UNIFIED_MODEL_INFO_PATH, 'r') as f:
                model_infos['unified'] = json.load(f)
        except:
            model_infos['unified'] = {
                'version': 'v1_4class',
                'classes': UNIFIED_CLASSES,
                'performance': {
                    'test_accuracy': 0.9608,
                    'caterpillar_recall': 0.9574,
                    'white_fly_recall': 0.8608
                }
            }

        print(f"    Version: v1 (4-class, Focal Loss)")
        print(f"    Classes: {UNIFIED_CLASSES}")
        print(f"    Accuracy: 96.08%")
        print(f"    Caterpillar Recall: 95.74%")
        print(f"    White Fly Recall: 86.08%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading unified model: {e}")
        models['unified'] = None
        model_infos['unified'] = None

    print("\n" + "=" * 60)
    loaded_count = sum(1 for m in models.values() if m is not None)
    print(f"  Models loaded: {loaded_count}/2")
    print("=" * 60)

def preprocess_image_mite(image_bytes):
    """Preprocess image for mite model v10 (0-1 scaling)"""
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)

def preprocess_image_unified(image_bytes):
    """Preprocess image for unified model (0-1 scaling, 224x224)"""
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Coconut Health Monitor - Pest Detection API',
        'version': '6.0.0',
        'models': {
            'mite': {
                'status': 'loaded' if models.get('mite') is not None else 'not loaded',
                'version': 'v10 (3-class, Focal Loss)',
                'accuracy': '91.44%'
            },
            'unified': {
                'status': 'loaded' if models.get('unified') is not None else 'not loaded',
                'version': 'v1 (4-class: caterpillar, healthy, not_coconut, white_fly)',
                'accuracy': '96.08%'
            }
        },
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/models': 'List all available models',
            '/predict/mite': 'POST - Detect coconut mite infection (3-class)',
            '/predict/caterpillar': 'POST - Detect caterpillar damage (uses unified 4-class model)',
            '/predict/white_fly': 'POST - Detect white fly damage (uses unified 4-class model)',
            '/predict/unified': 'POST - Unified caterpillar & white fly detection (4-class)',
            '/predict/all': 'POST - Run all pest detection with smart combined logic'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'mite': models.get('mite') is not None,
            'unified': models.get('unified') is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models with their info"""
    result = {}

    if model_infos.get('mite'):
        result['mite'] = {
            'name': 'Coconut Mite Detection Model',
            'version': 'v10 (3-class, Focal Loss)',
            'classes': MITE_CLASSES,
            'accuracy': 0.9144,
            'mite_recall': 0.79,
            'threshold': MITE_THRESHOLD,
            'boost_factor': MITE_BOOST_FACTOR,
            'loaded': models.get('mite') is not None
        }

    if model_infos.get('unified'):
        result['unified'] = {
            'name': 'Unified Caterpillar & White Fly Detection Model',
            'version': 'v1 (4-class, Focal Loss)',
            'classes': UNIFIED_CLASSES,
            'accuracy': 0.9608,
            'caterpillar_recall': 0.9574,
            'white_fly_recall': 0.8608,
            'loaded': models.get('unified') is not None
        }

    return jsonify(result)

@app.route('/predict/mite', methods=['POST'])
def predict_mite():
    """Detect coconut mite infection (v10 model - 3-class classification)"""

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

        # v10 3-class classification: softmax output
        predictions = models['mite'].predict(processed_image, verbose=0)[0]

        # Apply threshold adjustment (boost mite probability)
        adjusted_probs = predictions.copy()
        adjusted_probs[0] = adjusted_probs[0] * MITE_BOOST_FACTOR

        # Get predicted class
        predicted_idx = int(np.argmax(adjusted_probs))
        predicted_class = MITE_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        is_mite = predicted_class == 'coconut_mite'
        is_valid = predicted_class != 'not_coconut'

        probabilities = {
            'coconut_mite': float(predictions[0]),
            'healthy': float(predictions[1]),
            'not_coconut': float(predictions[2])
        }

        if not is_valid:
            return jsonify({
                'success': True,
                'pest_type': 'mite',
                'model_version': 'v10',
                'prediction': {
                    'class': 'not_coconut',
                    'confidence': confidence,
                    'is_infected': False,
                    'is_valid_image': False,
                    'label': 'Not a valid coconut image',
                    'message': 'The uploaded image does not appear to be a coconut. Please upload a clear image of a coconut fruit or leaf.'
                },
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({
            'success': True,
            'pest_type': 'mite',
            'model_version': 'v10',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_mite,
                'is_valid_image': True,
                'label': 'Coconut Mite Infected' if is_mite else 'Healthy'
            },
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/unified', methods=['POST'])
def predict_unified():
    """Unified caterpillar & white fly detection (v1 - 4-class model)"""

    if models.get('unified') is None:
        return jsonify({'error': 'Unified model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_unified(image_bytes)

        # 4-class classification: softmax output
        # Classes: ['caterpillar', 'healthy', 'not_coconut', 'white_fly']
        predictions = models['unified'].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = UNIFIED_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_caterpillar = predicted_class == 'caterpillar'
        is_white_fly = predicted_class == 'white_fly'
        is_infected = is_caterpillar or is_white_fly
        is_valid = predicted_class != 'not_coconut'

        probabilities = {
            'caterpillar': float(predictions[0]),
            'healthy': float(predictions[1]),
            'not_coconut': float(predictions[2]),
            'white_fly': float(predictions[3])
        }

        # Determine label
        if not is_valid:
            label = 'Not a valid coconut image'
            message = 'The uploaded image does not appear to be a coconut. Please upload a clear image of a coconut leaf.'
        elif is_caterpillar:
            label = 'Caterpillar Damage Detected'
            message = 'This coconut shows signs of caterpillar damage.'
        elif is_white_fly:
            label = 'White Fly Damage Detected'
            message = 'This coconut shows signs of white fly infestation.'
        else:
            label = 'Healthy'
            message = 'No pest damage detected.'

        return jsonify({
            'success': True,
            'pest_type': 'unified',
            'model_version': 'v1',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_infected,
                'is_caterpillar': is_caterpillar,
                'is_white_fly': is_white_fly,
                'is_valid_image': is_valid,
                'label': label,
                'message': message
            },
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/caterpillar', methods=['POST'])
def predict_caterpillar():
    """Detect caterpillar damage (uses unified 4-class model)"""

    if models.get('unified') is None:
        return jsonify({'error': 'Unified model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_unified(image_bytes)

        # 4-class classification
        predictions = models['unified'].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = UNIFIED_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_caterpillar = predicted_class == 'caterpillar'
        is_valid = predicted_class != 'not_coconut'

        probabilities = {
            'caterpillar': float(predictions[0]),
            'healthy': float(predictions[1]),
            'not_coconut': float(predictions[2]),
            'white_fly': float(predictions[3])
        }

        if not is_valid:
            return jsonify({
                'success': True,
                'pest_type': 'caterpillar',
                'model_version': 'unified_v1',
                'prediction': {
                    'class': 'not_coconut',
                    'confidence': confidence,
                    'is_infected': False,
                    'is_valid_image': False,
                    'label': 'Not a valid coconut image',
                    'message': 'The uploaded image does not appear to be a coconut. Please upload a clear image of a coconut leaf.'
                },
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({
            'success': True,
            'pest_type': 'caterpillar',
            'model_version': 'unified_v1',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_caterpillar,
                'is_valid_image': True,
                'label': 'Caterpillar Damage Detected' if is_caterpillar else ('White Fly Detected' if predicted_class == 'white_fly' else 'Healthy')
            },
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/white_fly', methods=['POST'])
def predict_white_fly():
    """Detect white fly damage (uses unified 4-class model)"""

    if models.get('unified') is None:
        return jsonify({'error': 'Unified model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_unified(image_bytes)

        # 4-class classification
        predictions = models['unified'].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = UNIFIED_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_white_fly = predicted_class == 'white_fly'
        is_valid = predicted_class != 'not_coconut'

        probabilities = {
            'caterpillar': float(predictions[0]),
            'healthy': float(predictions[1]),
            'not_coconut': float(predictions[2]),
            'white_fly': float(predictions[3])
        }

        if not is_valid:
            return jsonify({
                'success': True,
                'pest_type': 'white_fly',
                'model_version': 'unified_v1',
                'prediction': {
                    'class': 'not_coconut',
                    'confidence': confidence,
                    'is_infected': False,
                    'is_valid_image': False,
                    'label': 'Not a valid coconut image',
                    'message': 'The uploaded image does not appear to be a coconut. Please upload a clear image of a coconut leaf.'
                },
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({
            'success': True,
            'pest_type': 'white_fly',
            'model_version': 'unified_v1',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_white_fly,
                'is_valid_image': True,
                'label': 'White Fly Damage Detected' if is_white_fly else ('Caterpillar Detected' if predicted_class == 'caterpillar' else 'Healthy')
            },
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """
    Run all available pest detection models on the image.
    Uses: Mite v10 (3-class) + Unified v1 (4-class for caterpillar & white fly)
    """

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    image_bytes = file.read()
    results = {}
    detected_pests = []

    # Run Mite Detection (v10 - 3-class)
    if models.get('mite') is not None:
        try:
            processed = preprocess_image_mite(image_bytes)
            predictions = models['mite'].predict(processed, verbose=0)[0]

            # Apply threshold adjustment
            adjusted_probs = predictions.copy()
            adjusted_probs[0] = adjusted_probs[0] * MITE_BOOST_FACTOR

            predicted_idx = int(np.argmax(adjusted_probs))
            predicted_class = MITE_CLASSES[predicted_idx]
            confidence = float(predictions[predicted_idx])
            is_mite = predicted_class == 'coconut_mite'
            is_valid = predicted_class != 'not_coconut'

            results['mite'] = {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_mite,
                'is_valid_image': is_valid,
                'probabilities': {
                    'coconut_mite': float(predictions[0]),
                    'healthy': float(predictions[1]),
                    'not_coconut': float(predictions[2])
                }
            }
            if is_mite and is_valid:
                detected_pests.append('Coconut Mite')
        except Exception as e:
            results['mite'] = {'error': str(e)}

    # Run Unified Detection (v1 - 4-class for caterpillar & white fly)
    if models.get('unified') is not None:
        try:
            processed = preprocess_image_unified(image_bytes)
            predictions = models['unified'].predict(processed, verbose=0)[0]

            predicted_idx = int(np.argmax(predictions))
            predicted_class = UNIFIED_CLASSES[predicted_idx]
            confidence = float(predictions[predicted_idx])
            is_caterpillar = predicted_class == 'caterpillar'
            is_white_fly = predicted_class == 'white_fly'
            is_valid = predicted_class != 'not_coconut'

            probabilities = {
                'caterpillar': float(predictions[0]),
                'healthy': float(predictions[1]),
                'not_coconut': float(predictions[2]),
                'white_fly': float(predictions[3])
            }

            # Determine caterpillar-specific class and confidence
            if predicted_class == 'not_coconut':
                cat_class = 'not_coconut'
                cat_confidence = float(predictions[2])  # not_coconut probability
            elif is_caterpillar:
                cat_class = 'caterpillar'
                cat_confidence = float(predictions[0])  # caterpillar probability
            else:
                cat_class = 'healthy'
                cat_confidence = float(predictions[1])  # healthy probability

            # Determine white_fly-specific class and confidence
            if predicted_class == 'not_coconut':
                wf_class = 'not_coconut'
                wf_confidence = float(predictions[2])  # not_coconut probability
            elif is_white_fly:
                wf_class = 'white_fly'
                wf_confidence = float(predictions[3])  # white_fly probability
            else:
                wf_class = 'healthy'
                wf_confidence = float(predictions[1])  # healthy probability

            # Store as separate results for backward compatibility
            results['caterpillar'] = {
                'class': cat_class,
                'confidence': cat_confidence,
                'is_infected': is_caterpillar,
                'is_valid_image': is_valid,
                'probabilities': probabilities
            }

            results['white_fly'] = {
                'class': wf_class,
                'confidence': wf_confidence,
                'is_infected': is_white_fly,
                'is_valid_image': is_valid,
                'probabilities': probabilities
            }

            if is_caterpillar and is_valid:
                detected_pests.append('Caterpillar')
            if is_white_fly and is_valid:
                detected_pests.append('White Fly')

            # Cross-validation: If unified model confidently says "healthy",
            # remove mite detection (unified model is more reliable for healthy leaves)
            unified_healthy_confidence = float(predictions[1])  # healthy probability
            if unified_healthy_confidence > 0.80 and 'Coconut Mite' in detected_pests:
                # Unified model is confident this is healthy, don't trust mite detection
                detected_pests.remove('Coconut Mite')
                # Update mite result to show healthy instead
                if 'mite' in results:
                    results['mite']['class'] = 'healthy'
                    results['mite']['confidence'] = unified_healthy_confidence
                    results['mite']['is_infected'] = False

        except Exception as e:
            results['caterpillar'] = {'error': str(e)}
            results['white_fly'] = {'error': str(e)}

    # Smart Combined Decision Logic
    MIN_CONFIDENCE = 0.40
    valid_coconut_found = False

    # Check mite result
    if 'mite' in results and 'error' not in results['mite']:
        predicted_class = results['mite'].get('class', '')
        confidence = results['mite'].get('confidence', 0)
        if predicted_class in ['healthy', 'coconut_mite'] and confidence > MIN_CONFIDENCE:
            valid_coconut_found = True

    # Check unified result - caterpillar
    if 'caterpillar' in results and 'error' not in results['caterpillar']:
        predicted_class = results['caterpillar'].get('class', '')
        confidence = results['caterpillar'].get('confidence', 0)
        if predicted_class in ['healthy', 'caterpillar'] and confidence > MIN_CONFIDENCE:
            valid_coconut_found = True

    # Check unified result - white fly
    if 'white_fly' in results and 'error' not in results['white_fly']:
        predicted_class = results['white_fly'].get('class', '')
        confidence = results['white_fly'].get('confidence', 0)
        if predicted_class in ['healthy', 'white_fly'] and confidence > MIN_CONFIDENCE:
            valid_coconut_found = True

    should_reject = not valid_coconut_found

    if should_reject:
        summary = {
            'is_valid_image': False,
            'is_healthy': False,
            'pests_detected': [],
            'status': 'Invalid Image',
            'label': 'Not a valid coconut image',
            'message': 'The uploaded image does not appear to be a coconut. Please upload a clear image of a coconut fruit or leaf.',
            'recommendation': 'Please upload a clearer image of a coconut'
        }
    elif len(detected_pests) > 0:
        if len(detected_pests) >= 2:
            status = 'Multiple Pests Detected'
            label = f'{", ".join(detected_pests)} damage detected'
            message = f'WARNING: This coconut shows signs of multiple pest infections: {", ".join(detected_pests)}.'
            recommendation = 'Immediate treatment recommended. Apply comprehensive pest control measures.'
        elif 'Coconut Mite' in detected_pests:
            status = 'Mite Infection Detected'
            label = 'Coconut Mite Infected'
            message = 'This coconut shows signs of mite infection.'
            recommendation = 'Apply mite treatment spray and monitor affected trees.'
        elif 'Caterpillar' in detected_pests:
            status = 'Caterpillar Damage Detected'
            label = 'Caterpillar Damage Found'
            message = 'This coconut shows signs of caterpillar damage.'
            recommendation = 'Apply caterpillar control measures and inspect nearby trees.'
        elif 'White Fly' in detected_pests:
            status = 'White Fly Damage Detected'
            label = 'White Fly Infestation Found'
            message = 'This coconut shows signs of white fly infestation.'
            recommendation = 'Apply white fly control measures such as neem oil spray.'
        else:
            status = 'Pest Detected'
            label = detected_pests[0]
            message = f'This coconut shows signs of {detected_pests[0]} infection.'
            recommendation = 'Apply appropriate pest control measures.'

        summary = {
            'is_valid_image': True,
            'is_healthy': False,
            'pests_detected': detected_pests,
            'status': status,
            'label': label,
            'message': message,
            'recommendation': recommendation
        }
    else:
        summary = {
            'is_valid_image': True,
            'is_healthy': True,
            'pests_detected': [],
            'status': 'Healthy',
            'label': 'Healthy Coconut',
            'message': 'No pests detected. This coconut appears to be healthy.',
            'recommendation': 'Continue regular monitoring.'
        }

    return jsonify({
        'success': True,
        'results': results,
        'summary': summary,
        'models_used': {
            'mite': 'v10 (3-class, 91.44% accuracy)',
            'unified': 'v1 (4-class, 96.08% accuracy - caterpillar & white fly)'
        },
        'timestamp': datetime.now().isoformat()
    })

# Legacy endpoint
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
    load_models()

    print("\nStarting Coconut Health Monitor ML API v6.0...")
    print("  Mite Model: v10 (3-class, 91.44% accuracy)")
    print("  Unified Model: v1 (4-class - caterpillar + white_fly, 96.08% accuracy)")
    print("  Using unified model for better caterpillar/white_fly distinction!")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=False)
