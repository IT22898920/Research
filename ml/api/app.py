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

# Mite model paths (v10 - 3-class with Focal Loss)
MITE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v10', 'best_model.keras')
MITE_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v10', 'model_info.json')

# Mite v10 optimal threshold (from threshold tuning)
# Lower threshold = more sensitive to mites (catch more, but may have false positives)
MITE_THRESHOLD = 0.10
MITE_BOOST_FACTOR = 0.5 / MITE_THRESHOLD  # 5x boost for mite class

# Mite v10 class indices
MITE_CLASSES = ['coconut_mite', 'healthy', 'not_coconut']

# Minimum confidence threshold for valid predictions
MIN_CONFIDENCE_THRESHOLD = 0.50

# Caterpillar model paths (v2 - 3-class with Focal Loss)
CATERPILLAR_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_caterpillar_v2', 'best_model.keras')
CATERPILLAR_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_caterpillar_v2', 'model_info.json')

# Caterpillar v2 class indices
CATERPILLAR_CLASSES = ['caterpillar', 'healthy', 'not_coconut']

# Leaf Health model paths (v1 - 2-class)
LEAF_HEALTH_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v1', 'best_model.keras')
LEAF_HEALTH_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v1', 'model_info.json')

# Leaf Health v1 class indices
LEAF_HEALTH_CLASSES = ['healthy', 'unhealthy']

# Global variables for models
models = {}
model_infos = {}

def focal_loss(gamma=2.0, alpha=0.25):
    """Custom focal loss for loading v10 model"""
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

        # Load model with custom focal loss
        models['mite'] = tf.keras.models.load_model(
            MITE_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        # Try to load model info (may be incomplete due to JSON error during training)
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
        print(f"    Accuracy: 91.44% (with threshold tuning)")
        print(f"    Mite Recall: 79%")
        print(f"    Threshold: {MITE_THRESHOLD} (boost factor: {MITE_BOOST_FACTOR}x)")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading mite model: {e}")
        models['mite'] = None
        model_infos['mite'] = None

    # Load Caterpillar Model (v2 - 3-class with Focal Loss)
    try:
        print("\n[2] Loading Coconut Caterpillar model (v2 - 3-class)...")

        # Load model with custom focal loss
        models['caterpillar'] = tf.keras.models.load_model(
            CATERPILLAR_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        # Try to load model info
        try:
            with open(CATERPILLAR_MODEL_INFO_PATH, 'r') as f:
                model_infos['caterpillar'] = json.load(f)
        except:
            model_infos['caterpillar'] = {
                'version': 'v2_3class',
                'classes': CATERPILLAR_CLASSES,
                'performance': {'test_accuracy': 0.9747, 'caterpillar_recall': 0.9149}
            }

        print(f"    Version: v2 (3-class, Focal Loss)")
        print(f"    Classes: {CATERPILLAR_CLASSES}")
        print(f"    Accuracy: 97.47%")
        print(f"    Caterpillar Recall: 91.49%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading caterpillar model: {e}")
        models['caterpillar'] = None
        model_infos['caterpillar'] = None

    # Load Leaf Health Model (v1 - 2-class)
    try:
        print("\n[3] Loading Leaf Health model (v1 - 2-class)...")

        # Load model with custom focal loss
        models['leaf_health'] = tf.keras.models.load_model(
            LEAF_HEALTH_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        # Try to load model info
        try:
            with open(LEAF_HEALTH_MODEL_INFO_PATH, 'r') as f:
                model_infos['leaf_health'] = json.load(f)
        except:
            model_infos['leaf_health'] = {
                'version': 'v1_2class',
                'classes': LEAF_HEALTH_CLASSES,
                'performance': {'test_accuracy': 0.9370}
            }

        print(f"    Version: v1 (2-class, Focal Loss)")
        print(f"    Classes: {LEAF_HEALTH_CLASSES}")
        print(f"    Accuracy: 93.70%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading leaf health model: {e}")
        models['leaf_health'] = None
        model_infos['leaf_health'] = None

    print("\n" + "=" * 60)
    loaded_count = sum(1 for m in models.values() if m is not None)
    print(f"  Models loaded: {loaded_count}/3")
    print("=" * 60)

def preprocess_image_mite(image_bytes):
    """Preprocess image for mite model v10 (0-1 scaling)"""
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # v10 uses 224x224 input
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    # v10 uses simple 0-1 normalization (same as training)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)

def preprocess_image_caterpillar(image_bytes):
    """Preprocess image for caterpillar model v2 (0-1 scaling, 224x224)"""
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # v2 uses 224x224 input (same as mite model)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    # Simple 0-1 normalization (same as training)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Coconut Health Monitor - Pest Detection API',
        'version': '4.0.0',
        'models': {
            'mite': {
                'status': 'loaded' if models.get('mite') is not None else 'not loaded',
                'version': 'v10 (3-class, Focal Loss)',
                'accuracy': '91.44%'
            },
            'caterpillar': {
                'status': 'loaded' if models.get('caterpillar') is not None else 'not loaded',
                'version': 'v2 (3-class, Focal Loss)',
                'accuracy': '97.47%'
            }
        },
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/models': 'List all available models',
            '/predict/mite': 'POST - Detect coconut mite infection (3-class)',
            '/predict/caterpillar': 'POST - Detect caterpillar damage (3-class)',
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
            'caterpillar': models.get('caterpillar') is not None
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

    if model_infos.get('caterpillar'):
        result['caterpillar'] = {
            'name': 'Coconut Caterpillar Detection Model',
            'version': 'v2 (3-class, Focal Loss)',
            'classes': CATERPILLAR_CLASSES,
            'accuracy': 0.9747,
            'caterpillar_recall': 0.9149,
            'loaded': models.get('caterpillar') is not None
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
        # Classes: ['coconut_mite', 'healthy', 'not_coconut']
        predictions = models['mite'].predict(processed_image, verbose=0)[0]

        # Apply threshold adjustment (boost mite probability)
        adjusted_probs = predictions.copy()
        mite_idx = 0  # coconut_mite index
        adjusted_probs[mite_idx] = adjusted_probs[mite_idx] * MITE_BOOST_FACTOR

        # Get predicted class
        predicted_idx = int(np.argmax(adjusted_probs))
        predicted_class = MITE_CLASSES[predicted_idx]

        # Get confidence (from original probabilities)
        confidence = float(predictions[predicted_idx])
        is_mite = predicted_class == 'coconut_mite'
        is_valid = predicted_class != 'not_coconut'

        # Build response
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
                    'message': 'The uploaded image does not appear to be a coconut. Please upload a clear image of a coconut fruit or leaf.',
                    'threshold_used': MITE_THRESHOLD,
                    'boost_factor': MITE_BOOST_FACTOR
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
                'label': 'Coconut Mite Infected' if is_mite else 'Healthy',
                'threshold_used': MITE_THRESHOLD,
                'boost_factor': MITE_BOOST_FACTOR
            },
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/caterpillar', methods=['POST'])
def predict_caterpillar():
    """Detect caterpillar damage (v2 model - 3-class classification)"""

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

        # v2 3-class classification: softmax output
        # Classes: ['caterpillar', 'healthy', 'not_coconut']
        predictions = models['caterpillar'].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CATERPILLAR_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_caterpillar = predicted_class == 'caterpillar'
        is_valid = predicted_class != 'not_coconut'

        # Build response
        probabilities = {
            'caterpillar': float(predictions[0]),
            'healthy': float(predictions[1]),
            'not_coconut': float(predictions[2])
        }

        if not is_valid:
            return jsonify({
                'success': True,
                'pest_type': 'caterpillar',
                'model_version': 'v2',
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
            'model_version': 'v2',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_caterpillar,
                'is_valid_image': True,
                'label': 'Caterpillar Damage Detected' if is_caterpillar else 'Healthy'
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
    Smart Combined Logic:
    1. If EITHER model says "not_coconut" with high confidence → reject as not coconut
    2. Check each pest independently
    3. Return ONE coherent answer with all detections
    """

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    image_bytes = file.read()
    results = {}
    detected_pests = []

    # Track not_coconut confidence from each model
    not_coconut_confidences = []

    # Run Mite Detection (v10 - 3-class classification)
    if models.get('mite') is not None:
        try:
            processed = preprocess_image_mite(image_bytes)
            predictions = models['mite'].predict(processed, verbose=0)[0]

            # Apply threshold adjustment (boost mite probability)
            adjusted_probs = predictions.copy()
            adjusted_probs[0] = adjusted_probs[0] * MITE_BOOST_FACTOR

            # Get predicted class
            predicted_idx = int(np.argmax(adjusted_probs))
            predicted_class = MITE_CLASSES[predicted_idx]
            confidence = float(predictions[predicted_idx])
            is_mite = predicted_class == 'coconut_mite'
            is_valid = predicted_class != 'not_coconut'

            # Track not_coconut probability
            not_coconut_prob = float(predictions[2])  # not_coconut is index 2
            not_coconut_confidences.append(('mite', not_coconut_prob))

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

    # Run Caterpillar Detection (v2 - 3-class classification)
    if models.get('caterpillar') is not None:
        try:
            processed = preprocess_image_caterpillar(image_bytes)
            predictions = models['caterpillar'].predict(processed, verbose=0)[0]

            # Get predicted class
            predicted_idx = int(np.argmax(predictions))
            predicted_class = CATERPILLAR_CLASSES[predicted_idx]
            confidence = float(predictions[predicted_idx])
            is_caterpillar = predicted_class == 'caterpillar'
            is_valid = predicted_class != 'not_coconut'

            # Track not_coconut probability
            not_coconut_prob = float(predictions[2])  # not_coconut is index 2
            not_coconut_confidences.append(('caterpillar', not_coconut_prob))

            results['caterpillar'] = {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_caterpillar,
                'is_valid_image': is_valid,
                'probabilities': {
                    'caterpillar': float(predictions[0]),
                    'healthy': float(predictions[1]),
                    'not_coconut': float(predictions[2])
                }
            }
            if is_caterpillar and is_valid:
                detected_pests.append('Caterpillar')
        except Exception as e:
            results['caterpillar'] = {'error': str(e)}

    # Smart Combined Decision Logic (v7 - Confidence Threshold)
    # Valid coconut detection requires >40% confidence in valid class
    # Only reject if no confident valid detection found

    # Check if any model found a valid coconut with >40% confidence
    valid_coconut_found = False
    MIN_CONFIDENCE = 0.40  # Minimum confidence to trust a valid detection

    for model_name in ['mite', 'caterpillar']:
        if model_name in results and 'error' not in results[model_name]:
            predicted_class = results[model_name].get('class', '')
            confidence = results[model_name].get('confidence', 0)
            # If model predicts healthy or a pest WITH >40% confidence, it's valid
            if predicted_class in ['healthy', 'coconut_mite', 'caterpillar'] and confidence > MIN_CONFIDENCE:
                valid_coconut_found = True
                break

    # Reject if no confident valid detection found
    should_reject = not valid_coconut_found

    if should_reject:
        # Not a valid coconut image
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
        # Pests detected
        if len(detected_pests) == 2:
            status = 'Multiple Pests Detected'
            label = 'Both Mite and Caterpillar damage detected'
            message = 'WARNING: This coconut shows signs of both mite infection and caterpillar damage.'
            recommendation = 'Immediate treatment recommended. Consider both mite spray and caterpillar control measures.'
        elif 'Coconut Mite' in detected_pests:
            status = 'Mite Infection Detected'
            label = 'Coconut Mite Infected'
            message = 'This coconut shows signs of mite infection.'
            recommendation = 'Apply mite treatment spray and monitor affected trees.'
        else:
            status = 'Caterpillar Damage Detected'
            label = 'Caterpillar Damage Found'
            message = 'This coconut shows signs of caterpillar damage.'
            recommendation = 'Apply caterpillar control measures and inspect nearby trees.'

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
        # Healthy coconut
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
            'caterpillar': 'v2 (3-class, 97.47% accuracy)'
        },
        'timestamp': datetime.now().isoformat()
    })

# Leaf Health Detection Endpoint
@app.route('/predict/leaf-health', methods=['POST'])
def predict_leaf_health():
    """
    Predict if a coconut leaf is healthy or unhealthy (yellowing)

    Returns:
        JSON with prediction results
    """
    if models['leaf_health'] is None:
        return jsonify({'error': 'Leaf health model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        # Read image
        image_file = request.files['image']
        image_bytes = image_file.read()

        # Preprocess image (same as mite model - 224x224, 0-1 scaling)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = models['leaf_health'].predict(img_array, verbose=0)

        # Get class probabilities
        healthy_prob = float(predictions[0][0])
        unhealthy_prob = float(predictions[0][1])

        # Determine predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = LEAF_HEALTH_CLASSES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))

        # Prepare response
        result = {
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'healthy': healthy_prob,
                'unhealthy': unhealthy_prob
            },
            'is_healthy': predicted_class == 'healthy',
            'message': get_leaf_health_message(predicted_class, confidence),
            'recommendation': get_leaf_health_recommendation(predicted_class),
            'model_info': {
                'version': 'v1',
                'classes': LEAF_HEALTH_CLASSES,
                'accuracy': '93.70%'
            },
            'timestamp': datetime.now().isoformat()
        }

        # Add detailed conditions if unhealthy
        if predicted_class == 'unhealthy':
            result['possible_conditions'] = get_unhealthy_conditions_details()
            result['conditions_count'] = len(UNHEALTHY_LEAF_CONDITIONS)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Knowledge base for unhealthy leaf conditions
UNHEALTHY_LEAF_CONDITIONS = [
    {
        'condition': 'Nitrogen Deficiency',
        'reason': 'Lack of nitrogen in soil causes yellowing of older leaves first. Nitrogen is essential for chlorophyll production.',
        'symptoms': [
            'Yellowing starts from older leaves',
            'Stunted growth',
            'Pale green to yellow color'
        ],
        'solution': 'Apply nitrogen-rich fertilizers such as urea (46-0-0) or ammonium sulfate. Use 200-250g per tree, split into 2-3 applications. Alternatively, apply organic matter like compost or well-rotted manure.',
        'urgency': 'medium',
        'icon': '🍂'
    },
    {
        'condition': 'Potassium Deficiency',
        'reason': 'Potassium deficiency causes yellowing and browning of leaf tips and margins. Essential for water regulation and disease resistance.',
        'symptoms': [
            'Yellow/brown leaf tips',
            'Marginal leaf burn',
            'Weak stems'
        ],
        'solution': 'Apply potassium-rich fertilizers like muriate of potash (KCl 60%) at 250-300g per tree. Ensure proper irrigation to help potassium uptake.',
        'urgency': 'medium',
        'icon': '🔥'
    },
    {
        'condition': 'Magnesium Deficiency',
        'reason': 'Magnesium is crucial for photosynthesis. Deficiency causes interveinal chlorosis (yellowing between leaf veins).',
        'symptoms': [
            'Yellow areas between green veins',
            'Older leaves affected first',
            'Orange/red tints may appear'
        ],
        'solution': 'Apply magnesium sulfate (Epsom salt) as foliar spray: 20g/liter of water, spray every 2 weeks for 3 months. Or apply dolomite lime to soil.',
        'urgency': 'medium',
        'icon': '🌿'
    },
    {
        'condition': 'Water Stress (Under-watering)',
        'reason': 'Insufficient water causes leaves to yellow and dry out. Coconut palms need consistent moisture, especially during dry periods.',
        'symptoms': [
            'Overall yellowing',
            'Dry, brittle leaves',
            'Leaf tips turn brown'
        ],
        'solution': 'Increase irrigation frequency. Coconut trees need 50-100 liters of water per week during dry season. Mulch around base to retain moisture.',
        'urgency': 'high',
        'icon': '💧'
    },
    {
        'condition': 'Water Stress (Over-watering)',
        'reason': 'Excessive water causes root rot and prevents oxygen uptake, leading to yellowing leaves.',
        'symptoms': [
            'Yellowing with wilting',
            'Soggy soil',
            'Root rot smell'
        ],
        'solution': 'Improve drainage around the tree. Reduce watering frequency. Consider raised beds if soil stays waterlogged. Ensure proper drainage channels.',
        'urgency': 'high',
        'icon': '🌊'
    },
    {
        'condition': 'Root Disease',
        'reason': 'Fungal infections in roots (like Ganoderma or Phytophthora) prevent nutrient uptake, causing yellowing.',
        'symptoms': [
            'Progressive yellowing from bottom up',
            'Wilting despite watering',
            'Stunted growth'
        ],
        'solution': 'Remove infected roots if possible. Apply fungicides like copper oxychloride. Improve drainage. Infected severe cases may need tree removal to prevent spread.',
        'urgency': 'high',
        'icon': '🦠'
    },
    {
        'condition': 'Iron Deficiency (Chlorosis)',
        'reason': 'Iron deficiency causes yellowing of young leaves while veins remain green. Common in alkaline soils.',
        'symptoms': [
            'Young leaves turn yellow',
            'Green veins pattern',
            'Reduced growth'
        ],
        'solution': 'Apply chelated iron (Fe-EDTA) as foliar spray or soil drench. Reduce soil pH if too alkaline by adding sulfur or acidic organic matter.',
        'urgency': 'medium',
        'icon': '⚗️'
    },
    {
        'condition': 'Pest Damage',
        'reason': 'Pest infestations (mites, caterpillars, scale insects) damage leaf tissue and suck nutrients, causing yellowing.',
        'symptoms': [
            'Spotted yellowing',
            'Visible pests or webs',
            'Damaged leaf tissue'
        ],
        'solution': 'Identify specific pest and treat accordingly. Use neem oil spray (5ml/liter water) for general pest control. For severe infestations, use appropriate pesticides.',
        'urgency': 'high',
        'icon': '🐛'
    },
    {
        'condition': 'Natural Aging',
        'reason': 'Older leaves naturally yellow and die as the tree redirects nutrients to new growth. This is normal.',
        'symptoms': [
            'Only oldest (bottom) leaves yellow',
            'New growth is healthy green',
            'No other symptoms'
        ],
        'solution': 'No action needed. Remove yellowed fronds once completely dry. This is part of natural leaf cycle. Ensure tree gets balanced fertilization.',
        'urgency': 'low',
        'icon': '🍃'
    }
]

def get_leaf_health_message(predicted_class, confidence):
    """Get message based on leaf health prediction"""
    if predicted_class == 'healthy':
        if confidence > 0.95:
            return "Leaf appears to be very healthy!"
        elif confidence > 0.80:
            return "Leaf appears to be healthy."
        else:
            return "Leaf seems healthy but with lower confidence."
    else:  # unhealthy
        if confidence > 0.80:
            return "Leaf shows signs of yellowing/unhealthy condition. Multiple possible causes detected."
        else:
            return "Possible yellowing detected. Review the possible causes below."

def get_leaf_health_recommendation(predicted_class):
    """Get recommendation based on leaf health"""
    if predicted_class == 'healthy':
        return "Continue regular monitoring and maintain good care practices."
    else:  # unhealthy
        return "Review the detailed analysis below to identify the specific cause and apply the recommended treatment."

def get_unhealthy_conditions_details():
    """Get detailed information about all possible unhealthy conditions"""
    return UNHEALTHY_LEAF_CONDITIONS

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

    # Run the Flask app on port 5001 (port 5000 is used by Node.js auth backend)
    print("\nStarting Coconut Health Monitor ML API v5.0...")
    print("  Mite Model: v10 (3-class, 91.44% accuracy, 79% mite recall)")
    print("  Caterpillar Model: v2 (3-class, 97.47% accuracy, 91.49% caterpillar recall)")
    print("  Leaf Health Model: v1 (2-class, 93.70% accuracy)")
    print("  Mite & Caterpillar models support 'not_coconut' class for image validation!")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=True)
