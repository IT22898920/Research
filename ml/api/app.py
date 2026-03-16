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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import tensorflow as tf
from datetime import datetime

# Create a requests session with retry logic for Groq API
def create_groq_session():
    """Create a requests session with retry logic for transient failures"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=0.5,  # Wait 0.5, 1.0, 2.0 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["POST", "GET"],  # Allow retries on POST
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Global session for Groq API calls
groq_session = create_groq_session()

# Groq API Configuration for AI Chatbot
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Coconut Health Expert System Prompt
COCONUT_EXPERT_PROMPT = """You are a friendly and knowledgeable Coconut Health Expert Assistant for Sri Lankan farmers. Your role is to help farmers and agricultural workers with:

1. **Pest Identification & Treatment**: Coconut mite (පොල් මයිටාව/தென்னை பேன்), black-headed caterpillar (කළු හිස් දළඹුවා/கருந்தலை புழு), white fly (සුදු මැස්සා/வெள்ளை ஈ), rhinoceros beetle, red palm weevil, etc.
2. **Disease Management**: Bud rot, leaf rot, stem bleeding, root wilt, etc.
3. **Farming Best Practices**: Irrigation, fertilization, harvesting techniques
4. **General Coconut Care**: Plant health, nutrition, growth stages

CRITICAL LANGUAGE RULES - FOLLOW EXACTLY:

1. **Sinhala Unicode Script (සිංහල)**: If user writes in Sinhala script like "මට පොල් මයිටාව ගැන දැනගන්න ඕනේ", respond FULLY in Sinhala Unicode script.
   Example response: "ආයුබෝවන්! පොල් මයිටාව පාලනය කිරීමට නීම් තෙල්, සබන් ද්‍රාවණය හෝ වෙනත් කෘමිනාශක භාවිතා කළ හැක."

2. **Romanized Sinhala (Singlish)**: If user writes Sinhala words using English letters like "coconut mite eka gena denaganna one mata" or "pol gaha", respond FULLY in Sinhala Unicode script (NOT romanized).
   Example: User says "mite treatment eka mokakda" → Respond in proper Sinhala: "පොල් මයිටාව සඳහා නීම් තෙල් භාවිතා කරන්න..."

3. **Tamil Script (தமிழ்)**: If user writes in Tamil script, respond FULLY in Tamil Unicode script.
   Example response: "வணக்கம்! தென்னை பேன் சிகிச்சைக்கு வேப்பெண்ணெய் பயன்படுத்தலாம்."

4. **English**: If user writes in English, respond in English.

NEVER MIX LANGUAGES IN A SINGLE RESPONSE!
- Do NOT write half Sinhala and half English
- Do NOT write romanized Sinhala (like "oyata harima purudu")
- Always use proper Unicode script for Sinhala/Tamil responses
- Keep the ENTIRE response in ONE language

Guidelines:
- Give clear, practical advice that farmers can follow
- Use simple, everyday language that farmers understand
- Provide step-by-step treatment instructions when asked
- If asked about non-coconut topics, politely redirect to coconut-related help
- Keep responses concise but helpful (max 200 words unless detailed explanation needed)
- Always be encouraging and supportive to farmers
- Include both scientific and local names for pests when helpful

Common Sinhala terms: පොහොර (fertilizer), වතුර දැමීම (watering), කොළ (leaves), ගෙඩි (fruits), පළිබෝධ (pests), මයිටාව (mite), දළඹුවා (caterpillar), සුදු මැස්සා (white fly)
Common Tamil terms: உரம் (fertilizer), நீர்ப்பாசனம் (irrigation), இலைகள் (leaves), பழங்கள் (fruits), பூச்சிகள் (pests)

Remember: You're helping real Sri Lankan farmers protect their coconut trees and livelihoods!"""

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

# Disease Detection model paths (v2 - 4-class with Focal Loss)
DISEASE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'disease_detection_v2', 'best_model.keras')
DISEASE_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'disease_detection_v2', 'model_info.json')

# Disease model class indices (alphabetical order from ImageDataGenerator)
DISEASE_CLASSES = ['Leaf Rot', 'Leaf_Spot', 'healthy', 'not_cocount']

# Leaf Health model paths (v4 - MobileNetV2, 95.00% accuracy on new dataset)
LEAF_HEALTH_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v4', 'best_model.keras')
LEAF_HEALTH_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v4', 'model_info.json')

# Leaf Health v1 class indices
LEAF_HEALTH_CLASSES = ['healthy', 'unhealthy']

# Leaf Health v4 uses MobileNetV2 preprocess_input (no temperature scaling needed)

# Branch Health model paths (v1 - 2-class)
BRANCH_HEALTH_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_branch_health_v1', 'best_model.keras')
BRANCH_HEALTH_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_branch_health_v1', 'model_info.json')

# Branch Health v1 class indices
BRANCH_HEALTH_CLASSES = ['healthy', 'unhealthy']

# Tree Health model paths (v2 - EfficientNetB0, oversampled healthy class)
TREE_HEALTH_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_tree_health_v2', 'best_model.keras')
TREE_HEALTH_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_tree_health_v2', 'model_info.json')

# Tree Health v1 class indices
TREE_HEALTH_CLASSES = ['healthy', 'unhealthy']

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

def apply_temperature_scaling(probs, temperature):
    """
    Apply temperature scaling to calibrate overconfident model predictions.
    Formula: calibrated_p_i = p_i^(1/T) / sum(p_j^(1/T))
    Higher T = less confident predictions (more realistic for real-world use).
    """
    epsilon = 1e-10
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    scaled = np.power(probs, 1.0 / temperature)
    return scaled / np.sum(scaled)

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

    # Load Disease Detection Model (v2 - 4-class with Focal Loss)
    try:
        print("\n[3] Loading Disease Detection model (v2 - 4-class)...")

        models['disease'] = tf.keras.models.load_model(
            DISEASE_MODEL_PATH,
            custom_objects={'FocalLoss': focal_loss(gamma=2.0, alpha=0.25)}
        )

        try:
            with open(DISEASE_MODEL_INFO_PATH, 'r') as f:
                model_infos['disease'] = json.load(f)
        except:
            model_infos['disease'] = {
                'version': 'v2_4class',
                'classes': DISEASE_CLASSES,
                'performance': {
                    'test_accuracy': 0.9869,
                    'macro_f1': 0.9800
                }
            }

        print(f"    Version: v2 (4-class, Focal Loss)")
        print(f"    Classes: {DISEASE_CLASSES}")
        print(f"    Accuracy: 98.69%")
        print(f"    Macro F1: 98.00%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading disease model: {e}")
        models['disease'] = None
        model_infos['disease'] = None

    # Load Leaf Health Model (v4 - MobileNetV2, 95.00% accuracy on new dataset)
    try:
        print("\n[4] Loading Leaf Health model (v4 - 2-class)...")

        models['leaf_health'] = tf.keras.models.load_model(
            LEAF_HEALTH_MODEL_PATH
        )

        try:
            with open(LEAF_HEALTH_MODEL_INFO_PATH, 'r') as f:
                model_infos['leaf_health'] = json.load(f)
        except:
            model_infos['leaf_health'] = {
                'version': 'v4_2class',
                'classes': LEAF_HEALTH_CLASSES,
                'performance': {'test_accuracy': 0.9500, 'macro_f1': 0.9488}
            }

        print(f"    Version: v4 (2-class, MobileNetV2)")
        print(f"    Classes: {LEAF_HEALTH_CLASSES}")
        print(f"    Accuracy: 95.00%")
        print(f"    Macro F1: 94.88%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading leaf health model: {e}")
        models['leaf_health'] = None
        model_infos['leaf_health'] = None

    # Load Branch Health Model (v1 - 2-class)
    try:
        print("\n[5] Loading Branch Health model (v1 - 2-class)...")

        # Load model with custom focal loss
        models['branch_health'] = tf.keras.models.load_model(
            BRANCH_HEALTH_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        # Try to load model info
        try:
            with open(BRANCH_HEALTH_MODEL_INFO_PATH, 'r') as f:
                model_infos['branch_health'] = json.load(f)
        except:
            model_infos['branch_health'] = {
                'version': 'v1_2class',
                'classes': BRANCH_HEALTH_CLASSES,
                'performance': {'test_accuracy': 0.9963}
            }

        print(f"    Version: v1 (2-class, Focal Loss)")
        print(f"    Classes: {BRANCH_HEALTH_CLASSES}")
        print(f"    Accuracy: 99.63%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading branch health model: {e}")
        models['branch_health'] = None
        model_infos['branch_health'] = None

    # Load Tree Health Model (v2 - EfficientNetB0, oversampled healthy class)
    try:
        print("\n[6] Loading Tree Health model (v2 - 2-class)...")

        models['tree_health'] = tf.keras.models.load_model(
            TREE_HEALTH_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=3.0, alpha=0.25)}
        )

        try:
            with open(TREE_HEALTH_MODEL_INFO_PATH, 'r') as f:
                model_infos['tree_health'] = json.load(f)
        except:
            model_infos['tree_health'] = {
                'version': 'v2_2class',
                'classes': TREE_HEALTH_CLASSES,
                'performance': {'test_accuracy': 1.0, 'macro_f1': 1.0}
            }

        print(f"    Version: v2 (2-class, EfficientNetB0, Focal Loss + Oversampling)")
        print(f"    Classes: {TREE_HEALTH_CLASSES}")
        print(f"    Accuracy: 100%")
        print(f"    Macro F1: 100%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading tree health model: {e}")
        models['tree_health'] = None
        model_infos['tree_health'] = None

    print("\n" + "=" * 60)
    loaded_count = sum(1 for m in models.values() if m is not None)
    print(f"  Models loaded: {loaded_count}/6")
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

def preprocess_image_disease(image_bytes):
    """Preprocess image for disease model (0-1 scaling, 224x224)"""
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
        'service': 'Coconut Health Monitor - Pest & Disease Detection API',
        'version': '8.0.0',
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
            },
            'disease': {
                'status': 'loaded' if models.get('disease') is not None else 'not loaded',
                'version': 'v2 (4-class: Leaf Rot, Leaf_Spot, healthy, not_cocount)',
                'accuracy': '98.69%'
            },
            'leaf_health': {
                'status': 'loaded' if models.get('leaf_health') is not None else 'not loaded',
                'version': 'v4 (MobileNetV2, 2-class: healthy, unhealthy)',
                'accuracy': '95.00%'
            },
            'branch_health': {
                'status': 'loaded' if models.get('branch_health') is not None else 'not loaded',
                'version': 'v1 (2-class: healthy, unhealthy)',
                'accuracy': '99.63%'
            },
            'tree_health': {
                'status': 'loaded' if models.get('tree_health') is not None else 'not loaded',
                'version': 'v2 (EfficientNetB0, oversampled healthy class)',
                'accuracy': '100%'
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
            '/predict/disease': 'POST - Detect leaf diseases (Leaf Rot, Leaf Spot)',
            '/predict/leaf-health': 'POST - Detect leaf health (healthy vs unhealthy/yellowing)',
            '/predict/branch-health': 'POST - Detect branch health (healthy vs unhealthy)',
            '/predict/tree-health': 'POST - Detect tree health (healthy vs unhealthy)',
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
            'unified': models.get('unified') is not None,
            'disease': models.get('disease') is not None,
            'leaf_health': models.get('leaf_health') is not None,
            'branch_health': models.get('branch_health') is not None,
            'tree_health': models.get('tree_health') is not None
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

    if model_infos.get('disease'):
        result['disease'] = {
            'name': 'Coconut Leaf Disease Detection Model',
            'version': 'v2 (4-class, Focal Loss)',
            'classes': DISEASE_CLASSES,
            'accuracy': 0.9869,
            'macro_f1': 0.9800,
            'loaded': models.get('disease') is not None
        }

    if model_infos.get('leaf_health'):
        result['leaf_health'] = {
            'name': 'Coconut Leaf Health Detection Model',
            'version': 'v4 (2-class, MobileNetV2)',
            'classes': LEAF_HEALTH_CLASSES,
            'accuracy': 0.9500,
            'macro_f1': 0.9488,
            'loaded': models.get('leaf_health') is not None
        }

    if model_infos.get('branch_health'):
        result['branch_health'] = {
            'name': 'Coconut Branch Health Detection Model',
            'version': 'v1 (2-class, Focal Loss)',
            'classes': BRANCH_HEALTH_CLASSES,
            'accuracy': 0.9963,
            'loaded': models.get('branch_health') is not None
        }

    if model_infos.get('tree_health'):
        result['tree_health'] = {
            'name': 'Coconut Tree Health Detection Model',
            'version': 'v2 (2-class, EfficientNetB0, Focal Loss + Oversampling)',
            'classes': TREE_HEALTH_CLASSES,
            'accuracy': 1.0,
            'macro_f1': 1.0,
            'loaded': models.get('tree_health') is not None
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

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    """Detect coconut leaf diseases (v2 model - 4-class classification)

    Classes: Leaf Rot, Leaf_Spot, healthy, not_cocount
    """

    if models.get('disease') is None:
        return jsonify({'error': 'Disease model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_disease(image_bytes)

        # 4-class classification: softmax output
        # Classes: ['Leaf Rot', 'Leaf_Spot', 'healthy', 'not_cocount']
        predictions = models['disease'].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = DISEASE_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_leaf_rot = predicted_class == 'Leaf Rot'
        is_leaf_spot = predicted_class == 'Leaf_Spot'
        is_diseased = is_leaf_rot or is_leaf_spot
        is_healthy = predicted_class == 'healthy'
        is_valid = predicted_class != 'not_cocount'

        probabilities = {
            'leaf_rot': float(predictions[0]),
            'leaf_spot': float(predictions[1]),
            'healthy': float(predictions[2]),
            'not_coconut': float(predictions[3])
        }

        # Determine label and message
        if not is_valid:
            label = 'Not a valid coconut leaf image'
            message = 'The uploaded image does not appear to be a coconut leaf. Please upload a clear image of a coconut leaf.'
            status = 'invalid'
        elif is_leaf_rot:
            label = 'Leaf Rot Disease Detected'
            message = 'This coconut leaf shows signs of Leaf Rot disease. Early treatment is recommended.'
            status = 'diseased'
        elif is_leaf_spot:
            label = 'Leaf Spot Disease Detected'
            message = 'This coconut leaf shows signs of Leaf Spot disease. Apply appropriate fungicide treatment.'
            status = 'diseased'
        else:
            label = 'Healthy Leaf'
            message = 'No disease detected. This coconut leaf appears to be healthy.'
            status = 'healthy'

        return jsonify({
            'success': True,
            'detection_type': 'disease',
            'model_version': 'v2',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_diseased': is_diseased,
                'is_leaf_rot': is_leaf_rot,
                'is_leaf_spot': is_leaf_spot,
                'is_healthy': is_healthy,
                'is_valid_image': is_valid,
                'label': label,
                'message': message,
                'status': status
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

        # Preprocess image for MobileNetV2 (v4): resize + preprocess_input -> [-1, 1]
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # ── Color Health Score (HSV-based green pixel ratio) ──────────────────
        img_norm = img_array / 255.0
        r, g, b = img_norm[..., 0], img_norm[..., 1], img_norm[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        # Hue (0-180 OpenCV scale)
        h = np.zeros_like(r)
        m = delta > 0
        mr = m & (cmax == r); mg = m & (cmax == g); mb = m & (cmax == b)
        h[mr] = 60 * (((g[mr] - b[mr]) / delta[mr]) % 6)
        h[mg] = 60 * ((b[mg] - r[mg]) / delta[mg] + 2)
        h[mb] = 60 * ((r[mb] - g[mb]) / delta[mb] + 4)
        h = h / 2.0  # 0-180
        s = np.where(cmax > 0, delta / cmax, 0) * 255
        sat_mask = s > 40
        green_mask = sat_mask & (h >= 35) & (h <= 85)
        brown_mask = sat_mask & ((h < 20) | (h > 160)) & (r > 0.3)  # reddish-brown hues
        total_sat = np.sum(sat_mask)
        green_ratio = float(np.sum(green_mask) / total_sat) * 100 if total_sat > 0 else 0
        brown_ratio = float(np.sum(brown_mask) / total_sat) * 100 if total_sat > 0 else 0

        # ── Model Prediction (MobileNetV2 preprocess_input: [0,255] -> [-1,1]) ─
        img_mobilenet = img_array.copy()
        img_mobilenet = (img_mobilenet / 127.5) - 1.0   # preprocess_input equivalent
        img_input = np.expand_dims(img_mobilenet, axis=0)
        predictions = models['leaf_health'].predict(img_input, verbose=0)

        # Get class probabilities
        healthy_prob = float(predictions[0][0])
        unhealthy_prob = float(predictions[0][1])

        # Determine predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = LEAF_HEALTH_CLASSES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))

        # ── Color Override: brown/dried leaves the model may miss ─────────────
        # If green pixels very low AND brown/orange pixels dominant → force unhealthy
        color_override = False
        if predicted_class == 'healthy' and green_ratio < 15.0 and brown_ratio > 20.0:
            predicted_class = 'unhealthy'
            confidence = max(0.82, brown_ratio / 100.0)
            healthy_prob = 1.0 - confidence
            unhealthy_prob = confidence
            color_override = True
            print(f"[LeafHealth] Color override: green={green_ratio:.1f}% brown={brown_ratio:.1f}% -> unhealthy")

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
                'version': 'v4',
                'classes': LEAF_HEALTH_CLASSES,
                'accuracy': '95.00%',
                'macro_f1': '94.88%',
                'color_override': color_override
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

# Branch Health Detection Endpoint
@app.route('/predict/branch-health', methods=['POST'])
def predict_branch_health():
    """
    Predict if a coconut tree branch is healthy or unhealthy

    Returns:
        JSON with prediction results
    """
    if models.get('branch_health') is None:
        return jsonify({'error': 'Branch health model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        # Read image
        image_file = request.files['image']
        image_bytes = image_file.read()

        # Preprocess image (same as leaf model - 224x224, 0-1 scaling)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = models['branch_health'].predict(img_array, verbose=0)

        # Get class probabilities
        healthy_prob = float(predictions[0][0])
        unhealthy_prob = float(predictions[0][1])

        # Determine predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = BRANCH_HEALTH_CLASSES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))

        # Calculate unhealthy percentage (if unhealthy)
        unhealthy_percentage = int(unhealthy_prob * 100) if predicted_class == 'unhealthy' else 0

        # Prepare message
        if predicted_class == 'healthy':
            if confidence > 0.95:
                message = "Branch appears to be very healthy!"
            elif confidence > 0.80:
                message = "Branch appears to be healthy."
            else:
                message = "Branch seems healthy but with lower confidence."
            recommendation = "Continue regular monitoring and maintain good care practices."
        else:  # unhealthy
            if confidence > 0.80:
                message = f"Branch shows signs of being unhealthy ({unhealthy_percentage}% unhealthy)."
            else:
                message = f"Possible unhealthy condition detected ({unhealthy_percentage}% unhealthy)."
            recommendation = "Inspect the branch for pest damage, disease, or nutrient deficiencies. Consider pruning if severely damaged."

        # Prepare response
        result = {
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'healthy': healthy_prob,
                'unhealthy': unhealthy_prob
            },
            'unhealthy_percentage': unhealthy_percentage,
            'is_healthy': predicted_class == 'healthy',
            'message': message,
            'recommendation': recommendation,
            'model_info': {
                'version': 'v1',
                'classes': BRANCH_HEALTH_CLASSES,
                'accuracy': '99.63%'
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Tree Health Detection Endpoint
def get_tree_health_meditations(unhealthy_percentage):
    """
    Returns severity-based meditations and solutions for unhealthy coconut trees.
    Severity is determined by the unhealthy_percentage from the model.
    """
    if unhealthy_percentage <= 30:
        severity = "Mild"
        urgency = "Low"
        urgency_color = "#FFC107"
        severity_description = "Early signs of stress detected. The tree is showing minor health issues that can be managed with timely intervention."
        possible_conditions = [
            {
                "name": "Early Nutrient Deficiency",
                "icon": "🌿",
                "reason": "Low availability of Nitrogen, Potassium, or Magnesium in the soil disrupts chlorophyll production and overall tree metabolism.",
                "symptoms": [
                    "Slight yellowing on older or lower leaves",
                    "Pale green or dull leaf color",
                    "Marginally reduced new growth"
                ],
                "solution": "Apply a balanced NPK fertilizer (14-14-14) at 500g per tree. Re-test soil pH (ideal: 5.5–7.0) and correct with lime or sulfur as needed.",
                "urgency": "low"
            },
            {
                "name": "Minor Pest Activity",
                "icon": "🐛",
                "reason": "Low populations of coconut mites, white flies, or caterpillars begin feeding on leaf surfaces, causing early stress.",
                "symptoms": [
                    "Small brown or silvery spots on leaf surfaces",
                    "Minor leaf edge damage or curling",
                    "Tiny insects visible on leaf undersides"
                ],
                "solution": "Spray neem oil solution (2ml/L water) on affected leaves. Inspect weekly and apply systemic insecticide if population grows.",
                "urgency": "low"
            },
            {
                "name": "Mild Water Stress",
                "icon": "💧",
                "reason": "Inconsistent watering or slight drought conditions reduce the tree's ability to absorb nutrients and maintain healthy growth.",
                "symptoms": [
                    "Slightly drooping or limp fronds",
                    "Dry soil around the root zone",
                    "Leaf tips beginning to brown"
                ],
                "solution": "Water the tree deeply every 7–10 days. Apply a 10cm layer of mulch around the base to retain soil moisture.",
                "urgency": "low"
            },
            {
                "name": "Environmental Stress",
                "icon": "🌤️",
                "reason": "Sudden weather changes, strong winds, or prolonged sun exposure can temporarily stress the tree and affect leaf appearance.",
                "symptoms": [
                    "Mild leaf discoloration or sunburn patches",
                    "Wind-damaged frond edges",
                    "Temporary wilting during hot periods"
                ],
                "solution": "No chemical treatment needed. Ensure adequate irrigation during heat waves. Protect young trees from strong winds using windbreaks.",
                "urgency": "low"
            }
        ]
        immediate_actions = [
            "Visually inspect leaves and trunk for any visible pests or discoloration",
            "Check soil moisture — ensure it's neither too dry nor waterlogged",
            "Review recent fertilizer application schedule"
        ]
        treatment_steps = [
            {
                "step": 1,
                "title": "Soil Inspection",
                "detail": "Test soil pH (ideal: 5.5–7.0). Apply balanced NPK fertilizer (14-14-14) at 500g per tree if deficiency is suspected."
            },
            {
                "step": 2,
                "title": "Pest Monitoring",
                "detail": "Check undersides of leaves for mites or white flies. Apply neem oil spray (2ml/L water) as a preventive measure."
            },
            {
                "step": 3,
                "title": "Watering Adjustment",
                "detail": "Water the tree deeply once every 7–10 days. Ensure good drainage to avoid root waterlogging."
            },
            {
                "step": 4,
                "title": "Follow-up Monitoring",
                "detail": "Re-inspect the tree after 2 weeks to check if the condition has improved."
            }
        ]
        preventive_measures = [
            "Apply mulch (10cm layer) around the base to retain moisture",
            "Maintain regular fertilization schedule (every 3 months)",
            "Keep weeds cleared around the tree base"
        ]

    elif unhealthy_percentage <= 60:
        severity = "Moderate"
        urgency = "Medium"
        urgency_color = "#FF9800"
        severity_description = "Significant health issues detected. The tree requires attention within the next few days to prevent further deterioration."
        possible_conditions = [
            {
                "name": "Nutrient Deficiency",
                "icon": "🌿",
                "reason": "Prolonged shortage of Nitrogen, Potassium, or Magnesium weakens the tree, reducing its ability to produce healthy fronds and coconuts.",
                "symptoms": [
                    "Visible yellowing or browning of leaves",
                    "Stunted or slow new frond growth",
                    "Orange tints on mid-canopy leaves"
                ],
                "solution": "Apply high-potassium fertilizer (MOP 0-0-60) at 1kg/tree + magnesium sulphate 200g/tree. Follow up with foliar micronutrient spray (Zinc, Boron) monthly.",
                "urgency": "medium"
            },
            {
                "name": "Pest Infestation",
                "icon": "🐛",
                "reason": "Moderate populations of coconut mites, caterpillars, or white flies are actively feeding on leaves and sap, causing structural damage.",
                "symptoms": [
                    "Visible chewed leaf edges or holes",
                    "White/silvery patches on fronds",
                    "Presence of pest colonies on leaf undersides"
                ],
                "solution": "Apply systemic insecticide (Imidacloprid 17.8% SL at 0.5ml/L) by foliar spray. Repeat after 14 days. Install yellow sticky traps to monitor population.",
                "urgency": "medium"
            },
            {
                "name": "Fungal or Bacterial Infection",
                "icon": "🍄",
                "reason": "Humid conditions or physical damage allow fungal pathogens (Phytophthora, Colletotrichum) or bacteria to enter and infect foliage or trunk tissue.",
                "symptoms": [
                    "Brown or black spots on leaf surfaces",
                    "Rotting smell from the crown area",
                    "Wilting or collapse of young fronds"
                ],
                "solution": "Spray Copper Oxychloride (3g/L water) on affected areas. For crown symptoms, pour Bordeaux mixture (1%) into the crown. Repeat every 10 days for 3 applications.",
                "urgency": "medium"
            },
            {
                "name": "Root Stress",
                "icon": "🌱",
                "reason": "Waterlogging, soil compaction, or root disease reduces oxygen availability and blocks nutrient uptake, causing visible wilting and decline.",
                "symptoms": [
                    "General wilting despite adequate rainfall",
                    "Yellowing starting from lower fronds upward",
                    "Soggy or waterlogged soil at the base"
                ],
                "solution": "Improve drainage by creating trenches around the root zone. Drench with Trichoderma viride (10g/L water) to combat root pathogens and promote root recovery.",
                "urgency": "high"
            }
        ]
        immediate_actions = [
            "Closely examine the entire tree — trunk, leaves, and root base",
            "Identify visible pest colonies or fungal spots immediately",
            "Stop any waterlogging by improving drainage around the tree",
            "Isolate affected fronds to prevent disease spread if possible"
        ]
        treatment_steps = [
            {
                "step": 1,
                "title": "Pest Control Treatment",
                "detail": "Apply systemic insecticide (Imidacloprid 17.8% SL at 0.5ml/L water) by soil drenching or foliar spray. Repeat after 14 days."
            },
            {
                "step": 2,
                "title": "Fungicide Application",
                "detail": "Spray Copper Oxychloride (3g/L water) on affected areas. For bud rot, pour Bordeaux mixture (1%) into the crown."
            },
            {
                "step": 3,
                "title": "Fertilizer Boost",
                "detail": "Apply a high-potassium fertilizer (0-0-60 MOP) at 1kg per tree to strengthen tree immunity. Follow with magnesium sulphate (200g/tree)."
            },
            {
                "step": 4,
                "title": "Root Treatment",
                "detail": "Mix Trichoderma viride (10g/L water) and drench around the root zone to combat root diseases and promote recovery."
            },
            {
                "step": 5,
                "title": "Reassessment",
                "detail": "Monitor progress after 1 week. If condition persists or worsens, escalate to severe treatment protocol."
            }
        ]
        preventive_measures = [
            "Install yellow sticky traps to monitor and reduce white fly population",
            "Apply neem cake (500g/tree) to the soil every 3 months as pest deterrent",
            "Ensure proper spacing between trees for good air circulation",
            "Avoid over-irrigation — use drip irrigation if available"
        ]

    elif unhealthy_percentage <= 80:
        severity = "Severe"
        urgency = "High"
        urgency_color = "#F44336"
        severity_description = "Serious condition detected. Immediate treatment is required to save the tree. Delay may lead to irreversible damage."
        possible_conditions = [
            {
                "name": "Advanced Pest Infestation",
                "icon": "🐛",
                "reason": "Heavy infestations of rhinoceros beetles, coconut mites, or red palm weevils bore into and feed on trunk and crown tissue, causing extensive structural damage.",
                "symptoms": [
                    "Large entry holes in the trunk or crown",
                    "Fronds cut or chewed at the base",
                    "Sawdust-like frass around trunk entry points",
                    "Extensive browning or collapse of multiple fronds"
                ],
                "solution": "Inject Imidacloprid solution (1ml in 4ml water) into trunk entry holes and seal with mud. Apply Metarhizium anisopliae bio-pesticide to soil. Install pheromone traps for rhinoceros beetle monitoring.",
                "urgency": "high"
            },
            {
                "name": "Bud Rot Disease",
                "icon": "🍄",
                "reason": "Phytophthora palmivora fungal infection attacks the growing bud (heart) of the coconut palm, rapidly decaying the crown and preventing new growth.",
                "symptoms": [
                    "Foul smell from the crown area",
                    "Soft, water-soaked crown base when pressed",
                    "Young spear leaf turning brown and collapsing",
                    "Crown fronds wilting and falling together"
                ],
                "solution": "Pour Metalaxyl + Mancozeb solution (3g/L) into the crown. Remove rotting tissue with a sterilized knife. Apply Bordeaux paste on all cut surfaces. Repeat every 2 weeks.",
                "urgency": "high"
            },
            {
                "name": "Stem Bleeding",
                "icon": "🩸",
                "reason": "Bacterial infection (Erwinia or Thielaviopsis) causes internal trunk decay, resulting in reddish-brown sap oozing from cracks in the bark.",
                "symptoms": [
                    "Reddish-brown or dark liquid oozing from trunk",
                    "Soft, spongy areas on the trunk surface",
                    "Foul fermentation smell from affected trunk areas",
                    "Bark cracking or peeling around the oozing zone"
                ],
                "solution": "Chisel out decayed trunk tissue completely. Disinfect with Bordeaux paste or copper-based fungicide. Inject Fosetyl Aluminum (Aliette 3g/L) into the soil root zone. Monitor weekly.",
                "urgency": "high"
            },
            {
                "name": "Severe Nutrient Depletion",
                "icon": "⚠️",
                "reason": "Long-term neglect of fertilization causes critical multi-nutrient deficiency, starving the tree of essential elements needed for growth and defense.",
                "symptoms": [
                    "Widespread yellowing and browning of most fronds",
                    "Dramatically reduced or no new frond production",
                    "Very small or no coconut development",
                    "Thin, pale trunk with sparse canopy"
                ],
                "solution": "Apply 2kg NPK (12-12-17-2) + 1kg Muriate of Potash + 500g Magnesium Sulphate per tree. Split into 2 applications 1 month apart. Follow with foliar micronutrient spray (Zinc, Boron, Iron) every 3 weeks.",
                "urgency": "high"
            }
        ]
        immediate_actions = [
            "IMMEDIATELY inspect the tree crown for bud rot (foul smell, soft crown base)",
            "Check trunk for weevil entry holes or bleeding sap",
            "Remove and burn all severely infected fronds away from the plantation",
            "Isolate the tree from adjacent trees to prevent disease spread",
            "Contact an agricultural officer or expert as soon as possible"
        ]
        treatment_steps = [
            {
                "step": 1,
                "title": "Emergency Crown Treatment",
                "detail": "Pour Metalaxyl + Mancozeb solution (3g/L) into the crown. Remove rotting tissues with a sterilized knife. Apply Bordeaux paste on cut surfaces."
            },
            {
                "step": 2,
                "title": "Trunk Injection",
                "detail": "Inject Imidacloprid solution (1ml in 4ml water) into trunk holes to combat internal pests. Seal holes with mud after injection."
            },
            {
                "step": 3,
                "title": "Intensive Fertilization",
                "detail": "Apply 2kg NPK (12-12-17-2) + 1kg Muriate of Potash + 500g Magnesium Sulphate per tree. Split into 2 applications 1 month apart."
            },
            {
                "step": 4,
                "title": "Soil Drench Treatment",
                "detail": "Drench root zone with Fosetyl Aluminum (Aliette 3g/L) to combat root and soil-borne pathogens. Apply 5L per tree."
            },
            {
                "step": 5,
                "title": "Weekly Monitoring",
                "detail": "Inspect the tree every week. Document any changes. If no improvement after 3 weeks, consider expert consultation for possible removal."
            }
        ]
        preventive_measures = [
            "Install pheromone traps for rhinoceros beetle monitoring",
            "Apply Metarhizium anisopliae (bio-pesticide) to soil around root zone",
            "Remove all dead organic matter around the tree immediately",
            "Increase monitoring frequency to twice weekly for adjacent trees"
        ]

    else:
        severity = "Critical"
        urgency = "Emergency"
        urgency_color = "#B71C1C"
        severity_description = "CRITICAL condition detected. The tree is at high risk of death. Emergency intervention is required immediately."
        possible_conditions = [
            {
                "name": "Fatal Crown Rot",
                "icon": "💀",
                "reason": "Advanced Phytophthora bud rot has completely destroyed the growing point (apical meristem). Without a functional growing tip, the tree cannot produce new growth.",
                "symptoms": [
                    "All young fronds collapsed and rotted",
                    "Crown emitting strong foul odor",
                    "No new spear leaf visible",
                    "Crown base completely soft and black when touched"
                ],
                "solution": "Emergency: bring agricultural officer within 24 hours. Excavate all rotted crown tissue, disinfect with formalin (2%), apply copper-based fungicide paste. Recovery chance is low — prepare for possible removal.",
                "urgency": "high"
            },
            {
                "name": "Red Palm Weevil",
                "icon": "🪲",
                "reason": "Rhynchophorus ferrugineus larvae bore through and consume the internal trunk tissue, destroying vascular bundles essential for water and nutrient transport.",
                "symptoms": [
                    "Squeaking or rustling sounds from inside the trunk",
                    "Multiple large entry holes with fermented sap",
                    "Trunk feels hollow when tapped",
                    "Wilting of entire crown despite no obvious external cause"
                ],
                "solution": "Inject Emamectin Benzoate (0.4g in 4ml water) at 1m intervals along the trunk. Install pheromone traps. If damage exceeds 60% of trunk internally, tree removal may be required to protect the plantation.",
                "urgency": "high"
            },
            {
                "name": "Root System Collapse",
                "icon": "🌱",
                "reason": "Severe Ganoderma or Pythium root disease, or prolonged waterlogging, has destroyed the majority of the root system, cutting off all nutrient and water supply to the tree.",
                "symptoms": [
                    "Tree leaning or unstable despite no wind",
                    "Mushroom-like growth (Ganoderma conk) at trunk base",
                    "Entire crown wilting uniformly",
                    "Roots pulling out easily when dug up"
                ],
                "solution": "No curative treatment available for advanced Ganoderma. Remove the tree immediately. Sterilize soil with lime (2kg/m²) and leave fallow for 6 months before replanting with resistant varieties.",
                "urgency": "high"
            },
            {
                "name": "Multiple Disease Complex",
                "icon": "⛔",
                "reason": "Simultaneous infestation by multiple pests AND pathogens has overwhelmed the tree's immune defenses, causing rapid and widespread breakdown of all systems.",
                "symptoms": [
                    "Simultaneous symptoms from multiple disease types",
                    "Rapid decline with no response to treatment",
                    "Almost all fronds affected with different damage patterns",
                    "Tree shows no new healthy growth for several months"
                ],
                "solution": "Contact Sri Lanka Coconut Research Institute (CRI) immediately for expert assessment. Apply broad-spectrum treatment (systemic insecticide + fungicide + fertilizer) while awaiting expert. Isolate tree from neighbors.",
                "urgency": "high"
            }
        ]
        immediate_actions = [
            "EMERGENCY: Contact an agricultural expert or plant pathologist immediately",
            "Do NOT delay — within 24–48 hours the condition may become irreversible",
            "Isolate the tree completely to protect neighboring trees",
            "Remove all fallen leaves and debris around the tree — burn or bury them",
            "Photograph all symptoms for expert diagnosis assistance"
        ]
        treatment_steps = [
            {
                "step": 1,
                "title": "Expert Assessment",
                "detail": "Bring a certified agricultural officer to assess whether the tree can be saved. In many critical cases, removal is the only option to protect other trees."
            },
            {
                "step": 2,
                "title": "Emergency Chemical Treatment",
                "detail": "Inject Emamectin Benzoate (0.4g in 4ml water) directly into the trunk at 1m intervals. Apply Phosphorous Acid (Fosetyl-Al) soil drench immediately."
            },
            {
                "step": 3,
                "title": "Crown Excavation",
                "detail": "If crown rot is present, carefully excavate all rotted tissue, disinfect with formalin (2%), and apply protective copper-based fungicide paste."
            },
            {
                "step": 4,
                "title": "Intensive Care Protocol",
                "detail": "Daily monitoring for 2 weeks. Apply foliar micronutrient spray (Zinc, Boron, Iron) weekly to support any remaining healthy tissue."
            },
            {
                "step": 5,
                "title": "Removal Decision",
                "detail": "If no improvement after 3 weeks of intensive treatment, remove the tree immediately. Sterilize the soil with lime (2kg/m²) before replanting."
            }
        ]
        preventive_measures = [
            "Replant with certified disease-resistant coconut seedlings after soil treatment",
            "Implement plantation-wide pest surveillance program",
            "Establish buffer zone of 10m around the affected area",
            "Consult with Sri Lanka Coconut Research Institute (CRI) for region-specific advice"
        ]

    return {
        "severity": severity,
        "urgency": urgency,
        "urgency_color": urgency_color,
        "severity_description": severity_description,
        "possible_conditions": possible_conditions,
        "immediate_actions": immediate_actions,
        "treatment_steps": treatment_steps,
        "preventive_measures": preventive_measures
    }


# ==== Tree Health Visual Analysis ====

TREE_VISUAL_CONDITIONS = {
    'sparse_crown': {
        'name': 'Sparse Crown / Few Fronds',
        'icon': '🌴',
        'reason': 'The tree crown appears to have fewer fronds than a healthy coconut palm. A healthy tree should carry 25–30 live fronds at all times. Sparse fronds drastically reduce photosynthesis, weakening the tree and lowering coconut yield.',
        'symptoms': [
            'Crown looks thin or patchy when viewed from a distance',
            'Sky or background visible through wide gaps between fronds',
            'Less shade cast under the tree than a healthy coconut palm'
        ],
        'solution': 'Inspect the crown for rhinoceros beetle entry holes or bud rot symptoms. Apply NPK fertilizer (12-12-17) at 1kg per tree and water regularly (50–100L/week) to stimulate new frond growth.',
        'urgency': 'high'
    },
    'yellowing_fronds': {
        'name': 'Yellowing or Pale Fronds',
        'icon': '🍂',
        'reason': 'Fronds turning yellow or pale indicate Nitrogen or Potassium deficiency, or water stress. Yellow fronds cannot photosynthesize effectively, slowing tree growth and nut development.',
        'symptoms': [
            'Fronds showing yellow or pale green coloration',
            'Yellowing starting from older lower fronds and progressing upward',
            'Leaves look washed-out or dull rather than vibrant green'
        ],
        'solution': 'Apply urea (46-0-0) at 200g per tree for Nitrogen, or muriate of potash (60% KCl) at 250g per tree for Potassium. Water the tree 50–100L per week. Spray magnesium sulphate (20g/L water) as foliar feed monthly.',
        'urgency': 'medium'
    },
    'dead_fronds': {
        'name': 'Dead or Browning Fronds',
        'icon': '🍁',
        'reason': 'Brown or dead fronds still attached to the tree indicate frond death from disease, severe pest damage, or prolonged stress. Dead fronds harbour pests and fungal spores that can spread to healthy fronds.',
        'symptoms': [
            'Brown or blackened fronds hanging on the tree',
            'Dead fronds not naturally shedding as healthy fronds do',
            'Browning progressing from frond tips inward toward the midrib'
        ],
        'solution': 'Prune all dead fronds with a sterilized cutting tool. Apply Bordeaux paste or copper fungicide on all cut surfaces. Spray Copper Oxychloride (3g/L) on remaining fronds to prevent fungal spread.',
        'urgency': 'medium'
    },
    'crown_damage': {
        'name': 'Crown Damage or Distortion',
        'icon': '🌪️',
        'reason': 'The crown appears abnormally dark, asymmetric, or structurally damaged — likely from pest boring, mechanical damage, or disease affecting the growing point of the tree.',
        'symptoms': [
            'Crown appears lopsided or asymmetric when viewed from the side',
            'Abnormal dark patches or discoloration visible in the crown area',
            'Fronds growing in unusual directions or collapsing inward'
        ],
        'solution': 'Inspect the crown closely for rhinoceros beetle holes or bud rot (soft crown base, foul smell). Pour Metalaxyl + Mancozeb (3g/L) into the crown as a precaution. Consult an agricultural officer if symptoms persist.',
        'urgency': 'high'
    },
    'general_decline': {
        'name': 'General Tree Decline',
        'icon': '⚠️',
        'reason': 'The tree shows overall health decline without a single dominant visual cause — likely from combined stressors including nutrient depletion, inconsistent watering, or early-stage disease.',
        'symptoms': [
            'Overall dull or unhealthy appearance compared to healthy palms',
            'Reduced vigor with slow or no new frond production',
            'Tree appears stressed without a clearly dominant visible cause'
        ],
        'solution': 'Apply balanced NPK fertilizer (14-14-14) at 500g per tree. Water regularly (50–100L per week). Inspect crown, trunk, and root zone thoroughly. Consult an agricultural expert if no improvement within 4 weeks.',
        'urgency': 'medium'
    }
}


def analyze_tree_visual_reasons(img_array):
    """
    Analyzes color distribution in the tree image to detect visual health indicators.
    img_array: numpy array of shape (1, 224, 224, 3), values 0-1
    Returns: list of condition keys (max 3) from TREE_VISUAL_CONDITIONS
    """
    img = img_array[0]  # Shape: (224, 224, 3)

    # Upper 55% of image = crown/frond area
    crown = img[:123, :, :]
    r = crown[:, :, 0]
    g = crown[:, :, 1]
    b = crown[:, :, 2]

    # Healthy green fronds: green channel dominates
    green_mask = (g > r + 0.04) & (g > b + 0.04) & (g > 0.18)
    # Yellowing fronds: warm pale tones — high R, medium G, low B
    yellow_mask = (r > 0.42) & (g > 0.34) & (b < 0.28) & ~green_mask
    # Dead/brown fronds: brownish — medium-high R, low G, low B
    brown_mask = (r > 0.28) & (g < 0.28) & (b < 0.22) & ~green_mask
    # Sky visible through sparse crown: blue dominant, fairly bright
    sky_mask = (b > r + 0.06) & (b > g + 0.06) & (b > 0.38)
    # Dark areas suggesting crown damage or abnormality
    dark_mask = (r < 0.18) & (g < 0.18) & (b < 0.18)

    green_ratio = float(np.mean(green_mask))
    yellow_ratio = float(np.mean(yellow_mask))
    brown_ratio = float(np.mean(brown_mask))
    sky_ratio = float(np.mean(sky_mask))
    dark_ratio = float(np.mean(dark_mask))

    detected = []

    # Sparse crown: too much sky visible OR very low green coverage
    if sky_ratio > 0.16 or green_ratio < 0.10:
        detected.append('sparse_crown')

    # Yellowing: significant yellow/pale tones in crown area
    if yellow_ratio > 0.09:
        detected.append('yellowing_fronds')

    # Dead fronds: significant brown tones in crown area
    if brown_ratio > 0.09:
        detected.append('dead_fronds')

    # Crown damage: high dark ratio suggesting structural abnormality
    if dark_ratio > 0.28 and 'sparse_crown' not in detected:
        detected.append('crown_damage')

    # Fallback if nothing specific detected
    if not detected:
        if green_ratio < 0.22:
            detected.append('sparse_crown')
        else:
            detected.append('general_decline')

    return detected[:1]


def build_tree_health_analysis(visual_reasons, confidence):
    """
    Builds the complete health analysis info from detected visual reasons.
    """
    conditions = [TREE_VISUAL_CONDITIONS[r] for r in visual_reasons if r in TREE_VISUAL_CONDITIONS]

    urgencies = [c['urgency'] for c in conditions]
    if 'high' in urgencies:
        urgency = 'High'
        urgency_color = '#F44336'
    elif 'medium' in urgencies:
        urgency = 'Medium'
        urgency_color = '#FF9800'
    else:
        urgency = 'Low'
        urgency_color = '#FFC107'

    condition_names = ' and '.join([c['name'] for c in conditions])
    severity_description = (
        f"Visual analysis identified {len(conditions)} health concern(s): {condition_names}. "
        f"Address these issues promptly to prevent further decline."
    )

    immediate_actions = []
    if 'sparse_crown' in visual_reasons:
        immediate_actions.append("Closely inspect the crown for beetle entry holes or soft/rotting areas")
    if 'yellowing_fronds' in visual_reasons:
        immediate_actions.append("Check soil moisture — yellowing is linked to water stress or nutrient deficiency")
    if 'dead_fronds' in visual_reasons:
        immediate_actions.append("Prune all dead fronds immediately using a sterilized cutting tool")
    if 'crown_damage' in visual_reasons:
        immediate_actions.append("Inspect crown closely for pest boring, bud rot smell, or mechanical damage")
    if 'general_decline' in visual_reasons:
        immediate_actions.append("Conduct a full tree inspection — check trunk, crown, and root zone")
    immediate_actions.append("Apply balanced NPK fertilizer if last application was more than 3 months ago")
    immediate_actions.append("Ensure the tree receives adequate water (50–100 liters per week)")

    step_num = 1
    treatment_steps = []

    if 'sparse_crown' in visual_reasons or 'crown_damage' in visual_reasons:
        treatment_steps.append({
            'step': step_num,
            'title': 'Crown Inspection & Pest Control',
            'detail': 'Check for rhinoceros beetle holes. If found, inject Imidacloprid (1ml in 4ml water) into entry holes and seal with mud. Pour Metalaxyl + Mancozeb (3g/L) into the crown as disease prevention.'
        })
        step_num += 1

    if 'yellowing_fronds' in visual_reasons:
        treatment_steps.append({
            'step': step_num,
            'title': 'Nutrient Correction',
            'detail': 'Apply urea (200g/tree) for Nitrogen or muriate of potash (250g/tree) for Potassium. Follow with magnesium sulphate foliar spray (20g/L). Repeat monthly for 3 months.'
        })
        step_num += 1

    if 'dead_fronds' in visual_reasons:
        treatment_steps.append({
            'step': step_num,
            'title': 'Frond Pruning & Sanitation',
            'detail': 'Prune all dead fronds cleanly. Apply Bordeaux paste on all cut surfaces. Spray Copper Oxychloride (3g/L) on remaining fronds. Burn or bury removed fronds away from the tree.'
        })
        step_num += 1

    treatment_steps.append({
        'step': step_num,
        'title': 'General Fertilization',
        'detail': 'Apply NPK fertilizer (12-12-17-2) at 1kg per tree. Follow with 200g of magnesium sulphate. Irrigate thoroughly after application. Repeat every 3 months.'
    })
    step_num += 1

    treatment_steps.append({
        'step': step_num,
        'title': 'Follow-up Monitoring',
        'detail': 'Re-inspect the tree after 2–3 weeks. Photograph the crown to track progress. If condition worsens, consult the Sri Lanka Coconut Research Institute (CRI) or your local agricultural officer.'
    })

    preventive_measures = [
        "Monitor the tree weekly and photograph the crown for comparison over time",
        "Maintain a regular fertilization schedule every 3 months (NPK + magnesium)",
        "Ensure consistent irrigation — 50–100 liters per week, more during dry seasons",
        "Install pheromone traps nearby to monitor and control rhinoceros beetle populations",
        "Keep the area around the tree base free of weeds and decaying organic matter"
    ]

    return {
        'severity': 'Detected',
        'urgency': urgency,
        'urgency_color': urgency_color,
        'severity_description': severity_description,
        'possible_conditions': conditions,
        'immediate_actions': immediate_actions,
        'treatment_steps': treatment_steps,
        'preventive_measures': preventive_measures
    }


@app.route('/predict/tree-health', methods=['POST'])
def predict_tree_health():
    """
    Predict if a coconut tree is healthy or unhealthy

    Returns:
        JSON with prediction results
    """
    if models.get('tree_health') is None:
        return jsonify({'error': 'Tree health model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = models['tree_health'].predict(img_array, verbose=0)

        healthy_prob = float(predictions[0][0])
        unhealthy_prob = float(predictions[0][1])

        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = TREE_HEALTH_CLASSES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))

        unhealthy_percentage = int(unhealthy_prob * 100) if predicted_class == 'unhealthy' else 0

        if predicted_class == 'healthy':
            if confidence > 0.95:
                message = "Tree appears to be very healthy!"
            elif confidence > 0.80:
                message = "Tree appears to be healthy."
            else:
                message = "Tree seems healthy but with lower confidence."
            recommendation = "Continue regular monitoring and maintain good care practices."
            severity_info = None
        else:
            if confidence > 0.80:
                message = f"Tree shows signs of being unhealthy ({unhealthy_percentage}% unhealthy)."
            else:
                message = f"Possible unhealthy condition detected ({unhealthy_percentage}% unhealthy)."
            recommendation = "Inspect the tree for pest damage, disease, or nutrient deficiencies. Consider consulting an agricultural expert."
            visual_reasons = analyze_tree_visual_reasons(img_array)
            severity_info = build_tree_health_analysis(visual_reasons, confidence)

        result = {
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'healthy': healthy_prob,
                'unhealthy': unhealthy_prob
            },
            'unhealthy_percentage': unhealthy_percentage,
            'is_healthy': predicted_class == 'healthy',
            'message': message,
            'recommendation': recommendation,
            'severity_info': severity_info,
            'model_info': {
                'version': 'v2',
                'classes': TREE_HEALTH_CLASSES,
                'accuracy': model_infos['tree_health'].get('test_performance', {}).get('accuracy', 'TBD') if model_infos.get('tree_health') else 'TBD',
                'macro_f1': model_infos['tree_health'].get('test_performance', {}).get('macro_f1', 'TBD') if model_infos.get('tree_health') else 'TBD'
            },
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Legacy endpoint for backward compatibility

# Legacy endpoint

@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy endpoint - redirects to mite detection"""
    return predict_mite()


# ============================================================
# AI CHATBOT ENDPOINT (Groq API)
# ============================================================

@app.route('/chat', methods=['POST'])
def chat():
    """
    AI Chatbot endpoint using Groq API
    Specialized for coconut health and farming advice

    Request body:
    {
        "message": "How do I treat mite infection?",
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help?"}
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400

        user_message = data.get('message', '').strip()
        chat_history = data.get('history', [])

        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400

        # Build messages array with system prompt and history
        messages = [
            {"role": "system", "content": COCONUT_EXPERT_PROMPT}
        ]

        # Add chat history (last 10 messages to avoid token limit)
        for msg in chat_history[-10:]:
            messages.append({
                "role": msg.get('role', 'user'),
                "content": msg.get('content', '')
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Call Groq API with retry session
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False
        }

        # Use session with retry logic for better connection handling
        global groq_session
        try:
            response = groq_session.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        except (requests.exceptions.ConnectionError, ConnectionResetError) as conn_err:
            # Connection was reset, create a new session and retry once
            print(f"Connection reset, retrying with new session: {conn_err}")
            groq_session = create_groq_session()
            response = groq_session.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            return jsonify({
                'success': False,
                'error': f'Groq API error: {error_msg}'
            }), 500

        result = response.json()
        assistant_message = result['choices'][0]['message']['content']

        return jsonify({
            'success': True,
            'response': assistant_message,
            'model': GROQ_MODEL,
            'usage': result.get('usage', {})
        })

    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'Request timeout. Please try again.'
        }), 504
    except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
        # Reset the session for next request
        groq_session = create_groq_session()
        return jsonify({
            'success': False,
            'error': 'Connection error. Please try again.'
        }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Network error: {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/chat/health', methods=['GET'])
def chat_health():
    """Check if chat service is available"""
    global groq_session
    try:
        # Quick test to Groq API using session with retry
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        }

        try:
            response = groq_session.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        except (requests.exceptions.ConnectionError, ConnectionResetError):
            # Reset session and retry
            groq_session = create_groq_session()
            response = groq_session.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)

        if response.status_code == 200:
            return jsonify({
                'success': True,
                'status': 'online',
                'model': GROQ_MODEL,
                'message': 'Chat service is ready'
            })
        else:
            return jsonify({
                'success': False,
                'status': 'error',
                'message': 'Groq API not responding'
            }), 503

    except Exception as e:
        groq_session = create_groq_session()  # Reset for next request
        return jsonify({
            'success': False,
            'status': 'offline',
            'message': str(e)
        }), 503


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    load_models()


    # Run the Flask app on port 5001 (port 5000 is used by Node.js auth backend)
    print("\nStarting Coconut Health Monitor ML API v8.0...")
    print("  Mite Model: v10 (3-class, 91.44% accuracy)")
    print("  Unified Model: v1 (4-class - caterpillar + white_fly, 96.08% accuracy)")
    print("  Disease Model: v2 (4-class - Leaf Rot, Leaf Spot, 98.69% accuracy)")
    print("  Leaf Health Model: v4 (2-class, MobileNetV2, 95.00% accuracy)")
    print("  Branch Health Model: v1 (2-class, 99.63% accuracy)")
    print("  Tree Health Model: v2 (2-class, EfficientNetB0, 100% accuracy)")

    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=False)
