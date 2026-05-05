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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
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
BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

# Coconut vs Not-Coconut Validator (v1 - binary, 100% test accuracy)
VALIDATOR_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_vs_notcoconut_v1', 'best_model.keras')
VALIDATOR_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_vs_notcoconut_v1', 'model_info.json')
VALIDATOR_CLASSES = ['coconut', 'not_coconut']

# Mite model paths (v10 - 3-class with Focal Loss) - Used for /predict/all
MITE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v10', 'best_model.keras')
MITE_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v10', 'model_info.json')

# Mite v10 optimal threshold (from threshold tuning)
MITE_THRESHOLD = 0.10
MITE_BOOST_FACTOR = 0.5 / MITE_THRESHOLD  # 5x boost for mite class

# Mite v10 class indices
MITE_CLASSES = ['coconut_mite', 'healthy', 'not_coconut']

# Mite v12 model paths (2-class - Fruit Surface Focused) - Used for /predict/mite
MITE_V12_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v12', 'best_model.keras')
MITE_V12_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_mite_v12', 'model_info.json')

# Mite v12 class indices (binary classification: healthy vs mite)
MITE_V12_CLASSES = ['healthy', 'mite']

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

# Leaf Dieback Detection model paths (v4 - 3-class for baby coconut trees)
LEAF_DIEBACK_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_dieback_v4', 'best_model.keras')
LEAF_DIEBACK_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_dieback_v4', 'model_info.json')

# Leaf Dieback model class indices (alphabetical order from ImageDataGenerator)
LEAF_DIEBACK_CLASSES = ['healthy', 'leaf_die_back', 'not_cocount']

# Leaf Health model paths (v3 - 2-class, for drone images)
LEAF_HEALTH_V3_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v3', 'best_model.keras')
LEAF_HEALTH_V3_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v3', 'model_info.json')

# Leaf Health model paths (v4 - 2-class, for phone images)
LEAF_HEALTH_V4_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v4', 'best_model.keras')
LEAF_HEALTH_V4_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'leaf_health_v4', 'model_info.json')

# Leaf Health class indices (same for both v3 and v4)
LEAF_HEALTH_CLASSES = ['healthy', 'unhealthy']

# Branch Health model paths (v1 - 2-class)
BRANCH_HEALTH_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_branch_health_v1', 'best_model.keras')
BRANCH_HEALTH_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_branch_health_v1', 'model_info.json')

# Branch Health v1 class indices
BRANCH_HEALTH_CLASSES = ['healthy', 'unhealthy']

# Tree Health model paths (v2 - 2-class, 100% accuracy)
TREE_HEALTH_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_tree_health_v2', 'best_model.keras')
TREE_HEALTH_MODEL_INFO_PATH = os.path.join(BASE_MODEL_PATH, 'coconut_tree_health_v2', 'model_info.json')

# Tree Health v2 class indices
TREE_HEALTH_CLASSES = ['healthy', 'unhealthy']

# Bunch Detection TFLite model path
BUNCH_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'bunch_detection', 'best_float32.tflite')

# Bunch Detection configuration
BUNCH_CONFIDENCE_THRESHOLD = 0.56  # Confidence threshold (tested in Roboflow)
BUNCH_IOU_THRESHOLD = 0.45  # IoU threshold for NMS
BUNCH_INPUT_SIZE = 640  # YOLOv8 default input size
BUNCH_MAX_DETECTIONS = 50  # Maximum bunches to return

# Yield Estimator (Nuts per Tree) TFLite model path
YIELD_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'Coconut_Yield_Estimator (Nuts per Tree)', 'best_float32.tflite')

# Yield Estimator configuration
YIELD_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for nut detection
YIELD_IOU_THRESHOLD = 0.45  # IoU threshold for NMS
YIELD_INPUT_SIZE = 960  # Model trained with 960x960 input
YIELD_MAX_DETECTIONS = 100  # Maximum nuts to return per image

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

def focal_loss_fixed(y_true, y_pred):
    """Custom focal loss for mite v12 model"""
    gamma = 2.0
    alpha = 0.25
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
    fl = weight * ce
    return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

def load_models():
    """Load all trained models"""
    global models, model_infos

    print("=" * 60)
    print("  Loading Coconut Health Monitor Models...")
    print("=" * 60)

    # Load Coconut Validator (binary - 100% accuracy)
    try:
        print("\n[0] Loading Coconut Validator (binary - 100% accuracy)...")
        models['validator'] = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
        try:
            with open(VALIDATOR_MODEL_INFO_PATH, 'r') as f:
                model_infos['validator'] = json.load(f)
        except:
            model_infos['validator'] = {
                'version': 'v1',
                'classes': VALIDATOR_CLASSES,
                'accuracy': 1.0,
            }
        print(f"    Classes: {VALIDATOR_CLASSES}")
        print(f"    Test Accuracy: 100%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading validator: {e}")
        models['validator'] = None
        model_infos['validator'] = None

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

    # Load Mite Model v12 (2-class - Fruit Surface Focused) - For /predict/mite endpoint
    try:
        print("\n[1b] Loading Coconut Mite model (v12 - 2-class, Fruit Focused)...")

        models['mite_v12'] = tf.keras.models.load_model(
            MITE_V12_MODEL_PATH,
            custom_objects={'focal_loss_fixed': focal_loss_fixed}
        )

        try:
            with open(MITE_V12_MODEL_INFO_PATH, 'r') as f:
                model_infos['mite_v12'] = json.load(f)
        except:
            model_infos['mite_v12'] = {
                'version': 'v12_fruit_focused',
                'classes': MITE_V12_CLASSES,
                'performance': {'test_accuracy': 0.9744, 'mite_recall': 0.9614}
            }

        print(f"    Version: v12 (2-class, Fruit Surface Focused)")
        print(f"    Classes: {MITE_V12_CLASSES}")
        print(f"    Accuracy: 97.44%")
        print(f"    Mite Recall: 96.14%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading mite v12 model: {e}")
        models['mite_v12'] = None
        model_infos['mite_v12'] = None

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

    # Load Leaf Dieback Model (v4 - 3-class for baby coconut trees)
    try:
        print("\n[4] Loading Leaf Dieback model (v4 - 3-class for baby coconut)...")

        models['leaf_dieback'] = tf.keras.models.load_model(
            LEAF_DIEBACK_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        try:
            with open(LEAF_DIEBACK_MODEL_INFO_PATH, 'r') as f:
                model_infos['leaf_dieback'] = json.load(f)
        except:
            model_infos['leaf_dieback'] = {
                'version': 'v4_3class',
                'classes': LEAF_DIEBACK_CLASSES,
                'performance': {
                    'healthy_recall': 1.00,
                    'leaf_die_back_recall': 0.845,
                    'not_cocount_recall': 0.988
                }
            }

        print(f"    Version: v4 (3-class, MobileNetV2)")
        print(f"    Classes: {LEAF_DIEBACK_CLASSES}")
        print(f"    Healthy Recall: 100%")
        print(f"    Leaf Dieback Recall: 84.5%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading leaf dieback model: {e}")
        models['leaf_dieback'] = None
        model_infos['leaf_dieback'] = None

    # Load Leaf Health Model v3 (for drone images)
    try:
        print("\n[5a] Loading Leaf Health model v3 (drone images)...")

        models['leaf_health_v3'] = tf.keras.models.load_model(
            LEAF_HEALTH_V3_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        try:
            with open(LEAF_HEALTH_V3_MODEL_INFO_PATH, 'r') as f:
                model_infos['leaf_health_v3'] = json.load(f)
        except:
            model_infos['leaf_health_v3'] = {
                'version': 'v3_drone',
                'classes': LEAF_HEALTH_CLASSES,
                'performance': {'test_accuracy': 0.9993}
            }

        print(f"    Version: v3 (2-class, for drone images)")
        print(f"    Classes: {LEAF_HEALTH_CLASSES}")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading leaf health v3 model: {e}")
        models['leaf_health_v3'] = None
        model_infos['leaf_health_v3'] = None

    # Load Leaf Health Model v4 (for phone images)
    try:
        print("\n[5b] Loading Leaf Health model v4 (phone images)...")

        models['leaf_health_v4'] = tf.keras.models.load_model(
            LEAF_HEALTH_V4_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        try:
            with open(LEAF_HEALTH_V4_MODEL_INFO_PATH, 'r') as f:
                model_infos['leaf_health_v4'] = json.load(f)
        except:
            model_infos['leaf_health_v4'] = {
                'version': 'v4_phone',
                'classes': LEAF_HEALTH_CLASSES,
                'performance': {'test_accuracy': 0.95}
            }

        print(f"    Version: v4 (2-class, for phone images)")
        print(f"    Classes: {LEAF_HEALTH_CLASSES}")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading leaf health v4 model: {e}")
        models['leaf_health_v4'] = None
        model_infos['leaf_health_v4'] = None

    # Load Branch Health Model (v1 - 2-class)
    try:
        print("\n[6] Loading Branch Health model (v1 - 2-class)...")

        models['branch_health'] = tf.keras.models.load_model(
            BRANCH_HEALTH_MODEL_PATH,
            custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        )

        try:
            with open(BRANCH_HEALTH_MODEL_INFO_PATH, 'r') as f:
                model_infos['branch_health'] = json.load(f)
        except:
            model_infos['branch_health'] = {
                'version': 'v1_2class',
                'classes': BRANCH_HEALTH_CLASSES,
                'performance': {'test_accuracy': 0.9963}
            }

        print(f"    Version: v1 (2-class, MobileNetV2)")
        print(f"    Classes: {BRANCH_HEALTH_CLASSES}")
        print(f"    Accuracy: 99.63%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading branch health model: {e}")
        models['branch_health'] = None
        model_infos['branch_health'] = None

    # Load Tree Health Model (v2 - 2-class, 100% accuracy)
    try:
        print("\n[7] Loading Tree Health model (v2 - 2-class)...")

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
                'performance': {'test_accuracy': 1.0}
            }

        print(f"    Version: v2 (2-class, EfficientNetB0)")
        print(f"    Classes: {TREE_HEALTH_CLASSES}")
        print(f"    Accuracy: 100%")
        print("    Status: LOADED")
    except Exception as e:
        print(f"    ERROR loading tree health model: {e}")
        models['tree_health'] = None
        model_infos['tree_health'] = None

    # Load Bunch Detection Model (TFLite)
    try:
        print("\n[8] Loading Bunch Detection model (TFLite)...")
        # Note: TFLite model is loaded on demand in the predict function
        # Just check if the file exists
        if os.path.exists(BUNCH_MODEL_PATH):
            models['bunch'] = BUNCH_MODEL_PATH  # Store path instead of loaded model
            model_infos['bunch'] = {
                'version': 'v1_yolov8',
                'type': 'object_detection',
                'format': 'tflite'
            }
            print(f"    Format: TFLite (YOLOv8)")
            print(f"    Input Size: {BUNCH_INPUT_SIZE}x{BUNCH_INPUT_SIZE}")
            print("    Status: LOADED")
        else:
            raise FileNotFoundError(f"Model not found: {BUNCH_MODEL_PATH}")
    except Exception as e:
        print(f"    ERROR loading bunch detection model: {e}")
        models['bunch'] = None
        model_infos['bunch'] = None

    # Load Yield Estimator Model (TFLite - Nuts per Tree)
    try:
        print("\n[9] Loading Yield Estimator model (TFLite - Nuts per Tree)...")
        if os.path.exists(YIELD_MODEL_PATH):
            models['yield'] = YIELD_MODEL_PATH  # Store path instead of loaded model
            model_infos['yield'] = {
                'version': 'v1_yolov8',
                'type': 'object_detection',
                'format': 'tflite',
                'detects': 'coconut_nuts'
            }
            print(f"    Format: TFLite (YOLOv8)")
            print(f"    Input Size: {YIELD_INPUT_SIZE}x{YIELD_INPUT_SIZE}")
            print(f"    Detects: Individual coconut nuts")
            print("    Status: LOADED")
        else:
            raise FileNotFoundError(f"Model not found: {YIELD_MODEL_PATH}")
    except Exception as e:
        print(f"    ERROR loading yield estimator model: {e}")
        models['yield'] = None
        model_infos['yield'] = None

    print("\n" + "=" * 60)
    loaded_count = sum(1 for m in models.values() if m is not None)
    print(f"  Models loaded: {loaded_count}/10")
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

def preprocess_image_leaf_dieback(image_bytes):
    """Preprocess image for leaf dieback model (0-1 scaling, 224x224)"""
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)


def preprocess_image_yolo(image_bytes, input_size=640):
    """Preprocess image for YOLOv8 TFLite model"""
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Store original size for scaling detections back
    original_size = img.size  # (width, height)

    # Resize to model input size
    img = img.resize((input_size, input_size), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]

    return np.expand_dims(img_array, axis=0), original_size


def run_tflite_inference(model_path, input_data):
    """Run TFLite model inference"""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def parse_yolo_output(output, conf_threshold=0.5, iou_threshold=0.45, max_detections=50):
    """Parse YOLOv8 output and apply NMS"""
    # YOLOv8 output shape: [1, 5, 8400] or [1, 8400, 5] depending on export
    # Format: [x_center, y_center, width, height, confidence]

    if len(output.shape) == 3:
        if output.shape[1] == 5:
            # Shape [1, 5, 8400] - transpose to [1, 8400, 5]
            output = np.transpose(output, (0, 2, 1))
        output = output[0]  # Remove batch dimension

    detections = []

    for det in output:
        if len(det) >= 5:
            confidence = float(det[4])
            if confidence >= conf_threshold:
                x_center, y_center, width, height = det[0:4]
                detections.append({
                    'x': float(x_center),
                    'y': float(y_center),
                    'width': float(width),
                    'height': float(height),
                    'confidence': confidence
                })

    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    # Simple NMS
    final_detections = []
    for det in detections:
        if len(final_detections) >= max_detections:
            break

        # Check overlap with existing detections
        overlap = False
        for existing in final_detections:
            # Calculate IoU
            iou = calculate_iou(det, existing)
            if iou > iou_threshold:
                overlap = True
                break

        if not overlap:
            final_detections.append(det)

    return final_detections


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    # Convert center format to corner format
    x1_min = box1['x'] - box1['width'] / 2
    y1_min = box1['y'] - box1['height'] / 2
    x1_max = box1['x'] + box1['width'] / 2
    y1_max = box1['y'] + box1['height'] / 2

    x2_min = box2['x'] - box2['width'] / 2
    y2_min = box2['y'] - box2['height'] / 2
    x2_max = box2['x'] + box2['width'] / 2
    y2_max = box2['y'] + box2['height'] / 2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Coconut Health Monitor - Pest & Disease Detection API',
        'version': '9.0.0',
        'models': {
            'mite': {
                'status': 'loaded' if models.get('mite_v12') is not None else 'not loaded',
                'version': 'v12 (2-class, Fruit Surface Focused)',
                'accuracy': '97.44%',
                'note': 'Used for /predict/mite endpoint'
            },
            'mite_v10': {
                'status': 'loaded' if models.get('mite') is not None else 'not loaded',
                'version': 'v10 (3-class, Focal Loss)',
                'accuracy': '91.44%',
                'note': 'Used for /predict/all endpoint'
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
            'leaf_dieback': {
                'status': 'loaded' if models.get('leaf_dieback') is not None else 'not loaded',
                'version': 'v4 (3-class: healthy, leaf_die_back, not_cocount)',
                'description': 'Baby coconut tree disease detection'
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
            '/predict/leaf_dieback': 'POST - Detect leaf dieback in baby coconut trees (3-class)',
            '/predict/all': 'POST - Run all pest detection with smart combined logic'
        }
    })

@app.route('/predict/validate', methods=['POST'])
def predict_validate():
    """Validate if image is a coconut (binary classifier - 100% accuracy)

    Returns:
        is_coconut: bool
        confidence: float (0-1)
        class: 'coconut' | 'not_coconut'
        message: human-readable result
    """
    if models.get('validator') is None:
        return jsonify({'success': False, 'error': 'Validator model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400

    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        # Binary prediction - sigmoid output
        prediction = models['validator'].predict(arr, verbose=0)[0]

        # Handle both sigmoid (single value) and softmax (2 values)
        if len(prediction) == 1:
            not_coconut_prob = float(prediction[0])
            coconut_prob = 1.0 - not_coconut_prob
        else:
            coconut_prob = float(prediction[0])
            not_coconut_prob = float(prediction[1])

        is_coconut = coconut_prob > not_coconut_prob
        confidence = max(coconut_prob, not_coconut_prob)
        predicted_class = 'coconut' if is_coconut else 'not_coconut'

        if is_coconut:
            message = f'This is a coconut image (confidence: {confidence*100:.1f}%)'
        else:
            message = f'This is NOT a coconut image (confidence: {confidence*100:.1f}%). Please upload a coconut image.'

        return jsonify({
            'success': True,
            'is_coconut': is_coconut,
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'coconut': coconut_prob,
                'not_coconut': not_coconut_prob,
            },
            'message': message,
            'model_version': 'v1',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'validator': models.get('validator') is not None,
            'mite': models.get('mite') is not None,
            'unified': models.get('unified') is not None,
            'disease': models.get('disease') is not None,
            'leaf_dieback': models.get('leaf_dieback') is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models with their info"""
    result = {}

    if model_infos.get('mite_v12'):
        result['mite'] = {
            'name': 'Coconut Mite Detection Model (Fruit Surface Focused)',
            'version': 'v12 (2-class)',
            'classes': MITE_V12_CLASSES,
            'accuracy': 0.9744,
            'mite_recall': 0.9614,
            'endpoint': '/predict/mite',
            'loaded': models.get('mite_v12') is not None
        }

    if model_infos.get('mite'):
        result['mite_v10'] = {
            'name': 'Coconut Mite Detection Model (Legacy)',
            'version': 'v10 (3-class, Focal Loss)',
            'classes': MITE_CLASSES,
            'accuracy': 0.9144,
            'mite_recall': 0.79,
            'threshold': MITE_THRESHOLD,
            'boost_factor': MITE_BOOST_FACTOR,
            'endpoint': '/predict/all',
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

    if model_infos.get('leaf_dieback'):
        result['leaf_dieback'] = {
            'name': 'Baby Coconut Leaf Dieback Detection Model',
            'version': 'v4 (3-class, MobileNetV2)',
            'classes': LEAF_DIEBACK_CLASSES,
            'healthy_recall': 1.00,
            'leaf_dieback_recall': 0.845,
            'loaded': models.get('leaf_dieback') is not None
        }

    return jsonify(result)

@app.route('/predict/mite', methods=['POST'])
def predict_mite():
    """Detect coconut mite infection (v10 model - 3-class)

    Uses the v10 model with 3-class classification.
    Classes: coconut_mite, healthy, not_coconut
    Accuracy: 91.44%
    """

    if models.get('mite') is None:
        return jsonify({'error': 'Mite v10 model not loaded'}), 500

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

        # Apply mite boost factor for improved recall
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

        # Map to standard response format
        if not is_valid:
            label = 'Not a Coconut'
            message = 'This image does not appear to be a coconut. Please upload a coconut image.'
        elif is_mite:
            label = 'Coconut Mite Infected'
            message = 'This coconut shows signs of mite infection.'
        else:
            label = 'Healthy'
            message = 'No mite infection detected on this coconut.'

        return jsonify({
            'success': True,
            'pest_type': 'mite',
            'model_version': 'v10',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_infected': is_mite,
                'is_valid_image': is_valid,
                'label': label,
                'message': message
            },
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/mite_v12', methods=['POST'])
def predict_mite_v12():
    """Detect coconut mite infection (v12 model - 2-class, Fruit Surface Focused)

    Uses the v12 model with 2-class classification (binary).
    Classes: healthy, mite
    Accuracy: 97.44% (Fruit Surface Focused)
    """
    if models.get('mite_v12') is None:
        return jsonify({'success': False, 'error': 'Mite v12 model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_mite(image_bytes)

        # v12 2-class classification: softmax output
        # Classes: ['healthy', 'mite']
        predictions = models['mite_v12'].predict(processed_image, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = MITE_V12_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        is_mite = predicted_class == 'mite'

        probabilities = {
            'healthy': float(predictions[0]),
            'mite': float(predictions[1])
        }

        if is_mite:
            label = 'Coconut Mite Infected'
            message = f'This coconut shows mite damage on the surface (confidence: {confidence*100:.1f}%)'
        else:
            label = 'Healthy'
            message = f'No mite damage detected (confidence: {confidence*100:.1f}%)'

        return jsonify({
            'success': True,
            'pest_type': 'mite',
            'model_version': 'v12',
            'accuracy': '97.44%',
            'prediction': {
                'class': 'coconut_mite' if is_mite else 'healthy',
                'confidence': confidence,
                'is_infected': is_mite,
                'is_valid_image': True,
                'label': label,
                'message': message
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

@app.route('/predict/leaf_dieback', methods=['POST'])
def predict_leaf_dieback():
    """Detect leaf dieback in baby coconut trees (v4 model - 3-class classification)

    Classes: healthy, leaf_die_back, not_cocount
    Specifically designed for young/baby coconut palm disease detection.
    """

    if models.get('leaf_dieback') is None:
        return jsonify({'error': 'Leaf Dieback model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image_leaf_dieback(image_bytes)

        # 3-class classification: softmax output
        # Classes: ['healthy', 'leaf_die_back', 'not_cocount']
        predictions = models['leaf_dieback'].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = LEAF_DIEBACK_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_leaf_dieback = predicted_class == 'leaf_die_back'
        is_healthy = predicted_class == 'healthy'
        is_valid = predicted_class != 'not_cocount'

        probabilities = {
            'healthy': float(predictions[0]),
            'leaf_die_back': float(predictions[1]),
            'not_coconut': float(predictions[2])
        }

        # Determine label and message
        if not is_valid:
            label = 'Not a valid baby coconut leaf image'
            message = 'The uploaded image does not appear to be a baby coconut leaf. Please upload a clear image of a young coconut palm leaf.'
            status = 'invalid'
        elif is_leaf_dieback:
            label = 'Leaf Dieback Disease Detected'
            message = 'This baby coconut leaf shows signs of Leaf Dieback disease. This disease can severely affect young coconut palms. Immediate treatment is recommended.'
            status = 'diseased'
        else:
            label = 'Healthy Baby Coconut Leaf'
            message = 'No disease detected. This baby coconut leaf appears to be healthy.'
            status = 'healthy'

        return jsonify({
            'success': True,
            'detection_type': 'leaf_dieback',
            'model_version': 'v4',
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_diseased': is_leaf_dieback,
                'is_leaf_dieback': is_leaf_dieback,
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

@app.route('/predict/leaf-health', methods=['POST'])
def predict_leaf_health():
    """Detect coconut leaf health (2-class classification)

    Supports two modes:
    - phone: Uses leaf_health_v4 model (optimized for phone camera images)
    - drone: Uses leaf_health_v3 model (optimized for drone aerial images)

    Pass mode as form field or query parameter. Default is 'phone'.
    Classes: healthy, unhealthy
    """

    # Get mode from form data or query parameter (default: phone)
    mode = request.form.get('mode', request.args.get('mode', 'phone')).lower()

    # Select model based on mode
    if mode == 'drone':
        model_key = 'leaf_health_v3'
        model_version = 'v3_drone'
    else:
        model_key = 'leaf_health_v4'
        model_version = 'v4_phone'

    if models.get(model_key) is None:
        return jsonify({'error': f'Leaf Health model ({model_version}) not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()

        # Preprocess image (224x224, 0-1 scaling)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        processed_image = np.expand_dims(img_array, axis=0)

        # 2-class classification: softmax output
        # Classes: ['healthy', 'unhealthy']
        predictions = models[model_key].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = LEAF_HEALTH_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_healthy = predicted_class == 'healthy'

        probabilities = {
            'healthy': float(predictions[0]),
            'unhealthy': float(predictions[1])
        }

        # Determine message and recommendation
        if is_healthy:
            message = 'This coconut leaf appears to be healthy with no visible signs of stress or disease.'
            recommendation = 'Continue with regular maintenance and monitoring. Ensure proper watering and nutrition.'
        else:
            message = 'This coconut leaf shows signs of being unhealthy. It may have yellowing, wilting, or other stress indicators.'
            recommendation = 'Inspect the tree for pest damage, nutrient deficiencies, or water stress. Consider consulting an agricultural expert.'

        # Possible conditions for unhealthy leaves
        possible_conditions = []
        if not is_healthy:
            possible_conditions = [
                {
                    'condition': 'Nitrogen Deficiency',
                    'icon': '🌿',
                    'urgency': 'medium',
                    'reason': 'Lack of nitrogen in soil affects chlorophyll production',
                    'symptoms': ['Yellowing of older leaves', 'Stunted growth', 'Pale green color'],
                    'solution': 'Apply nitrogen-rich fertilizer (urea or ammonium sulfate) at 500g per tree'
                },
                {
                    'condition': 'Potassium Deficiency',
                    'icon': '🍂',
                    'urgency': 'high',
                    'reason': 'Potassium is essential for fruit development and disease resistance',
                    'symptoms': ['Orange/yellow spotting on leaflets', 'Leaf tip necrosis', 'Reduced yield'],
                    'solution': 'Apply potassium chloride (muriate of potash) at 1-2kg per tree annually'
                },
                {
                    'condition': 'Magnesium Deficiency',
                    'icon': '💛',
                    'urgency': 'medium',
                    'reason': 'Magnesium is central to chlorophyll molecule',
                    'symptoms': ['Interveinal yellowing', 'Yellow bands along leaflets', 'Older leaves affected first'],
                    'solution': 'Apply magnesium sulfate (Epsom salt) at 500g per tree or spray 2% solution'
                },
                {
                    'condition': 'Water Stress',
                    'icon': '💧',
                    'urgency': 'high',
                    'reason': 'Insufficient or excessive water affects nutrient uptake',
                    'symptoms': ['Wilting leaves', 'Yellowing', 'Leaf drooping'],
                    'solution': 'Ensure proper irrigation (40-50 liters per day during dry season)'
                },
                {
                    'condition': 'Pest Damage',
                    'icon': '🐛',
                    'urgency': 'high',
                    'reason': 'Insects feeding on leaves cause physical damage and nutrient loss',
                    'symptoms': ['Holes in leaves', 'Discoloration', 'Webbing or insects visible'],
                    'solution': 'Identify specific pest and apply appropriate treatment. Consider neem oil spray.'
                }
            ]

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'is_healthy': is_healthy,
            'probabilities': probabilities,
            'message': message,
            'recommendation': recommendation,
            'possible_conditions': possible_conditions,
            'conditions_count': len(possible_conditions),
            'mode': mode,
            'model_info': {
                'version': model_version,
                'accuracy': model_infos.get(model_key, {}).get('performance', {}).get('test_accuracy', 'N/A')
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/tree-health', methods=['POST'])
def predict_tree_health():
    """Detect overall coconut tree health (v1 model - 2-class classification)

    Classes: healthy, unhealthy
    Analyzes images of full coconut trees to determine overall health status.
    """

    if models.get('tree_health') is None:
        return jsonify({'error': 'Tree Health model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()

        # Preprocess image (same as other models - 224x224, 0-1 scaling)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        processed_image = np.expand_dims(img_array, axis=0)

        # 2-class classification: softmax output
        # Classes: ['healthy', 'unhealthy']
        predictions = models['tree_health'].predict(processed_image, verbose=0)[0]

        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_class = TREE_HEALTH_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        is_healthy = predicted_class == 'healthy'
        is_unhealthy = predicted_class == 'unhealthy'

        # Calculate unhealthy percentage
        unhealthy_percentage = float(predictions[1]) * 100

        probabilities = {
            'healthy': float(predictions[0]),
            'unhealthy': float(predictions[1])
        }

        # Determine label and message
        if is_healthy:
            label = 'Healthy Coconut Tree'
            message = 'This coconut tree appears to be in good health. No significant issues detected.'
            status = 'healthy'
            recommendation = 'Continue regular maintenance and monitoring. Ensure proper watering and fertilization.'
        else:
            label = 'Unhealthy Coconut Tree'
            message = f'This coconut tree shows signs of poor health ({unhealthy_percentage:.1f}% unhealthy indicators detected).'
            status = 'unhealthy'
            recommendation = 'Inspect the tree for pest infestations, nutrient deficiencies, or diseases. Consider consulting an agricultural expert.'

        return jsonify({
            'success': True,
            'detection_type': 'tree_health',
            'model_version': 'v1',
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'unhealthy_percentage': unhealthy_percentage,
            'is_healthy': is_healthy,
            'message': message,
            'label': label,
            'status': status,
            'recommendation': recommendation,
            'model_info': {
                'name': 'Coconut Tree Health Model',
                'version': 'v2',
                'accuracy': '100%'
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/bunch', methods=['POST'])
def predict_bunch():
    """Detect coconut bunches for yield prediction (TFLite YOLOv8)

    Accepts 1 or 2 images from opposite sides of the tree.
    Returns total bunch count and detection details.
    """

    if models.get('bunch') is None:
        return jsonify({'success': False, 'error': 'Bunch detection model not loaded'}), 500

    if 'image1' not in request.files and 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    try:
        results = []
        total_bunches = 0
        total_confidence = 0

        # Process image1 (or 'image' for backward compatibility)
        file1 = request.files.get('image1') or request.files.get('image')
        if file1 and file1.filename:
            image_bytes1 = file1.read()
            processed1, original_size1 = preprocess_image_yolo(image_bytes1, BUNCH_INPUT_SIZE)

            output1 = run_tflite_inference(models['bunch'], processed1)
            detections1 = parse_yolo_output(
                output1,
                conf_threshold=BUNCH_CONFIDENCE_THRESHOLD,
                iou_threshold=BUNCH_IOU_THRESHOLD,
                max_detections=BUNCH_MAX_DETECTIONS
            )

            bunch_count1 = len(detections1)
            avg_conf1 = sum(d['confidence'] for d in detections1) / bunch_count1 if bunch_count1 > 0 else 0

            results.append({
                'image': 'image1',
                'bunch_count': bunch_count1,
                'average_confidence': avg_conf1,
                'detections': detections1
            })
            total_bunches += bunch_count1
            total_confidence += avg_conf1

        # Process image2 (optional)
        file2 = request.files.get('image2')
        if file2 and file2.filename:
            image_bytes2 = file2.read()
            processed2, original_size2 = preprocess_image_yolo(image_bytes2, BUNCH_INPUT_SIZE)

            output2 = run_tflite_inference(models['bunch'], processed2)
            detections2 = parse_yolo_output(
                output2,
                conf_threshold=BUNCH_CONFIDENCE_THRESHOLD,
                iou_threshold=BUNCH_IOU_THRESHOLD,
                max_detections=BUNCH_MAX_DETECTIONS
            )

            bunch_count2 = len(detections2)
            avg_conf2 = sum(d['confidence'] for d in detections2) / bunch_count2 if bunch_count2 > 0 else 0

            results.append({
                'image': 'image2',
                'bunch_count': bunch_count2,
                'average_confidence': avg_conf2,
                'detections': detections2
            })
            total_bunches += bunch_count2
            total_confidence += avg_conf2

        images_processed = len(results)
        average_confidence = total_confidence / images_processed if images_processed > 0 else 0

        # Generate message and recommendation
        if total_bunches == 0:
            message = 'No coconut bunches detected in the image(s).'
            recommendation = 'Make sure the image clearly shows the coconut tree crown with bunches visible.'
        elif total_bunches < 5:
            message = f'Detected {total_bunches} bunch(es). Low yield expected.'
            recommendation = 'Consider checking tree health and nutrition for better yield.'
        elif total_bunches < 10:
            message = f'Detected {total_bunches} bunches. Moderate yield expected.'
            recommendation = 'Tree appears to be producing normally. Continue regular care.'
        else:
            message = f'Detected {total_bunches} bunches. Good yield expected!'
            recommendation = 'Excellent bunch count! Ensure proper support for heavy bunches.'

        return jsonify({
            'success': True,
            'total_bunch_count': total_bunches,
            'average_confidence': average_confidence,
            'images_processed': images_processed,
            'results': results,
            'message': message,
            'recommendation': recommendation,
            'model_info': {
                'name': 'Bunch Detection Model',
                'version': 'v1',
                'format': 'TFLite YOLOv8'
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/yield', methods=['POST'])
def predict_yield():
    """Estimate coconut yield by counting individual nuts (TFLite YOLOv8)

    Detects individual coconut nuts on the tree for yield estimation.
    Returns total nut count and detection details.
    """

    if models.get('yield') is None:
        return jsonify({'success': False, 'error': 'Yield estimator model not loaded'}), 500

    if 'image1' not in request.files and 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    try:
        results = []
        total_nuts = 0
        total_confidence = 0

        # Process image1 (or 'image' for backward compatibility)
        file1 = request.files.get('image1') or request.files.get('image')
        if file1 and file1.filename:
            image_bytes1 = file1.read()
            processed1, original_size1 = preprocess_image_yolo(image_bytes1, YIELD_INPUT_SIZE)

            output1 = run_tflite_inference(models['yield'], processed1)
            detections1 = parse_yolo_output(
                output1,
                conf_threshold=YIELD_CONFIDENCE_THRESHOLD,
                iou_threshold=YIELD_IOU_THRESHOLD,
                max_detections=YIELD_MAX_DETECTIONS
            )

            nut_count1 = len(detections1)
            avg_conf1 = sum(d['confidence'] for d in detections1) / nut_count1 if nut_count1 > 0 else 0

            results.append({
                'image': 'image1',
                'nut_count': nut_count1,
                'average_confidence': avg_conf1,
                'detections': detections1
            })
            total_nuts += nut_count1
            total_confidence += avg_conf1

        # Process image2 (optional)
        file2 = request.files.get('image2')
        if file2 and file2.filename:
            image_bytes2 = file2.read()
            processed2, original_size2 = preprocess_image_yolo(image_bytes2, YIELD_INPUT_SIZE)

            output2 = run_tflite_inference(models['yield'], processed2)
            detections2 = parse_yolo_output(
                output2,
                conf_threshold=YIELD_CONFIDENCE_THRESHOLD,
                iou_threshold=YIELD_IOU_THRESHOLD,
                max_detections=YIELD_MAX_DETECTIONS
            )

            nut_count2 = len(detections2)
            avg_conf2 = sum(d['confidence'] for d in detections2) / nut_count2 if nut_count2 > 0 else 0

            results.append({
                'image': 'image2',
                'nut_count': nut_count2,
                'average_confidence': avg_conf2,
                'detections': detections2
            })
            total_nuts += nut_count2
            total_confidence += avg_conf2

        images_processed = len(results)
        average_confidence = total_confidence / images_processed if images_processed > 0 else 0

        # Estimate yield (assume 2 sides captured, so don't double count)
        # If only 1 image, estimate total by multiplying by 1.5
        if images_processed == 1:
            estimated_total = int(total_nuts * 1.5)
            estimation_note = 'Estimated from single image (x1.5)'
        else:
            estimated_total = total_nuts
            estimation_note = 'Count from both sides of tree'

        # Generate message and recommendation
        if total_nuts == 0:
            message = 'No coconut nuts detected in the image(s).'
            recommendation = 'Make sure the image clearly shows coconuts on the tree.'
            yield_category = 'none'
        elif estimated_total < 20:
            message = f'Detected {total_nuts} nuts. Estimated total: {estimated_total} nuts.'
            recommendation = 'Low yield. Check tree health, nutrition, and pollination.'
            yield_category = 'low'
        elif estimated_total < 50:
            message = f'Detected {total_nuts} nuts. Estimated total: {estimated_total} nuts.'
            recommendation = 'Moderate yield. Tree is producing normally.'
            yield_category = 'moderate'
        elif estimated_total < 100:
            message = f'Detected {total_nuts} nuts. Estimated total: {estimated_total} nuts.'
            recommendation = 'Good yield! Tree is healthy and productive.'
            yield_category = 'good'
        else:
            message = f'Detected {total_nuts} nuts. Estimated total: {estimated_total} nuts.'
            recommendation = 'Excellent yield! This is a highly productive tree.'
            yield_category = 'excellent'

        return jsonify({
            'success': True,
            'total_nut_count': total_nuts,
            'estimated_total': estimated_total,
            'estimation_note': estimation_note,
            'yield_category': yield_category,
            'average_confidence': average_confidence,
            'images_processed': images_processed,
            'results': results,
            'message': message,
            'recommendation': recommendation,
            'model_info': {
                'name': 'Coconut Yield Estimator',
                'version': 'v1',
                'format': 'TFLite YOLOv8',
                'detects': 'Individual coconut nuts'
            },
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

# Load models on startup (works for both `python app.py` and gunicorn/HF Spaces)
print("\n[STARTUP] Loading models for Coconut Health Monitor ML API v9.0...")
load_models()
print("[STARTUP] Models loaded - API ready!\n")

if __name__ == '__main__':
    print("=" * 60)
    print("  Mite Model: v10 (3-class, 91.44% accuracy) - /predict/mite")
    print("  Mite Model: v12 (2-class, 97.44% accuracy) - loaded but unused")
    print("  Unified Model: v1 (4-class - caterpillar + white_fly, 96.08% accuracy)")
    print("  Disease Model: v2 (4-class - Leaf Rot, Leaf Spot, 98.69% accuracy)")
    print("  Leaf Dieback Model: v4 (3-class - baby coconut disease)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=False)
