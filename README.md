# Coconut Health Monitor

AI-powered drone-based coconut tree health monitoring and yield prediction system.

**SLIIT Research Project**

## Overview

A React Native mobile application that uses machine learning models to detect pests and diseases in coconut trees. The system is designed to work with drone-captured images for large-scale plantation monitoring.

## Features

- **Coconut Mite Detection** - EfficientNetB0 model (91.44% accuracy)
- **Coconut Caterpillar Detection** - Unified MobileNetV2 model (96.08% accuracy)
- **White Fly Detection** - Unified MobileNetV2 model (96.08% accuracy)
- **Smart All Pests Detection** - Combined analysis with cross-validation logic
- **Non-Coconut Image Rejection** - Filters out invalid images automatically
- **Cross-Validation** - Prevents false positives by validating between models
- **Real-time Pest Prediction** - Flask API v6.0 for instant results
- **Firebase Authentication** - Secure login with Google Sign-In

## Tech Stack

### Mobile App
| Technology | Version |
|------------|---------|
| React Native | 0.82.1 |
| React Navigation | 7.x |
| Firebase Auth | Latest |
| Google Sign-In | Latest |

### Machine Learning
| Technology | Version |
|------------|---------|
| Python | 3.13 |
| TensorFlow | 2.20.0 |
| Flask | 3.1.2 |
| Focal Loss | Custom implementation |

## ML Models

### Coconut Mite Detection (v10)
- **Architecture:** EfficientNetB0 (Transfer Learning)
- **Accuracy:** 91.44%
- **Classes:** `coconut_mite`, `healthy`, `not_coconut`
- **Features:** Focal Loss, Mite boost factor, Non-coconut rejection

### Unified Caterpillar & White Fly Detection (v1)
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Accuracy:** 96.08%
- **Macro F1 Score:** 92.38%
- **Classes:** `caterpillar`, `healthy`, `not_coconut`, `white_fly`
- **Features:** Focal Loss, 4-class classification, Cross-validation with Mite model

## Project Structure

```
Research/
├── ml/
│   ├── api/                                    # Flask API server (v6.0)
│   │   ├── app.py                              # API routes with cross-validation
│   │   └── test_api.py                         # API tests
│   ├── models/
│   │   ├── coconut_mite_v10/                   # Mite model (3-class)
│   │   └── unified_caterpillar_whitefly_v1/    # Unified model (4-class)
│   └── notebooks/
│       ├── 09_mite_v10_final_results.ipynb
│       └── 12_unified_caterpillar_whitefly_v1.ipynb
├── src/
│   ├── screens/
│   │   └── PestDetectionScreen.js              # Main pest detection UI
│   └── services/
│       └── pestDetectionApi.js                 # API client (v5.0)
├── android/                                    # Android native code
└── App.tsx                                     # App entry point
```

## Quick Start

### Run Mobile App
```powershell
# Start Metro bundler
node node_modules\@react-native-community\cli\build\bin.js start

# Run on Android (another terminal)
node node_modules\@react-native-community\cli\build\bin.js run-android
```

### Run ML API
```powershell
cd ml/api
python app.py
```

API will start on `http://localhost:5001`

## API Endpoints (v6.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check with model status |
| `/models` | GET | List all loaded models |
| `/predict/mite` | POST | Detect coconut mite (3-class) |
| `/predict/caterpillar` | POST | Detect caterpillar damage (4-class unified) |
| `/predict/white_fly` | POST | Detect white fly damage (4-class unified) |
| `/predict/unified` | POST | Unified model detection (4-class) |
| `/predict/all` | POST | Smart combined pest detection with cross-validation |

### Smart Combined Logic with Cross-Validation

The `/predict/all` endpoint uses intelligent decision logic:

```
Valid Detection = (healthy OR pest) WITH >40% confidence

Cross-Validation:
- If unified model says "healthy" >80% confidence, ignore mite false positives
- Prevents mite model errors from overriding correct unified predictions

Rules:
- Accept if ANY model confidently detects valid coconut class (>40%)
- Reject only if NO model finds valid detection
```

**Example Scenarios:**
| Image Type | Mite Says | Unified Says | Result |
|------------|-----------|--------------|--------|
| Healthy leaf | mite 30% | healthy 96% | ✅ Healthy (cross-validation) |
| Healthy leaf | not_coconut | healthy 98% | ✅ Healthy |
| Mite infected | mite 55% | not_coconut | ✅ Mite Infected |
| Caterpillar | not_coconut | caterpillar 95% | ✅ Caterpillar |
| White Fly | not_coconut | white_fly 92% | ✅ White Fly |
| Garden scene | mite 39% | not_coconut 99% | ❌ Not Valid |

## Model Training

Jupyter notebooks in `ml/notebooks/`:

| Notebook | Description |
|----------|-------------|
| `09_mite_v10_final_results.ipynb` | Mite v10 evaluation |
| `12_unified_caterpillar_whitefly_v1.ipynb` | Unified model training & evaluation |

## Model Version History

### Coconut Mite
| Version | Accuracy | Classes | Status |
|---------|----------|---------|--------|
| v4-v7 | 82-92% | 2 | Deprecated |
| v8-v9 | ~88% | 3 | Deprecated |
| **v10** | **91.44%** | **3** | **Current** |

### Unified Caterpillar & White Fly
| Version | Accuracy | Classes | Status |
|---------|----------|---------|--------|
| **v1** | **96.08%** | **4** | **Current** |

## API Response Format

### Single Pest Detection (`/predict/mite`, `/predict/caterpillar`, `/predict/white_fly`)
```json
{
  "success": true,
  "prediction": {
    "class": "coconut_mite",
    "confidence": 0.85,
    "is_infected": true,
    "is_valid_image": true,
    "label": "Coconut Mite Infected",
    "message": "This coconut shows signs of mite infection."
  },
  "probabilities": {
    "coconut_mite": 0.85,
    "healthy": 0.10,
    "not_coconut": 0.05
  }
}
```

### All Pests Detection (`/predict/all`)
```json
{
  "success": true,
  "results": {
    "mite": { "class": "healthy", "confidence": 0.96 },
    "caterpillar": { "class": "healthy", "confidence": 0.96 },
    "white_fly": { "class": "healthy", "confidence": 0.96 }
  },
  "summary": {
    "is_valid_image": true,
    "is_healthy": true,
    "status": "Healthy",
    "label": "Healthy Coconut",
    "pests_detected": [],
    "recommendation": "No treatment needed. Continue regular monitoring."
  }
}
```

## Team

SLIIT Research Team

## License

Research Project - SLIIT
