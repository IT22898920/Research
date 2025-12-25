# Coconut Health Monitor

AI-powered drone-based coconut tree health monitoring and yield prediction system.

**SLIIT Research Project**

## Overview

A React Native mobile application that uses machine learning models to detect pests and diseases in coconut trees. The system is designed to work with drone-captured images for large-scale plantation monitoring.

## Features

- **Coconut Mite Detection** - EfficientNetB0 model (91.44% accuracy)
- **Coconut Caterpillar Detection** - MobileNetV2 model (97.47% accuracy)
- **Smart All Pests Detection** - Combined analysis with intelligent decision logic
- **Non-Coconut Image Rejection** - Filters out invalid images automatically
- **Real-time Pest Prediction** - Flask API for instant results
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

### Coconut Caterpillar Detection (v2)
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Accuracy:** 97.47%
- **Macro F1 Score:** 96.30%
- **Classes:** `caterpillar`, `healthy`, `not_coconut`

## Project Structure

```
Research/
├── ml/
│   ├── api/                         # Flask API server (v4.0)
│   │   ├── app.py                   # API routes with smart logic
│   │   └── test_api.py              # API tests
│   ├── models/
│   │   ├── coconut_mite_v10/        # Latest mite model (3-class)
│   │   └── coconut_caterpillar_v2/  # Latest caterpillar model (3-class)
│   └── notebooks/
│       ├── 07_mite_3class_training.ipynb
│       ├── 09_mite_v10_final_results.ipynb
│       └── 10_caterpillar_v2_final_results.ipynb
├── src/
│   ├── screens/
│   │   └── PestDetectionScreen.js   # Main pest detection UI
│   └── services/
│       └── pestDetectionApi.js      # API client (v4.0)
├── android/                         # Android native code
└── App.tsx                          # App entry point
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

## API Endpoints (v4.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check with model status |
| `/models` | GET | List all loaded models |
| `/predict/mite` | POST | Detect coconut mite (3-class) |
| `/predict/caterpillar` | POST | Detect caterpillar damage (3-class) |
| `/predict/all` | POST | Smart combined pest detection |

### Smart Combined Logic

The `/predict/all` endpoint uses v7 decision logic:

```
Valid Detection = (healthy OR pest) WITH >40% confidence

Rules:
- Accept if ANY model confidently detects valid coconut class (>40%)
- Reject only if NO model finds valid detection
```

**Example Scenarios:**
| Image Type | Mite Says | Caterpillar Says | Result |
|------------|-----------|------------------|--------|
| Healthy leaf | not_coconut | healthy 98% | Healthy |
| Infected coconut | mite 55% | not_coconut | Mite Infected |
| Garden scene | mite 39% | not_coconut 99% | Not Valid |

## Model Training

Jupyter notebooks in `ml/notebooks/`:

| Notebook | Description |
|----------|-------------|
| `07_mite_3class_training.ipynb` | Mite v8-v10 training |
| `09_mite_v10_final_results.ipynb` | Mite v10 evaluation |
| `10_caterpillar_v2_final_results.ipynb` | Caterpillar v2 evaluation |

## Model Version History

### Coconut Mite
| Version | Accuracy | Classes | Status |
|---------|----------|---------|--------|
| v4-v7 | 82-92% | 2 | Deprecated |
| v8-v9 | ~88% | 3 | Deprecated |
| **v10** | **91.44%** | **3** | **Current** |

### Coconut Caterpillar
| Version | Accuracy | Classes | Status |
|---------|----------|---------|--------|
| v1 | 98.91% | 2 | Deprecated |
| **v2** | **97.47%** | **3** | **Current** |

## API Response Format

### Single Pest Detection (`/predict/mite` or `/predict/caterpillar`)
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
    "mite": { "class": "coconut_mite", "confidence": 0.55 },
    "caterpillar": { "class": "not_coconut", "confidence": 0.65 }
  },
  "summary": {
    "is_valid_image": true,
    "is_healthy": false,
    "label": "Coconut Mite Infected",
    "pests_detected": ["Coconut Mite"],
    "recommendation": "Apply mite treatment spray and monitor affected trees."
  }
}
```

## Team

SLIIT Research Team

## License

Research Project - SLIIT
