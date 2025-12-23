# Coconut Health Monitor

AI-powered drone-based coconut tree health monitoring and yield prediction system.

**SLIIT Research Project**

## Overview

A React Native mobile application that uses machine learning models to detect pests and diseases in coconut trees. The system is designed to work with drone-captured images for large-scale plantation monitoring.

## Features

- Coconut Mite Detection (EfficientNetB0)
- Coconut Caterpillar Detection (MobileNetV2)
- Real-time pest prediction via Flask API
- Mobile-optimized ML inference
- Firebase Authentication with Google Sign-In

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

## ML Models

### Coconut Mite Detection (v7)
- **Architecture:** EfficientNetB0 (Transfer Learning)
- **Accuracy:** 82.54%
- **F1 Score:** 82.47%
- **Classes:** coconut_mite, healthy
- **Optimizations:** Anti-overfit (Dropout 0.6, L2 0.02, Label Smoothing 0.1)

### Coconut Caterpillar Detection
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Accuracy:** 98.91%
- **F1 Score:** 98.91%
- **Classes:** caterpillar, healthy
- **Dataset:** 9,108 images

## Project Structure

```
Research/
├── ml/
│   ├── api/                    # Flask API server
│   │   ├── app.py              # API routes
│   │   └── run_api.py          # Server launcher
│   ├── models/
│   │   ├── coconut_mite_v7/    # Latest mite model
│   │   └── coconut_caterpillar/ # Caterpillar model
│   └── notebooks/
│       ├── 02_coconut_caterpillar_training.ipynb
│       ├── 03_coconut_mite_training.ipynb
│       ├── 04_coconut_mite_proper_training.ipynb
│       ├── 05_mite_model_results.ipynb
│       └── 06_mite_v7_results.ipynb
├── src/
│   ├── screens/                # React Native screens
│   └── services/               # API clients
├── android/                    # Android native code
└── App.tsx                     # App entry point
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
python run_api.py
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/mite` | POST | Detect coconut mite |
| `/predict/caterpillar` | POST | Detect caterpillar |
| `/health` | GET | API health check |

## Model Training

Jupyter notebooks are available in `ml/notebooks/`:

1. **02_coconut_caterpillar_training.ipynb** - Caterpillar model training
2. **03_coconut_mite_training.ipynb** - Mite model v4-v5 training
3. **04_coconut_mite_proper_training.ipynb** - Proper training pipeline
4. **05_mite_model_results.ipynb** - v6 results analysis
5. **06_mite_v7_results.ipynb** - v7 anti-overfit results

## Model Version History

| Version | Accuracy | F1 Score | Train-Val Gap | Status |
|---------|----------|----------|---------------|--------|
| Mite v4 | ~90% | - | High | Deprecated |
| Mite v5 | ~92% | - | High | Deprecated |
| Mite v6 | ~85% | - | Medium | Deprecated |
| Mite v7 | 82.54% | 82.47% | 6.2% | **Current** |
| Caterpillar | 98.91% | 98.91% | - | **Current** |

## Team

SLIIT Research Team

## License

Research Project - SLIIT
